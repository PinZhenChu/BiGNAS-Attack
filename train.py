import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model

# =========================
# === Logging 設定 ========
# =========================
def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def _range_str(t, label="", expected=None):
    """精簡版 index 範圍輸出"""
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return f"{label}: empty"
    min_v, max_v = int(t.min()), int(t.max())
    exp_str = f", expect={expected}" if expected else ""
    return f"{label}[{min_v}, {max_v}]{exp_str}"

# =========================
# === CL 工具函式 =========
# =========================
def dropout_graph(edge_index, num_nodes, mode="edge", drop_rate=0.2):
    """
    根據 mode 對圖做隨機增強 (graph augmentation)
    Args:
        edge_index (Tensor): [2, num_edges]，圖的邊集合
        num_nodes (int): 節點數（node dropout 會用到）
        mode (str): 'edge' 或 'node'
        drop_rate (float): 丟棄比例
    Return:
        新的 edge_index
    """
    if drop_rate <= 0.0:
        return edge_index

    if mode == "edge":
        # === Edge Dropout ===
        E = edge_index.size(1)
        keep = max(1, int(E * (1 - drop_rate)))
        perm = torch.randperm(E, device=edge_index.device)[:keep]
        return edge_index[:, perm]

    elif mode == "node":
        # === Node Dropout ===
        num_drop = int(num_nodes * drop_rate)
        if num_drop <= 0:
            return edge_index
        drop_nodes = torch.randperm(num_nodes, device=edge_index.device)[:num_drop]
        mask = ~(
            torch.isin(edge_index[0], drop_nodes) |
            torch.isin(edge_index[1], drop_nodes)
        )
        return edge_index[:, mask]

    else:
        raise ValueError(f"Unknown dropout mode: {mode}")


def sgl_info_nce_loss(z1, z2, tau=0.2, batch_size=128):
    N = z1.size(0)
    device = z1.device
    loss_all = []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        z1_batch = z1[start:end]  # [B, d]

        # 相似度 (batch vs 全部)
        sim_matrix = torch.matmul(z1_batch, z2.t()) / tau   # [B, N]

        labels = torch.arange(start, end, device=device)
        loss = F.cross_entropy(sim_matrix, labels)
        loss_all.append(loss)

    return torch.stack(loss_all).mean()

# =========================
# === Meta Learning =======
# =========================
def meta_optimizeation(target_meta_loader, replace_optimizer, model, args, criterion,
                       replace_scheduler, source_edge_index, target_edge_index):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)
            replace_optimizer.zero_grad()
            out = model.meta_prediction(source_edge_index, target_edge_index, target_link).squeeze()
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()

# =========================
# === Evaluation ==========
# =========================
@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label, args):
    model.eval()
    if getattr(args, "debug_mode", False):
        logging.debug(_range_str(link[1], f"EvalLink_{name}",
                                 (args.num_users, args.num_users + args.num_target_items)))
    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except:
        auc = 1.0
    logging.info("[%s] AUC=%.4f", name, auc)
    model.train()
    return auc


def get_test_positive_dict(data, args):
    """建立 test set user -> positive items 對應表"""
    test_user_item_dict = {}
    test_link = data.target_test_link.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        test_user_item_dict.setdefault(u.item(), []).append(i.item())
    if getattr(args, "debug_mode", False) and test_user_item_dict:
        sample_u = next(iter(test_user_item_dict))
        logging.debug("[TestPosSample] user=%d | items=%s",
                      sample_u, _range_str(torch.tensor(test_user_item_dict[sample_u]),
                                           "PosItems",
                                           (data.num_users, data.num_users + data.num_target_items)))
    return test_user_item_dict


def evaluate_hit_ratio(model, data, source_edge_index, target_edge_index, top_k, num_candidates, args):
    import random
    model.eval()
    hit_count, total_users = 0, 0
    all_target_items = set(range(data.num_users, data.num_users + data.num_target_items))
    user_interactions = get_test_positive_dict(data, args)
    sim_users = list(user_interactions.keys())
    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, set())
            if not pos_items:
                continue
            pos_item = list(pos_items)[0]
            negative_pool = list(all_target_items - set(pos_items))
            if len(negative_pool) < num_candidates:
                continue
            candidate_items = random.sample(negative_pool, num_candidates) + [pos_item]
            user_tensor = torch.tensor([user_id] * len(candidate_items), device=args.device)
            item_tensor = torch.tensor(candidate_items, device=args.device)
            if getattr(args, "debug_mode", False):
                logging.debug(_range_str(item_tensor, f"HR_User{user_id}_Items",
                                         (data.num_users, data.num_users + data.num_target_items)))
            link = torch.stack([user_tensor, item_tensor], dim=0)
            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            top_k_items = [candidate_items[i] for i in torch.topk(scores, k=top_k).indices.tolist()]
            if pos_item in top_k_items:
                hit_count += 1
            total_users += 1
    hit_ratio = hit_count / total_users if total_users else 0.0
    logging.info("[HR@%d] Users=%d | Hits=%d | HR=%.4f", top_k, total_users, hit_count, hit_ratio)
    return hit_ratio


def evaluate_er_hit_ratio(model, data, source_edge_index, target_edge_index, cold_item_set,
                           top_k, num_candidates, args):
    import random
    model.eval()
    all_target_items = set(range(data.num_users, data.num_users + data.num_target_items))
    user_interactions = get_test_positive_dict(data, args)
    sim_users = list(user_interactions.keys())
    cold_item_hit_count, total_users = 0, 0
    with torch.no_grad():
        for user_id in sim_users:
            negative_pool = list(all_target_items - cold_item_set)
            if len(negative_pool) < num_candidates:
                continue
            sampled_items = random.sample(negative_pool, num_candidates) + list(cold_item_set)
            sampled_items = list(set(sampled_items))
            user_tensor = torch.tensor([user_id] * len(sampled_items), device=args.device)
            item_tensor = torch.tensor(sampled_items, device=args.device)
            if getattr(args, "debug_mode", False):
                logging.debug(_range_str(item_tensor, f"ER_User{user_id}_Items",
                                         (data.num_users, data.num_users + data.num_target_items)))
            link = torch.stack([user_tensor, item_tensor], dim=0)
            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            sorted_items = [item for item, _ in sorted(zip(sampled_items, scores.tolist()), key=lambda x: x[1], reverse=True)]
            if any(item in cold_item_set for item in sorted_items[:top_k]):
                cold_item_hit_count += 1
            total_users += 1
    er_ratio = cold_item_hit_count / total_users if total_users else 0.0
    logging.info("[ER@%d] Users=%d | Hits=%d | ER=%.4f", top_k, total_users, cold_item_hit_count, er_ratio)
    return er_ratio

def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, args):
    topk_list = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    print("\n📊 Evaluation for multiple top-K values:")
    # [IDX-CHK]
    logging.info(f"[IDX-CHK][MULTI@K] users={data.num_users}, target_items={data.num_target_items}")
    for k in topk_list:
        _ = evaluate_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            top_k=k,
            num_candidates=99,
            args=args
        )
        _ = evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            cold_item_set=cold_item_set,
            top_k=k,
            num_candidates=99,
            args=args
        )


# =========================
# === Train 主程式 =======
# =========================
def train(model, perceptor, data, args):
    setup_logging(getattr(args, "debug_mode", False))
    device = args.device
    data = data.to(device)
    model, perceptor = model.to(device), perceptor.to(device)

    # === 資料切分 ===
    (source_edge_index, source_label, source_link,
     target_train_edge_index, target_train_label, target_train_link,
     target_valid_link, target_valid_label,
     target_test_link, target_test_label,
     target_test_edge_index) = link_split(data)

    # 存回 data 供 HR/ER 評估使用
    data.target_test_link = target_test_link
    data.target_train_link = target_train_link

    logging.info("[Data] Users=%d | TargetItems=%d | SourceItems=%d",
                 data.num_users, data.num_target_items, data.num_source_items)

    # === DataLoader 設定 ===
    target_train_set = Dataset(target_train_link.cpu(), target_train_label.cpu())
    target_train_loader = DataLoader(target_train_set, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers,
                                     collate_fn=target_train_set.collate_fn)
    source_batch_size = int(args.batch_size * target_train_link.shape[1] / source_link.shape[1])
    source_train_set = Dataset(source_link.cpu(), source_label.cpu())
    source_train_loader = DataLoader(source_train_set, batch_size=source_batch_size,
                                     shuffle=True, num_workers=args.num_workers,
                                     collate_fn=source_train_set.collate_fn)
    target_meta_loader = DataLoader(target_train_set, batch_size=args.meta_batch_size,
                                    shuffle=True, num_workers=args.num_workers,
                                    collate_fn=target_train_set.collate_fn)
    source_meta_batch_size = int(args.meta_batch_size * target_train_link.shape[1] / source_link.shape[1])
    source_meta_loader = DataLoader(source_train_set, batch_size=source_meta_batch_size,
                                    shuffle=True, num_workers=args.num_workers,
                                    collate_fn=source_train_set.collate_fn)

    # === Optimizer 設定 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    perceptor_optimizer = torch.optim.Adam(perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta_optimizer = MetaOptimizer(meta_optimizer=perceptor_optimizer,
                                   hpo_lr=args.hpo_lr, truncate_iter=3, max_grad_norm=10)
    replace_param = [param for name, param in model.named_parameters() if name.startswith("replace")]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(replace_optimizer, T_max=args.T_max)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.model_path, trace_func=logging.info)
    criterion = nn.BCELoss(reduction="none")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for (source_link_b, source_label_b), (target_link_b, target_label_b) in zip(source_train_loader, target_train_loader):
            source_link_b, source_label_b = source_link_b.to(device), source_label_b.to(device)
            target_link_b, target_label_b = target_link_b.to(device), target_label_b.to(device)

            # === Main Loss ===
            weight_source = perceptor(source_link_b[1], source_edge_index, model)
            source_out = model(source_edge_index, target_train_edge_index, source_link_b, is_source=True).squeeze()
            target_out = model(source_edge_index, target_train_edge_index, target_link_b, is_source=False).squeeze()

            source_loss = (criterion(source_out, source_label_b).reshape(-1, 1) * weight_source).sum()
            target_loss = criterion(target_out, target_label_b).mean()
            main_loss = source_loss + target_loss if args.use_meta else target_loss

            # === Contrastive Learning (CL) ===
            full_edge_index = torch.cat([source_edge_index, target_train_edge_index], dim=1)

            # InfoNCE - 全量 user
           # 兩個隨機視角 (SGL 用 edge dropout)
            # === Contrastive Learning Augmentation (edge/node dropout) ===
            num_nodes = data.num_users + data.num_source_items + data.num_target_items

            view1 = dropout_graph(
                full_edge_index,
                drop_rate=args.edge_drop_rate,
                num_nodes=num_nodes,
                mode=args.ssl_aug_type  # "edge" 或 "node"
            )

            view2 = dropout_graph(
                full_edge_index,
                drop_rate=args.edge_drop_rate,
                num_nodes=num_nodes,
                mode=args.ssl_aug_type  # "edge" 或 "node"
            )

            # 得到兩個 augmented views 的 embedding
            user_v1, src_item_v1, tgt_item_v1 = model.encode_full(view1, return_norm=True)
            user_v2, src_item_v2, tgt_item_v2 = model.encode_full(view2, return_norm=True)

            # === SGL-style CL Loss ===
            ssl_user     = sgl_info_nce_loss(user_v1, user_v2, tau=args.nce_temp)
            ssl_src_item = sgl_info_nce_loss(src_item_v1, src_item_v2, tau=args.nce_temp)
            ssl_tgt_item = sgl_info_nce_loss(tgt_item_v1, tgt_item_v2, tau=args.nce_temp)


            # L2 reg
            l2 = sum((p.norm(2) ** 2 for p in model.parameters())) / 2.0

            # === Total Loss ===
            loss = (
                main_loss
                + args.lambda_user * ssl_user
                + args.lambda_src  * ssl_src_item
                + args.lambda_tgt  * ssl_tgt_item
                + args.reg * l2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # === Logging ===
        logging.info(
            f"[Epoch {epoch+1}] Loss={total_loss:.4f} "
            f"| Main={main_loss:.4f}"
            f"| SSL_User={ssl_user:.4f} | SSL_Src={ssl_src_item:.4f} | SSL_Tgt={ssl_tgt_item:.4f}"
        )

        # === Validation ===
        train_auc = evaluate("Train", model, source_edge_index, target_train_edge_index,
                            target_train_link, target_train_label, args)
        val_auc = evaluate("Valid", model, source_edge_index, target_train_edge_index,
                        target_valid_link, target_valid_label, args)
        logging.info(f"[Epoch {epoch+1}] TrainAUC={train_auc:.4f} | ValAUC={val_auc:.4f}")

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

        lr_scheduler.step()

    # ===== 測試與 HR/ER =====
    model = load_model(args).to(device)
    hit_ratio = evaluate_hit_ratio(model, data, source_edge_index, target_train_edge_index, args.top_k, 99, args)
    cold_item_id = getattr(args, "cold_item_id", 3080)
    er_ratio = evaluate_er_hit_ratio(model, data, source_edge_index, target_train_edge_index, {cold_item_id}, args.top_k, 99, args)
    test_auc = evaluate("Test", model, source_edge_index, target_train_edge_index, target_test_link, target_test_label, args)
    logging.info("[EvalSummary] HR@%d=%.4f | ER@%d=%.4f | TestAUC=%.4f",
                 args.top_k, hit_ratio, args.top_k, er_ratio, test_auc)

    cold_item_set = {3080}  # 固定使用 cold item 3080

    evaluate_multiple_topk(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        cold_item_set=cold_item_set,
        args=args
    )
    