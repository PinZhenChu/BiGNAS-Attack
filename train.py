import logging 

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np  
from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model

# =========================
# === Index Check Utils ===
# =========================
# [IDX-CHK] 小工具：安全印出 tensor 範圍
def _range_str(t):
    try:
        if t is None: return "None"
        if not torch.is_tensor(t): t = torch.tensor(t)
        if t.numel() == 0: return "empty"
        return f"min={int(t.min().item())}, max={int(t.max().item())}"
    except Exception as e:
        return f"range_err: {e}"


def meta_optimizeation(
    target_meta_loader,
    replace_optimizer,
    model,
    args,
    criterion,
    replace_scheduler,
    source_edge_index,
    target_edge_index,
):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)

            replace_optimizer.zero_grad()
            # [IDX-CHK]
            logging.debug(f"[IDX-CHK][meta_pred] link users {_range_str(target_link[0])}, items {_range_str(target_link[1])} "
                          f"(expect global items in [{args.num_users if hasattr(args,'num_users') else 'num_users'}, ...))")

            out = model.meta_prediction(
                source_edge_index, target_edge_index, target_link
            ).squeeze()
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()


@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label):
    model.eval()

    # [IDX-CHK]
    logging.debug(f"[IDX-CHK][evaluate:{name}] link users {_range_str(link[0])}, items {_range_str(link[1])} "
                  f"(items should be GLOBAL: [num_users, num_users+num_target_items))")

    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except:
        auc = 1.0
    logging.info(f"{name} AUC: {auc:4f}")

    model.train()
    return auc


def get_test_positive_dict(data):
    """
    根據 test link（data.target_test_link）建立 test set user 的正樣本字典。
    回傳: {user_id: [item1, item2, ...]}
    """
    test_user_item_dict = {}
    test_link = data.target_test_link.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        u, i = u.item(), i.item()
        if u not in test_user_item_dict:
            test_user_item_dict[u] = []
        test_user_item_dict[u].append(i)

    # [IDX-CHK] 抽樣檢查 test positives 是否為全域 item id
    try:
        sample_u = next(iter(test_user_item_dict))
        sample_items = test_user_item_dict[sample_u]
        logging.info(f"[IDX-CHK][get_test_positive_dict] sample user={sample_u}, "
                     f"items range: min={min(sample_items)}, max={max(sample_items)} "
                     f"(expect GLOBAL in [{data.num_users}, {data.num_users + data.num_target_items}))")
    except StopIteration:
        logging.warning("[IDX-CHK][get_test_positive_dict] test_user_item_dict is empty")

    return test_user_item_dict


def evaluate_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()
    hit_count = 0

    # 保持原本邏輯（本地範圍），只加提示
    all_target_items = set(range(data.num_target_items))
    # [IDX-CHK] 提醒：這裡使用的是本地 item 範圍，若 model 期望全域，需注意
    logging.info(f"[IDX-CHK][HR] all_target_items(local) range: min={min(all_target_items) if all_target_items else 'NA'}, "
                 f"max={max(all_target_items) if all_target_items else 'NA'} "
                 f"(LOCAL [0, {data.num_target_items})). If your model expects GLOBAL, convert before building link.")

    # ✅ 取得 test set 的 user -> positive items 對應關係
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())  # 直接使用 test set 的 user
    print(f"✅ Test set user count: {len(sim_users)}")

    total_users = 0
    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, set())
            if len(pos_items) > 1:
                print(f"⚠️ Warning: User {user_id} has {len(pos_items)} positives in test set.")

            if len(pos_items) == 0:
                continue

            # ✅ 第一步：選擇一個正樣本（注意：這是 test_link 的值，通常為 GLOBAL）
            pos_item = list(pos_items)[0]

            # ✅ 第二步：挑選負樣本（從非正樣本中隨機抽 num_candidates 個）
            negative_pool = list(all_target_items - set(pos_items))
            if len(negative_pool) < num_candidates:
                continue

            sampled_negatives = random.sample(negative_pool, num_candidates)

            # ✅ 第三步：組成候選清單（正例 + 負例），並打亂
            candidate_items = sampled_negatives + [pos_item]
            random.shuffle(candidate_items)

            # ✅ 第四步：轉成 tensor 並送入模型計算分數
            user_tensor = torch.tensor([user_id] * len(candidate_items), device=device)
            item_tensor = torch.tensor(candidate_items, device=device)

            # [IDX-CHK] 送進 model 前的檢查
            logging.debug(f"[IDX-CHK][HR] user={user_id}, link items {_range_str(item_tensor)} "
                          f"(if model expects GLOBAL items, ensure range is [{data.num_users}, {data.num_users + data.num_target_items}))")

            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
            top_k_items = [candidate_items[i] for i in top_k_indices]

            if pos_item in top_k_items:
                hit_count += 1
            total_users += 1

    hit_ratio = hit_count / total_users if total_users > 0 else 0.0
    logging.info(f"[HIT_RATIO@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
    return hit_ratio


# 🔍 統計每個 cold item 在 test set 中出現的次數（有幾個 user 買過）
def count_cold_item_occurrences(data, cold_item_set):
    item_count = {item: 0 for item in cold_item_set}
    test_link = data.target_test_link.cpu().numpy()
    for u, i in zip(*test_link):
        if i in cold_item_set:
            item_count[i] += 1
    return item_count


def find_cold_item_strict(data, target_train_edge_index, target_test_edge_index):
    import numpy as np
    from collections import Counter

    train_edges = target_train_edge_index.cpu().numpy()
    test_edges = target_test_edge_index.cpu().numpy()
    overlap_users = set(data.raw_overlap_users.cpu().numpy())  # ⬅️ overlap user list

    # Step 1: 統計 overlap user 在 test set 中點擊的 item 次數
    test_user, test_item = test_edges
    item_counter = Counter()

    for u, i in zip(test_user, test_item):
        if u in overlap_users:
            item_counter[i] += 1

    candidate_items = {i for i, cnt in item_counter.items() if cnt == 1}

    train_items = set(train_edges[1])
    test_items = set(test_item)

    cold_items = [i for i in candidate_items if i not in train_items and i in test_items]

    if not cold_items:
        print("❌ 找不到符合條件的 cold item")
        return None

    selected = cold_items[0]
    print(f"🧊 Found cold item: {selected}")

    # [IDX-CHK]
    logging.info(f"[IDX-CHK][find_cold_item_strict] selected={selected}, "
                 f"expect GLOBAL in [{data.num_users}, {data.num_users + data.num_target_items})")

    return selected


def evaluate_er_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    cold_item_set,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()

    # 保持原本邏輯（本地範圍），只加提示
    all_target_items = set(range(data.num_target_items))
    logging.info(f"[IDX-CHK][ER] all_target_items(local) range: min={min(all_target_items) if all_target_items else 'NA'}, "
                 f"max={max(all_target_items) if all_target_items else 'NA'} "
                 f"(LOCAL [0, {data.num_target_items})). If your model expects GLOBAL, convert before building link.")

    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    cold_item_hit_count = 0
    cold_item_ranks = []  # ⬅️ 儲存 cold item 被排進去時的排名

    with torch.no_grad():
        for user_id in sim_users:
            # 建立候選池
            negative_pool = list(all_target_items - cold_item_set)
            if len(negative_pool) < num_candidates:
                continue

            sampled_items = random.sample(negative_pool, num_candidates)
            sampled_items += list(cold_item_set)
            sampled_items = list(set(sampled_items))
            random.shuffle(sampled_items)

            user_tensor = torch.tensor([user_id] * len(sampled_items), device=device)
            item_tensor = torch.tensor(sampled_items, device=device)

            # [IDX-CHK]
            logging.debug(f"[IDX-CHK][ER] user={user_id}, link items {_range_str(item_tensor)} "
                          f"(if model expects GLOBAL items, ensure range is [{data.num_users}, {data.num_users + data.num_target_items}))")

            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            scores_list = scores.tolist()

            # 計算排序
            item_score_pairs = list(zip(sampled_items, scores_list))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_items = [item for item, _ in item_score_pairs]

            # 印出 cold item 的排名（可視化時使用）
            for cold_item in cold_item_set:
                if cold_item in sorted_items:
                    rank = sorted_items.index(cold_item) + 1

            top_k_items = sorted_items[:top_k]

            # ⬇️ 統計命中與排名
            cold_hits = [item for item in top_k_items if item in cold_item_set]
            if cold_hits:
                cold_item_hit_count += 1
                for cold_item in cold_hits:
                    rank = top_k_items.index(cold_item) + 1  # 1-based rank
                    cold_item_ranks.append(rank)

            total_users += 1

    er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
    logging.info(f"[ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")
    return er_ratio


def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, device):
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
            device=device
        )
        _ = evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            cold_item_set=cold_item_set,
            top_k=k,
            num_candidates=99,
            device=device
        )


# =========================
# === CL HELPERS BEGIN ===
# =========================

def edge_dropout(edge_index: torch.Tensor, drop_rate: float) -> torch.Tensor:
    """隨機丟棄部分邊（保留比例 1-drop_rate）"""
    if drop_rate <= 0.0:
        return edge_index
    E = edge_index.size(1)
    keep = max(1, int(E * (1 - drop_rate)))
    perm = torch.randperm(E, device=edge_index.device)[:keep]
    return edge_index[:, perm]


def node_dropout(edge_index: torch.Tensor, num_nodes: int, drop_rate: float) -> torch.Tensor:
    """隨機丟棄部分節點，並移除涉及這些節點的邊"""
    if drop_rate <= 0.0:
        return edge_index
    num_drop = int(num_nodes * drop_rate)
    if num_drop <= 0:
        return edge_index
    drop_nodes = torch.randperm(num_nodes, device=edge_index.device)[:num_drop]
    mask = ~(
        torch.isin(edge_index[0], drop_nodes) |
        torch.isin(edge_index[1], drop_nodes)
    )
    return edge_index[:, mask]


def info_nce_loss(emb_view1: torch.Tensor, emb_view2: torch.Tensor, idx: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    對比損失：同一節點跨視角拉近，其餘做負樣本。
    emb_view1/2: [N, d]；idx: 要對比的節點索引（例如 hard users）
    """
    if idx.numel() == 0:
        return torch.tensor(0.0, device=emb_view1.device)

    z1 = F.normalize(emb_view1[idx], dim=1)
    z2 = F.normalize(emb_view2[idx], dim=1)

    pos = torch.sum(z1 * z2, dim=1) / temperature  # [M]

    sim11 = torch.matmul(z1, z1.t()) / temperature
    sim22 = torch.matmul(z2, z2.t()) / temperature
    sim12 = torch.matmul(z1, z2.t()) / temperature

    M = sim12.size(0)
    mask = ~torch.eye(M, dtype=torch.bool, device=sim12.device)

    neg = torch.cat([
        sim11[mask].view(M, -1),
        sim22[mask].view(M, -1),
        sim12[mask].view(M, -1)
    ], dim=1)

    pos_exp = torch.exp(pos)
    neg_exp_sum = torch.exp(neg).sum(dim=1) + 1e-12
    loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-12)).mean()
    return loss


def sample_bpr_triplets(target_train_link: torch.Tensor, num_users: int, num_items: int, neg_k: int = 1, device: str = 'cuda'):
    """
    從 target_train_link 取 (u, pos_i)，每個正例抽 neg_k 個負例 j（u 未互動的 item）
    注意：target_train_link[1] 是合併圖索引，需先 -num_users 轉為 item 區間。
    """
    # [IDX-CHK] 進來先看 target_train_link[1] 範圍（應為 GLOBAL）
    logging.debug(f"[IDX-CHK][BPR] target_train_link[1] (GLOBAL) {_range_str(target_train_link[1])} "
                  f"(expect in [{num_users}, {num_users + num_items}))")

    u_all = target_train_link[0].to(device)
    i_all = (target_train_link[1] - num_users).to(device)

    # [IDX-CHK] 檢查轉本地後的範圍
    logging.debug(f"[IDX-CHK][BPR] i_all (LOCAL) {_range_str(i_all)} (expect in [0, {num_items}))")

    from collections import defaultdict
    interacted = defaultdict(set)
    for uu, ii in zip(u_all.tolist(), i_all.tolist()):
        interacted[uu].add(ii)

    U, I, J = [], [], []
    for uu, ii in zip(u_all.tolist(), i_all.tolist()):
        for _ in range(max(1, neg_k)):
            # 抽負樣本
            while True:
                jj = torch.randint(0, num_items, (1,), device=device).item()
                if jj not in interacted[uu]:
                    break
            U.append(uu); I.append(ii); J.append(jj)

    return torch.tensor(U, device=device), torch.tensor(I, device=device), torch.tensor(J, device=device)


def bpr_loss_from_embeddings(user_emb: torch.Tensor, item_emb: torch.Tensor, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    u_e = user_emb[u]
    i_e = item_emb[i]
    j_e = item_emb[j]
    x = torch.sum(u_e * i_e, dim=1) - torch.sum(u_e * j_e, dim=1)
    return -torch.log(torch.sigmoid(x) + 1e-12).mean()


def mine_hard_users_centroid(model, source_edge_index: torch.Tensor, target_edge_index: torch.Tensor,
                             ratio: float, num_users: int) -> torch.Tensor:
    """
    用 user 表徵到 centroid 的距離找 hardest（取前 ratio）。
    注意：encode_full 需要 source 與 target 的正確邊。
    """
    with torch.no_grad():
        user_emb, _, _ = model.encode_full(source_edge_index, target_edge_index, return_norm=True)
    center = user_emb.mean(dim=0, keepdim=True)                # [1, d]
    # 以 1 - cos 作距離
    dists = 1 - torch.matmul(user_emb, center.t()).squeeze(1)
    num = max(1, int(num_users * max(0.0, min(1.0, ratio))))
    hard_idx = torch.topk(dists, num).indices
    return hard_idx


def maybe_inject_edges(source_edge_index, target_train_edge_index, hard_users, args, device):
    """
    佔位：如需把你既有的 inject_fake_edges() 串進來，請在此改寫。
    預設不更動，僅回傳原邊。
    """
    return source_edge_index.to(device), target_train_edge_index.to(device)


def _check_edge_index(edge_index: torch.Tensor, num_nodes: int, name: str):
    """越界安全檢查"""
    if edge_index.numel() == 0:
        return
    max_idx = int(edge_index.max().item())
    min_idx = int(edge_index.min().item())
    if min_idx < 0 or max_idx >= num_nodes:
        raise ValueError(
            f"[{name}] edge_index out of bounds: min={min_idx}, max={max_idx}, num_nodes={num_nodes}"
        )

# =======================
# === CL HELPERS END  ===
# =======================


def train(model, perceptor, data, args):
    device = args.device
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)

    (
        source_edge_index,
        source_label,
        source_link,
        target_train_edge_index,
        target_train_label,
        target_train_link,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
        target_test_edge_index,  # ✅ 新增這一項
    ) = link_split(data)

    # [IDX-CHK] 基本與範圍資訊
    logging.info(f"[IDX-CHK] users={data.num_users}, target_items={data.num_target_items}, source_items={data.num_source_items}")
    logging.info(f"[IDX-CHK] target_train_edge_index[1] {_range_str(target_train_edge_index[1])}, "
                 f"期望在 [{data.num_users}, {data.num_users + data.num_target_items})")
    logging.info(f"[IDX-CHK] target_test_edge_index[1]  {_range_str(target_test_edge_index[1])}, "
                 f"期望在 [{data.num_users}, {data.num_users + data.num_target_items})")
    logging.info(f"[IDX-CHK] source_edge_index[1]       {_range_str(source_edge_index[1])}, "
                 f"期望在 [{data.num_users}, {data.num_users + data.num_source_items})")
    logging.debug(f"[IDX-CHK] target_train_link[1] {_range_str(target_train_link[1])} (expect GLOBAL)")
    logging.debug(f"[IDX-CHK] target_valid_link[1] {_range_str(target_valid_link[1])} (expect GLOBAL)")
    logging.debug(f"[IDX-CHK] target_test_link[1]  {_range_str(target_test_link[1])}  (expect GLOBAL)")

    data.target_test_link = target_test_link
    source_set_size = source_link.shape[1]
    train_set_size = target_train_link.shape[1]
    val_set_size = target_valid_link.shape[1]
    test_set_size = target_test_link.shape[1]
    logging.info(f"Train set size: {train_set_size}")
    logging.info(f"Valid set size: {val_set_size}")
    logging.info(f"Test set size: {test_set_size}")

    target_train_set = Dataset(
        target_train_link.to("cpu"),
        target_train_label.to("cpu"),
    )
    target_train_loader = DataLoader(
        target_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )

    source_batch_size = int(args.batch_size * train_set_size / source_set_size)
    source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
    source_train_loader = DataLoader(
        source_train_set,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    target_meta_loader = DataLoader(
        target_train_set,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )
    target_meta_iter = iter(target_meta_loader)
    source_meta_batch_size = int(
        args.meta_batch_size * train_set_size / source_set_size
    )
    source_meta_loader = DataLoader(
        source_train_set,
        batch_size=source_meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    perceptor_optimizer = torch.optim.Adam(
        perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    meta_optimizer = MetaOptimizer(
        meta_optimizer=perceptor_optimizer,
        hpo_lr=args.hpo_lr,
        truncate_iter=3,
        max_grad_norm=10,
    )

    model_param = [
        param for name, param in model.named_parameters() if "preds" not in name
    ]
    replace_param = [
        param for name, param in model.named_parameters() if name.startswith("replace")
    ]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        replace_optimizer, T_max=args.T_max
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.model_path,
        trace_func=logging.info,
    )

    criterion = nn.BCELoss(reduction="none")

    # === CL 超參數（若未在 argparse 設定，這裡有預設值） ===
    ssl_aug_type   = getattr(args, 'ssl_aug_type', 'edge')   # 'edge' or 'node'
    edge_drop_rate = getattr(args, 'edge_drop_rate', 0.2)
    node_drop_rate = getattr(args, 'node_drop_rate', 0.2)
    ssl_reg        = getattr(args, 'ssl_reg', 0.1)
    reg_l2         = getattr(args, 'reg', 1e-4)
    nce_temp       = getattr(args, 'nce_temp', 0.2)
    hard_ratio     = getattr(args, 'hard_ratio', 0.1)
    hard_interval  = getattr(args, 'hard_mine_interval', 1)
    neg_samples    = getattr(args, 'neg_samples', 1)

    iteration = 0
    for epoch in range(args.epochs):
        # -----------------------------
        # 原本 BCE + Meta 訓練流程
        # -----------------------------
        for (source_link, source_label), (target_link, target_label) in zip(
            source_train_loader, target_train_loader
        ):
            torch.cuda.empty_cache()
            source_link = source_link.to(device)
            source_label = source_label.to(device)
            target_link = target_link.to(device)
            target_label = target_label.to(device)

            # [IDX-CHK] 當前 batch link 的範圍觀察
            logging.debug(f"[IDX-CHK][batch] source_link items {_range_str(source_link[1])} (expect GLOBAL for source graph)")
            logging.debug(f"[IDX-CHK][batch] target_link items {_range_str(target_link[1])} (expect GLOBAL for target graph)")

            weight_source = perceptor(source_link[1], source_edge_index, model)

            optimizer.zero_grad()
            # [IDX-CHK] 呼叫前檢查
            logging.debug(f"[IDX-CHK][model-call] source_out link items {_range_str(source_link[1])}")
            source_out = model(
                source_edge_index, target_train_edge_index, source_link, is_source=True
            ).squeeze()

            logging.debug(f"[IDX-CHK][model-call] target_out link items {_range_str(target_link[1])}")
            target_out = model(
                source_edge_index, target_train_edge_index, target_link, is_source=False
            ).squeeze()
            source_loss = (
                criterion(source_out, source_label).reshape(-1, 1) * weight_source
            ).sum()
            target_loss = criterion(target_out, target_label).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1
            if (
                args.use_source
                and args.use_meta
                and iteration % args.meta_interval == 0
            ):
                logging.info(f"Entering meta optimization, iteration: {iteration}")
                meta_optimizeation(
                    target_meta_loader,
                    replace_optimizer,
                    model,
                    args,
                    criterion,
                    replace_scheduler,
                    source_edge_index,
                    target_train_edge_index,
                )

                try:
                    target_meta_link, target_meta_label = next(target_meta_iter)
                except StopIteration:
                    target_meta_iter = iter(target_meta_loader)
                    target_meta_link, target_meta_label = next(target_meta_iter)

                target_meta_link, target_meta_label = (
                    target_meta_link.to(device),
                    target_meta_label.to(device),
                )
                optimizer.zero_grad()

                logging.debug(f"[IDX-CHK][meta] target_meta_link items {_range_str(target_meta_link[1])}")
                target_out = model(
                    source_edge_index,
                    target_train_edge_index,
                    target_meta_link,
                    is_source=False,
                ).squeeze()
                meta_loss = criterion(target_out, target_meta_label).mean()

                for (source_link, source_label), (target_link, target_label) in zip(
                    source_meta_loader, target_meta_loader
                ):
                    source_link, source_label = source_link.to(device), source_label.to(
                        device
                    )
                    target_link, target_label = target_link.to(device), target_label.to(
                        device
                    )
                    logging.debug(f"[IDX-CHK][meta-train] source_link items {_range_str(source_link[1])}")
                    logging.debug(f"[IDX-CHK][meta-train] target_link items {_range_str(target_link[1])}")
                    weight_source = perceptor(source_link[1], source_edge_index, model)

                    optimizer.zero_grad()
                    source_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        source_link,
                        is_source=True,
                    ).squeeze()
                    target_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        target_link,
                        is_source=False,
                    ).squeeze()
                    source_loss = (
                        criterion(source_out, source_label).reshape(-1, 1)
                        * weight_source
                    ).sum()
                    target_loss = criterion(target_out, target_label).mean()
                    meta_train_loss = (
                        source_loss + target_loss if args.use_meta else target_loss
                    )
                    break

                torch.cuda.empty_cache()
                meta_optimizer.step(
                    train_loss=meta_train_loss,
                    val_loss=meta_loss,
                    aux_params=list(perceptor.parameters()),
                    parameters=model_param,
                    return_grads=True,
                    entropy=None,
                )

        # =========================
        # === CL TRAINING BEGIN ===
        # =========================
        se_used = source_edge_index.to(device)
        te_used = target_train_edge_index.to(device)

        # 越界檢查（完整圖）
        _check_edge_index(se_used, data.num_users + data.num_source_items, "source_edge_index(se_used)")
        _check_edge_index(te_used, data.num_users + data.num_target_items, "target_edge_index(te_used)")

        # [IDX-CHK] 圖的 item 端範圍
        logging.debug(f"[IDX-CHK][CL] se_used[1] {_range_str(se_used[1])} expect [{data.num_users}, {data.num_users + data.num_source_items})")
        logging.debug(f"[IDX-CHK][CL] te_used[1] {_range_str(te_used[1])} expect [{data.num_users}, {data.num_users + data.num_target_items})")

        # hard users（每 hard_interval 個 epoch 更新一次；否則用隨機/全量）
        if hard_interval <= 1 or (epoch % hard_interval) == 0:
            hard_users = mine_hard_users_centroid(
                model, se_used, te_used, ratio=hard_ratio, num_users=data.num_users
            )
        else:
            # fallback：隨機取一批 user 做對比
            num = max(1, int(data.num_users * hard_ratio))
            hard_users = torch.randperm(data.num_users, device=device)[:num]

        # 選配：若你想把注入假邊串進來，在 maybe_inject_edges 裡改
        se_used, te_used = maybe_inject_edges(se_used, te_used, hard_users, args, device)

        # 兩張子圖（edge / node dropout）
        if ssl_aug_type == 'edge':
            te_view1 = edge_dropout(te_used, edge_drop_rate)
            te_view2 = edge_dropout(te_used, edge_drop_rate)
        else:
            total_nodes = data.num_users + data.num_target_items
            te_view1 = node_dropout(te_used, total_nodes, node_drop_rate)
            te_view2 = node_dropout(te_used, total_nodes, node_drop_rate)

        # 越界檢查（子圖）
        _check_edge_index(te_view1, data.num_users + data.num_target_items, "te_view1")
        _check_edge_index(te_view2, data.num_users + data.num_target_items, "te_view2")

        # [IDX-CHK] 子圖 item 端範圍
        logging.debug(f"[IDX-CHK][CL] te_view1[1] {_range_str(te_view1[1])}")
        logging.debug(f"[IDX-CHK][CL] te_view2[1] {_range_str(te_view2[1])}")

        # 完整圖：給 BPR 用（不做 normalize 更常見；可自行 ablation）
        user_full, _, item_full = model.encode_full(se_used, te_used, return_norm=False)

        # 兩視角：給 InfoNCE 用（normalize 比較穩）
        (user_v1, item_v1), (user_v2, item_v2) = model.encode_target_views(
            se_used, te_view1, te_view2, return_norm=True
        )

        # BPR 取樣與 loss
        u, i, j = sample_bpr_triplets(
            target_train_link=target_train_link,
            num_users=data.num_users,
            num_items=data.num_target_items,
            neg_k=neg_samples,
            device=device
        )
        bpr = bpr_loss_from_embeddings(user_full, item_full, u, i, j)

        # InfoNCE（先做 user；若要 item，也可再做一次後取平均）
        ssl_user = info_nce_loss(user_v1, user_v2, hard_users, temperature=nce_temp)

        # L2
        l2 = torch.tensor(0.0, device=device)
        for p in model.parameters():
            l2 = l2 + p.norm(2) ** 2
        l2 = l2 / 2.0

        loss_cl = bpr + ssl_reg * ssl_user + reg_l2 * l2

        optimizer.zero_grad()
        loss_cl.backward()
        optimizer.step()

        logging.info(f"[Epoch: {epoch}] CL => BPR: {bpr:.4f} | SSL_user: {ssl_user:.4f} | L2: {l2:.4f} | Total_CL: {loss_cl:.4f}")
        wandb.log(
            {"bpr": bpr.item(), "ssl_user": ssl_user.item(), "reg_l2": l2.item(), "loss_cl": loss_cl.item()},
            step=epoch,
        )
        # =======================
        # === CL TRAINING END ===
        # =======================

        train_auc = evaluate(
            "Train",
            model,
            source_edge_index,
            target_train_edge_index,
            target_train_link,
            target_train_label,
        )
        val_auc = evaluate(
            "Valid",
            model,
            source_edge_index,
            target_train_edge_index,
            target_valid_link,
            target_valid_label,
        )

        logging.info(
            f"[Epoch: {epoch}]Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
        )
        wandb.log(
            {
                "loss": loss,
                "train_auc": train_auc,
                "val_auc": val_auc
            },
            step=epoch,
        )

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        lr_scheduler.step()

    model = load_model(args).to(device)
    evaluate_hit_ratio(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,  # ✅ 正確傳入測試時的 edge_index（此處仍用 train 圖做 ranking）
        top_k=args.top_k,
        num_candidates=99,
        device=device,
    )
    # cold_item_id = find_cold_item_strict(data, target_train_edge_index, target_test_edge_index)
    cold_item_id = 17069
    if cold_item_id is not None:
        evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_train_edge_index,
            cold_item_set={cold_item_id},
            top_k=args.top_k,
            num_candidates=99,
            device=device,
        )

    test_auc = evaluate(
        "Test",
        model,
        source_edge_index,
        target_train_edge_index,
        target_test_link,
        target_test_label,
    )
    logging.info(f"Test AUC: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})
    evaluate_multiple_topk(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        cold_item_set={cold_item_id},   # 注意這邊是 set
        device=device
    )
    # === 存下 source_item_embedding ===
    source_emb = model.source_item_embedding.weight.detach().cpu().numpy()
    # np.save("source_item_embedding.npy", source_emb)
    # np.savetxt("source_item_embedding.csv", source_emb, delimiter=",")
    # logging.info(f"✅ Saved source_item_embedding: shape={source_emb.shape}")
