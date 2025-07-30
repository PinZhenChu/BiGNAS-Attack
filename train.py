import torch
import logging
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn

from dataset import Dataset
from utils import link_split, load_model
from pytorchtools import EarlyStopping
from auxilearn.optim import MetaOptimizer


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

    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except Exception:
        auc = 1.0
    logging.info(f"{name} AUC: {auc:.4f}")

    model.train()
    return auc

def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, device):
    topk_list = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    print("\n📊 Evaluation for multiple top-K values:")
    for k in topk_list:
        hr = evaluate_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            top_k=k,
            num_candidates=99,
            device=device
        )

        er = evaluate_er_hit_ratio(
            model=model,
            data=data,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            cold_item_set=cold_item_set,
            top_k=k,
            num_candidates=99,
            device=device
        )

        # print(f"[Top-{k:>3}] HR@{k:<3}: {{hr:.4f}}  |  ER@{k:<3}: {{er:.4f}}")



def get_test_positive_dict(data):
    test_user_item_dict = {}
    test_link = data.target_test_edge_index.cpu()
    for u, i in zip(test_link[0], test_link[1]):
        u, i = u.item(), i.item()
        if u not in test_user_item_dict:
            test_user_item_dict[u] = []
        test_user_item_dict[u].append(i)
    return test_user_item_dict



def evaluate_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()
    hit_count = 0
    all_target_items = set(range(data.num_target_items))

    # 取得 test set 的 user -> positive items 對應關係
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())  # 使用 test set users
    logging.info(f"Test set user count: {len(sim_users)}")

    total_users = 0
    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    with torch.no_grad():
        for user_id in sim_users:
            pos_items = user_interactions.get(user_id, set())
            if len(pos_items) > 1:
                logging.warning(f"User {user_id} has {len(pos_items)} positives in test set.")

            if len(pos_items) == 0:
                continue

            # 選一個正樣本
            pos_item = list(pos_items)[0]

            # 從非正樣本中抽取負樣本
            negative_pool = list(all_target_items - set(pos_items))
            if len(negative_pool) < num_candidates:
                continue

            sampled_negatives = random.sample(negative_pool, num_candidates)

            # 組成候選清單，正例 + 負例，並打亂
            candidate_items = sampled_negatives + [pos_item]
            random.shuffle(candidate_items)

            user_tensor = torch.tensor([user_id] * len(candidate_items), device=device)
            item_tensor = torch.tensor(candidate_items, device=device)
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


def evaluate_er_hit_ratio(
    model, data, source_edge_index, target_edge_index,
    cold_item_set,
    top_k, num_candidates=99,
    device=None
):
    import random
    model.eval()

    all_target_items = set(range(data.num_target_items))
    user_interactions = get_test_positive_dict(data)
    sim_users = list(user_interactions.keys())

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)

    total_users = 0
    cold_item_hit_count = 0
    cold_item_ranks = []  # 儲存 cold item 被排進去時的排名

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
            link = torch.stack([user_tensor, item_tensor], dim=0)

            scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
            scores_list = scores.tolist()

            # 計算排序
            item_score_pairs = list(zip(sampled_items, scores_list))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_items = [item for item, _ in item_score_pairs]

            top_k_items = sorted_items[:top_k]

            # 統計命中與排名
            cold_hits = [item for item in top_k_items if item in cold_item_set]
            if cold_hits:
                cold_item_hit_count += 1
                for cold_item in cold_hits:
                    rank = top_k_items.index(cold_item) + 1  # 1-based rank
                    cold_item_ranks.append(rank)

            total_users += 1

    er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
    avg_rank = sum(cold_item_ranks) / len(cold_item_ranks) if cold_item_ranks else -1
    median_rank = (
        sorted(cold_item_ranks)[len(cold_item_ranks) // 2] if cold_item_ranks else -1
    )

    logging.info(f"[ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")
    return er_ratio


def train(model, perceptor, data, args, source_edge_index=None):
    device = args.device
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)

    (
        orig_source_edge_index,
        source_label,
        source_link,
        target_train_edge_index,
        target_train_label,
        target_train_link,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
        target_test_edge_index,
    ) = link_split(data)

    if source_edge_index is not None:
        source_edge_index = source_edge_index.to(device)
    else:
        source_edge_index = orig_source_edge_index.to(device)

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
    iteration = 0

    for epoch in range(args.epochs):
        for (source_link, source_label), (target_link, target_label) in zip(
            source_train_loader, target_train_loader
        ):
            torch.cuda.empty_cache()
            source_link = source_link.to(device)
            source_label = source_label.to(device)
            target_link = target_link.to(device)
            target_label = target_label.to(device)
            weight_source = perceptor(source_link[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(
                source_edge_index, target_train_edge_index, source_link, is_source=True
            ).squeeze()
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

                target_meta_link = target_meta_link.to(device)
                target_meta_label = target_meta_label.to(device)

                optimizer.zero_grad()
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
                    source_link = source_link.to(device)
                    source_label = source_label.to(device)
                    target_link = target_link.to(device)
                    target_label = target_label.to(device)
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
            f"[Epoch: {epoch}] Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
        )

        # 可選額外log到 wandb（需先 import wandb）
        # import wandb
        # wandb.log({"loss": loss, "train_auc": train_auc, "val_auc": val_auc}, step=epoch)

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

        lr_scheduler.step()

    # 讀取最佳模型、測試評估
    model = load_model(args).to(device)
    evaluate_hit_ratio(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        top_k=args.top_k,
        num_candidates=99,
        device=device,
    )

    cold_item_id = 2286  # 你可以改成要評估的冷門商品id
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
    evaluate_multiple_topk(
        model=model,
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_train_edge_index,
        cold_item_set={2286},   # 注意這邊是 set，不是 cold_item_id=
        device=device
    )


    return model
#######################################################
# import logging

# import torch
# import torch.nn as nn
# import wandb
# from sklearn.metrics import roc_auc_score
# from torch.utils.data import DataLoader

# from auxilearn.optim import MetaOptimizer
# from dataset import Dataset
# from pytorchtools import EarlyStopping
# from utils import link_split, load_model


# def meta_optimizeation(
#     target_meta_loader,
#     replace_optimizer,
#     model,
#     args,
#     criterion,
#     replace_scheduler,
#     source_edge_index,
#     target_edge_index,
# ):
#     device = args.device
#     for batch, (target_link, target_label) in enumerate(target_meta_loader):
#         if batch < args.descent_step:
#             target_link, target_label = target_link.to(device), target_label.to(device)

#             replace_optimizer.zero_grad()
#             out = model.meta_prediction(
#                 source_edge_index, target_edge_index, target_link
#             ).squeeze()
#             loss_target = criterion(out, target_label).mean()
#             loss_target.backward()
#             replace_optimizer.step()
#         else:
#             break
#     replace_scheduler.step()


# @torch.no_grad()
# def evaluate(name, model, source_edge_index, target_edge_index, link, label):
#     model.eval()

#     out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
#     try:
#         auc = roc_auc_score(label.tolist(), out.tolist())
#     except:
#         auc = 1.0
#     logging.info(f"{name} AUC: {auc:4f}")

#     model.train()
#     return auc
# def get_test_positive_dict(data):
#     """
#     根據 test link（data.target_test_link）建立 test set user 的正樣本字典。
#     回傳: {user_id: [item1, item2, ...]}
#     """
#     test_user_item_dict = {}
#     test_link = data.target_test_link.cpu()
#     for u, i in zip(test_link[0], test_link[1]):
#         u, i = u.item(), i.item()
#         if u not in test_user_item_dict:
#             test_user_item_dict[u] = []
#         test_user_item_dict[u].append(i)
#     return test_user_item_dict

# def evaluate_hit_ratio(
#     model, data, source_edge_index, target_edge_index,
#     top_k, num_candidates=99,
#     device=None
# ):
#     import random
#     model.eval()
#     hit_count = 0
#     all_target_items = set(range(data.num_target_items))

#     # ✅ 取得 test set 的 user -> positive items 對應關係
#     user_interactions = get_test_positive_dict(data)
#     sim_users = list(user_interactions.keys())  # 直接使用 test set 的 user
#     print(f"✅ Test set user count: {len(sim_users)}")

#     total_users = 0
#     source_edge_index = source_edge_index.to(device)
#     target_edge_index = target_edge_index.to(device)

#     with torch.no_grad():
#         for user_id in sim_users:
#             pos_items = user_interactions.get(user_id, set())
#             if len(pos_items) > 1:
#                 print(f"⚠️ Warning: User {user_id} has {len(pos_items)} positives in test set.")

#             if len(pos_items) == 0:
#                 continue

#             # ✅ 第一步：選擇一個正樣本
#             pos_item = list(pos_items)[0]
#             # print(f"\n=== [User {user_id}] ===")
#             # print(f"👉 Positive item: {pos_item}")

#             # ✅ 第二步：挑選負樣本（從非正樣本中隨機抽 num_candidates 個）
#             negative_pool = list(all_target_items - set(pos_items))
#             if len(negative_pool) < num_candidates:
#                 # print(f"❌ Negative pool too small for user {user_id}, skipping.")
#                 continue

#             sampled_negatives = random.sample(negative_pool, num_candidates)
#             # print(f"🎯 Sampled {num_candidates} negatives: {sampled_negatives[:10]}...")

#             # ✅ 第三步：組成候選清單（正例 + 負例），並打亂
#             candidate_items = sampled_negatives + [pos_item]
#             random.shuffle(candidate_items)
#             # print(f"🧮 Candidate items (shuffled): {candidate_items[:10]}...")

#             # ✅ 第四步：轉成 tensor 並送入模型計算分數
#             user_tensor = torch.tensor([user_id] * len(candidate_items), device=device)
#             item_tensor = torch.tensor(candidate_items, device=device)
#             link = torch.stack([user_tensor, item_tensor], dim=0)

#             scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
#             top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
#             top_k_items = [candidate_items[i] for i in top_k_indices]

#             # print(f"📈 Top-{top_k} prediction: {top_k_items}")
#             # print(f"✔️ Hit? {'Yes ✅' if pos_item in top_k_items else 'No ❌'}")

#             if pos_item in top_k_items:
#                 hit_count += 1
#             total_users += 1

#     hit_ratio = hit_count / total_users if total_users > 0 else 0.0
#     logging.info(f"[HIT_RATIO@{top_k}] Users={total_users}, Hits={hit_count}, Hit Ratio={hit_ratio:.4f}")
#     return hit_ratio

# # 🔍 統計每個 cold item 在 test set 中出現的次數（有幾個 user 買過）
# def count_cold_item_occurrences(data, cold_item_set):
#     item_count = {item: 0 for item in cold_item_set}
#     test_link = data.target_test_link.cpu().numpy()
#     for u, i in zip(*test_link):
#         if i in cold_item_set:
#             item_count[i] += 1
#     return item_count

# def find_cold_item_strict(data, target_train_edge_index, target_test_edge_index):
#     import numpy as np
#     from collections import defaultdict

#     train_edges = target_train_edge_index.cpu().numpy()
#     test_edges = target_test_edge_index.cpu().numpy()
#     overlap_users = set(data.raw_overlap_users.cpu().numpy())  # ⬅️ overlap user list

#     train_items = set(train_edges[1])
#     test_user, test_item = test_edges

#     # ✅ 建立 test set 中 item → user 的 mapping
#     item_user_map = defaultdict(set)
#     for u, i in zip(test_user, test_item):
#         if u in overlap_users:
#             item_user_map[i].add(u)

#     candidate_info = []  # 存放 (item_id, user_id, user_source_count)

#     for item, users in item_user_map.items():
#         if item not in train_items and len(users) == 1:
#             user = list(users)[0]

#             # ✅ 計算這位 user 在 source domain 買過幾個 item
#             source_edges = data.source_link.cpu()
#             source_items = source_edges[1, source_edges[0] == user]
#             num_bought = len(source_items)

#             candidate_info.append((item, user, num_bought))

#     if not candidate_info:
#         print("❌ 找不到符合條件的 cold item")
#         return None

#     # ✅ 依照 source domain 買的數量做排序（由大到小）
#     candidate_info.sort(key=lambda x: x[2], reverse=True)

#     # selected, seed_user, source_count = candidate_info[100]
#     selected = 2286
#     # 統計出現次數
#     train_count = (train_edges[1] == selected).sum()
#     test_count = (test_item == selected).sum()

#     print("🧊 Found cold item:", selected)
#     print("🔗 ASIN:", data.target_id2asin.get(selected, "N/A"))
#     print(f"📊 Appears in train set: {train_count} times")
#     print(f"📊 Appears in test set : {test_count} times")
#     # print(f"👤 Seed user ID: {seed_user}")
#     # print(f"🛍️  Bought {source_count} items in source domain")

#     return selected



# def evaluate_er_hit_ratio(
#     model, data, source_edge_index, target_edge_index,
#     cold_item_set,
#     top_k, num_candidates=99,
#     device=None
# ):
#     import random
#     model.eval()

#     all_target_items = set(range(data.num_target_items))
#     user_interactions = get_test_positive_dict(data)
#     sim_users = list(user_interactions.keys())

#     source_edge_index = source_edge_index.to(device)
#     target_edge_index = target_edge_index.to(device)

#     total_users = 0
#     cold_item_hit_count = 0
#     cold_item_ranks = []  # ⬅️ 儲存 cold item 被排進去時的排名

#     with torch.no_grad():
#         for user_id in sim_users:
#             # 建立候選池
#             negative_pool = list(all_target_items - cold_item_set)
#             if len(negative_pool) < num_candidates:
#                 continue

#             sampled_items = random.sample(negative_pool, num_candidates)
#             sampled_items += list(cold_item_set)
#             sampled_items = list(set(sampled_items))
#             random.shuffle(sampled_items)

#             user_tensor = torch.tensor([user_id] * len(sampled_items), device=device)
#             item_tensor = torch.tensor(sampled_items, device=device)
#             link = torch.stack([user_tensor, item_tensor], dim=0)

#             scores = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
#             scores_list = scores.tolist()

#             # 印出每個 item 的分數
#             # print(f"\n=== [User {user_id}] ===")
#             # for item, score in zip(sampled_items, scores_list):
#             #     tag = "🧊 COLD" if item in cold_item_set else ""
#             #     print(f"Item {item:4d} | Score: {score:.4f} {tag}")

#             # 計算排序
#             item_score_pairs = list(zip(sampled_items, scores_list))
#             item_score_pairs.sort(key=lambda x: x[1], reverse=True)
#             sorted_items = [item for item, _ in item_score_pairs]

#             # 印出 cold item 的排名
#             for cold_item in cold_item_set:
#                 if cold_item in sorted_items:
#                     rank = sorted_items.index(cold_item) + 1
#                     # print(f"🔍 Cold item {cold_item} ranked #{rank} / {len(sorted_items)}")

#             top_k_items = sorted_items[:top_k]


#             # ⬇️ 統計命中與排名
#             cold_hits = [item for item in top_k_items if item in cold_item_set]
#             if cold_hits:
#                 cold_item_hit_count += 1
#                 for cold_item in cold_hits:
#                     rank = top_k_items.index(cold_item) + 1  # 1-based rank
#                     cold_item_ranks.append(rank)

#             total_users += 1

#     er_ratio = cold_item_hit_count / total_users if total_users > 0 else 0.0
#     avg_rank = sum(cold_item_ranks) / len(cold_item_ranks) if cold_item_ranks else -1
#     median_rank = (
#         sorted(cold_item_ranks)[len(cold_item_ranks) // 2] if cold_item_ranks else -1
#     )

#     logging.info(f"[ER@{top_k}] Users={total_users}, Cold Item Hits={cold_item_hit_count}, ER Ratio={er_ratio:.4f}")
#     # logging.info(f"[ER@{top_k}] Cold item avg rank: {avg_rank:.2f}, median rank: {median_rank}")

#     return er_ratio

# def evaluate_multiple_topk(model, data, source_edge_index, target_edge_index, cold_item_set, device):
#     topk_list = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
#     print("\n📊 Evaluation for multiple top-K values:")
#     for k in topk_list:
#         hr = evaluate_hit_ratio(
#             model=model,
#             data=data,
#             source_edge_index=source_edge_index,
#             target_edge_index=target_edge_index,
#             top_k=k,
#             num_candidates=99,
#             device=device
#         )

#         er = evaluate_er_hit_ratio(
#             model=model,
#             data=data,
#             source_edge_index=source_edge_index,
#             target_edge_index=target_edge_index,
#             cold_item_set=cold_item_set,
#             top_k=k,
#             num_candidates=99,
#             device=device
#         )
# def train(model, perceptor, data, args):
#     device = args.device
#     data = data.to(device)
#     model = model.to(device)
#     perceptor = perceptor.to(device)

#     (
#         source_edge_index,
#         source_label,
#         source_link,
#         target_train_edge_index,
#         target_train_label,
#         target_train_link,
#         target_valid_link,
#         target_valid_label,
#         target_test_link,
#         target_test_label,
#         target_test_edge_index,  # ✅ 新增這一項
#     ) = link_split(data)
#     data.target_test_link = target_test_link
#     source_set_size = source_link.shape[1]
#     train_set_size = target_train_link.shape[1]
#     val_set_size = target_valid_link.shape[1]
#     test_set_size = target_test_link.shape[1]
#     logging.info(f"Train set size: {train_set_size}")
#     logging.info(f"Valid set size: {val_set_size}")
#     logging.info(f"Test set size: {test_set_size}")

#     target_train_set = Dataset(
#         target_train_link.to("cpu"),
#         target_train_label.to("cpu"),
#     )
#     target_train_loader = DataLoader(
#         target_train_set,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=target_train_set.collate_fn,
#     )

#     source_batch_size = int(args.batch_size * train_set_size / source_set_size)
#     source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
#     source_train_loader = DataLoader(
#         source_train_set,
#         batch_size=source_batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=source_train_set.collate_fn,
#     )

#     target_meta_loader = DataLoader(
#         target_train_set,
#         batch_size=args.meta_batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=target_train_set.collate_fn,
#     )
#     target_meta_iter = iter(target_meta_loader)
#     source_meta_batch_size = int(
#         args.meta_batch_size * train_set_size / source_set_size
#     )
#     source_meta_loader = DataLoader(
#         source_train_set,
#         batch_size=source_meta_batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=source_train_set.collate_fn,
#     )

#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=args.lr, weight_decay=args.weight_decay
#     )
#     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=args.epochs
#     )

#     perceptor_optimizer = torch.optim.Adam(
#         perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay
#     )
#     meta_optimizer = MetaOptimizer(
#         meta_optimizer=perceptor_optimizer,
#         hpo_lr=args.hpo_lr,
#         truncate_iter=3,
#         max_grad_norm=10,
#     )

#     model_param = [
#         param for name, param in model.named_parameters() if "preds" not in name
#     ]
#     replace_param = [
#         param for name, param in model.named_parameters() if name.startswith("replace")
#     ]
#     replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
#     replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         replace_optimizer, T_max=args.T_max
#     )

#     early_stopping = EarlyStopping(
#         patience=args.patience,
#         verbose=True,
#         path=args.model_path,
#         trace_func=logging.info,
#     )

#     criterion = nn.BCELoss(reduction="none")
#     iteration = 0
#     for epoch in range(args.epochs):
#         for (source_link, source_label), (target_link, target_label) in zip(
#             source_train_loader, target_train_loader
#         ):
#             torch.cuda.empty_cache()
#             source_link = source_link.to(device)
#             source_label = source_label.to(device)
#             target_link = target_link.to(device)
#             target_label = target_label.to(device)
#             weight_source = perceptor(source_link[1], source_edge_index, model)

#             optimizer.zero_grad()
#             source_out = model(
#                 source_edge_index, target_train_edge_index, source_link, is_source=True
#             ).squeeze()
#             target_out = model(
#                 source_edge_index, target_train_edge_index, target_link, is_source=False
#             ).squeeze()
#             source_loss = (
#                 criterion(source_out, source_label).reshape(-1, 1) * weight_source
#             ).sum()
#             target_loss = criterion(target_out, target_label).mean()
#             loss = source_loss + target_loss if args.use_meta else target_loss
#             loss.backward()
#             optimizer.step()

#             iteration += 1
#             if (
#                 args.use_source
#                 and args.use_meta
#                 and iteration % args.meta_interval == 0
#             ):
#                 logging.info(f"Entering meta optimization, iteration: {iteration}")
#                 meta_optimizeation(
#                     target_meta_loader,
#                     replace_optimizer,
#                     model,
#                     args,
#                     criterion,
#                     replace_scheduler,
#                     source_edge_index,
#                     target_train_edge_index,
#                 )

#                 try:
#                     target_meta_link, target_meta_label = next(target_meta_iter)
#                 except StopIteration:
#                     target_meta_iter = iter(target_meta_loader)
#                     target_meta_link, target_meta_label = next(target_meta_iter)

#                 target_meta_link, target_meta_label = (
#                     target_meta_link.to(device),
#                     target_meta_label.to(device),
#                 )
#                 optimizer.zero_grad()
#                 target_out = model(
#                     source_edge_index,
#                     target_train_edge_index,
#                     target_meta_link,
#                     is_source=False,
#                 ).squeeze()
#                 meta_loss = criterion(target_out, target_meta_label).mean()

#                 for (source_link, source_label), (target_link, target_label) in zip(
#                     source_meta_loader, target_meta_loader
#                 ):
#                     source_link, source_label = source_link.to(device), source_label.to(
#                         device
#                     )
#                     target_link, target_label = target_link.to(device), target_label.to(
#                         device
#                     )
#                     weight_source = perceptor(source_link[1], source_edge_index, model)

#                     optimizer.zero_grad()
#                     source_out = model(
#                         source_edge_index,
#                         target_train_edge_index,
#                         source_link,
#                         is_source=True,
#                     ).squeeze()
#                     target_out = model(
#                         source_edge_index,
#                         target_train_edge_index,
#                         target_link,
#                         is_source=False,
#                     ).squeeze()
#                     source_loss = (
#                         criterion(source_out, source_label).reshape(-1, 1)
#                         * weight_source
#                     ).sum()
#                     target_loss = criterion(target_out, target_label).mean()
#                     meta_train_loss = (
#                         source_loss + target_loss if args.use_meta else target_loss
#                     )
#                     break

#                 torch.cuda.empty_cache()
#                 meta_optimizer.step(
#                     train_loss=meta_train_loss,
#                     val_loss=meta_loss,
#                     aux_params=list(perceptor.parameters()),
#                     parameters=model_param,
#                     return_grads=True,
#                     entropy=None,
#                 )
#         train_auc = evaluate(
#             "Train",
#             model,
#             source_edge_index,
#             target_train_edge_index,
#             target_train_link,
#             target_train_label,
#         )
#         val_auc = evaluate(
#             "Valid",
#             model,
#             source_edge_index,
#             target_train_edge_index,
#             target_valid_link,
#             target_valid_label,
#         )

#         logging.info(
#             f"[Epoch: {epoch}]Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
#         )
#         wandb.log(
#             {
#                 "loss": loss,
#                 "train_auc": train_auc,
#                 "val_auc": val_auc
#             },
#             step=epoch,
#         )

#         early_stopping(val_auc, model)
#         if early_stopping.early_stop:
#             logging.info("Early stopping")
#             break

#         lr_scheduler.step()

#     model = load_model(args).to(device)
#     evaluate_hit_ratio(
#         model=model,
#         data=data,
#         source_edge_index=source_edge_index,
#         target_edge_index=target_train_edge_index,  # ✅ 正確傳入測試集 edge_index
#         top_k=args.top_k,
#         num_candidates=99,
#         device=device,
#     )
#     cold_item_id = 2286
   
#     id = find_cold_item_strict(data, data.target_train_edge_index, data.target_test_edge_index)
#     print("id======", id)

#     if cold_item_id is not None:
#         evaluate_er_hit_ratio(
#             model=model,
#             data=data,
#             source_edge_index=source_edge_index,
#             target_edge_index=target_train_edge_index,
#             cold_item_set={cold_item_id},
#             top_k=args.top_k,
#             num_candidates=99,
#             device=device,
#         )

#     # logging.info(f"Hit Ratio (no injection): {pre_hit_ratio:.4f}")
#     test_auc = evaluate(
#         "Test",
#         model,
#         source_edge_index,
#         target_train_edge_index,
#         target_test_link,
#         target_test_label,
#     )
#     logging.info(f"Test AUC: {test_auc:.4f}")
#     wandb.log({"Test AUC": test_auc})
#     evaluate_multiple_topk(
#     model=model,
#     data=data,
#     source_edge_index=source_edge_index,
#     target_edge_index=target_train_edge_index,
#     cold_item_set={2286},   # 注意這邊是 set，不是 cold_item_id=
#     device=device
#     )