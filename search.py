import argparse
import logging
import os
import time

import wandb

from collections import Counter

from dataset import CrossDomain
from model import Model, Perceptor
from train import train
from utils import set_logging, set_seed


# def inject_fake_edges( 
#     data,
#     source_edge_index,
#     target_edge_index,
#     user_fraction=0.01,        # 從 raw_overlap_users 中選多少比例 user
#     item_fraction=0.5,         # 每位 user 模仿多少比例 seed user 的行為
#     id2asin=None,
#     cold_item_ids=None,
#     seed_user=None,
#     log_detail=False,
#     args=None
# ):
#     import torch
#     import random
#     from collections import defaultdict, Counter
#     import logging

#     device = source_edge_index.device
#     logger = logging.getLogger(__name__)
#     logger.info(f"[inject_fake_edges] Before injection: source_edge_index has {source_edge_index.shape[1]} edges")

#     # 確保 cold_item_ids 為 list
#     if isinstance(cold_item_ids, int):
#         cold_item_ids = [cold_item_ids]

#     if args is not None:
#         args.cold_item_ids = cold_item_ids

#     # 印出冷門商品資訊
#     if log_detail:
#         logger.info("===== 冷門商品列表 =====")
#         for item_id in cold_item_ids:
#             asin = id2asin[item_id] if id2asin and item_id in id2asin else f"ID:{item_id}"
#             logger.info(f"[RANKING] 冷門商品 ID: {item_id}, ASIN: {asin}")

#     # === STEP 1: 決定 seed user（冷門商品互動者）
#     user2cold_items = defaultdict(list)

#     if seed_user is not None:
#         cold_users = [seed_user]
#         logger.info(f"[inject_fake_edges] ✅ Using manually specified seed user: {seed_user}")
#     else:
#         for u, i in zip(target_edge_index[0].tolist(), target_edge_index[1].tolist()):
#             if i in cold_item_ids:
#                 user2cold_items[u].append(i)
#         cold_users = list(user2cold_items.keys())
#         logger.info(f"[inject_fake_edges] Found {len(cold_users)} cold users: {cold_users}")

#     # === STEP 2: 建立 source domain user-item 對應表
#     user_item_dict = defaultdict(set)
#     for u, i in zip(source_edge_index[0].tolist(), source_edge_index[1].tolist()):
#         user_item_dict[u].add(i)

#     if log_detail:
#         logger.info(f"[inject_fake_edges] Sample of user-item mappings (前5用戶):")
#         for idx, (u, items) in enumerate(user_item_dict.items()):
#             logger.info(f"  User {u}: {list(items)[:10]} (共{len(items)}項目)")
#             if idx >= 4:
#                 break
#         if seed_user is not None:
#             seed_items = user_item_dict.get(seed_user, set())
#             logger.info(f"[inject_fake_edges] Seed user {seed_user} 行為數量: {len(seed_items)}，範例: {list(seed_items)[:10]}")

#     # === STEP 3: 從 overlap users 中選 attack users
#     overlap_users_list = data.raw_overlap_users.tolist()
#     num_fake_users = max(1, int(len(overlap_users_list) * user_fraction))
#     selected_users = random.sample(overlap_users_list, num_fake_users)

#     logger.info(f"[inject_fake_edges] Total overlap users: {len(overlap_users_list)}")
#     logger.info(f"[inject_fake_edges] Selecting {num_fake_users} users (~{user_fraction*100:.2f}%) to inject fake edges.")
#     logger.info(f"[inject_fake_edges] 選中用戶範例（最多前10個）: {selected_users[:10]}")

#     # === STEP 4: 注入 source domain 假邊
#     source_fake_edges = []
#     target_fake_edges = []
#     sim_users = set()

#     for u2 in selected_users:
#         sim_users.add(u2)
#         seed_user_for_injection = random.choice(cold_users)
#         seed_items = list(user_item_dict.get(seed_user_for_injection, []))

#         if not seed_items:
#             top_items = [i for i, _ in Counter(source_edge_index[1].tolist()).most_common(10)]
#             seed_items = top_items
#             logger.warning(f"[inject_fake_edges] Seed user {seed_user_for_injection} 無 source 行為，改用熱門商品 {top_items}")

#         num_items_to_inject = max(1, int(len(seed_items) * item_fraction))
#         sampled_items = random.sample(seed_items, num_items_to_inject)

#         if log_detail:
#             logger.info(f"[inject_fake_edges] 用戶 {u2} 注入 {num_items_to_inject} 件商品，模仿種子用戶 {seed_user_for_injection}")

#         for item in sampled_items:
#             source_fake_edges.append((u2, item))

#         # 同時為 target domain 注入 cold item（已 + offset）
#         for item in cold_item_ids:
#             target_fake_edges.append((u2, item + data.num_users))

#     # === STEP 5: 合併邊集
#     def combine_edges(original, fake):
#         if fake:
#             u, i = zip(*fake)
#             u_tensor = torch.tensor(u, dtype=torch.long).to(device)
#             i_tensor = torch.tensor(i, dtype=torch.long).to(device)
#             fake_tensor = torch.stack([u_tensor, i_tensor], dim=0)
#             return torch.cat([original, fake_tensor], dim=1)
#         return original

#     new_source_edge_index = combine_edges(source_edge_index, source_fake_edges)
#     new_target_edge_index = combine_edges(target_edge_index, target_fake_edges)

#     logger.info(f"[inject_fake_edges] 注入了 {len(source_fake_edges)} 條 source 假邊")
#     logger.info(f"[inject_fake_edges] 注入了 {len(target_fake_edges)} 條 target 假邊")
#     logger.info(f"[inject_fake_edges] 注入後 source_edge_index 邊數: {new_source_edge_index.shape[1]}")
#     logger.info(f"[inject_fake_edges] 注入後 target_edge_index 邊數: {new_target_edge_index.shape[1]}")
#     logger.info(f"[inject_fake_edges] 總模擬用戶數: {len(sim_users)}")

#     return new_source_edge_index, new_target_edge_index, cold_item_ids, list(sim_users)

def inject_fake_edges(
    data,
    model,
    source_edge_index,
    target_edge_index,
    cold_item_id,
    user_fraction=0.01,
    device=None,
    log_detail=True,
    args=None,
    inject_source=True,  # ✅ 是否注入 source 假邊
    inject_target=True,  # ✅ 是否注入 target 假邊
    source_edge_fraction=None, 
):
    import torch
    import random
    import logging

    logger = logging.getLogger(__name__)
    device = device or source_edge_index.device
    num_users = data.num_users

    source_edge_index = source_edge_index.to(device)
    target_edge_index = target_edge_index.to(device)
    model = model.to(device)

    logger.info(f"[inject_fake_edges] Start. Original edges: {source_edge_index.shape[1]}")

    # === Step 1: 選出對 cold item 預測分數最高的 overlap users ===
    overlap_users = data.raw_overlap_users.tolist()
    user_tensor = torch.tensor(overlap_users, device=device)
    item_tensor = torch.tensor([cold_item_id] * len(overlap_users), device=device)
    link = torch.stack([user_tensor, item_tensor], dim=0)

    model.eval()
    with torch.no_grad():
        scores = model(source_edge_index, target_edge_index, link, is_source=False).view(-1)

    top_k = max(1, int(len(overlap_users) * user_fraction))
    _, top_indices = torch.topk(scores, top_k)
    selected_users = [overlap_users[i] for i in top_indices.tolist()]
    logger.info(f"[inject_fake_edges] Selected {len(selected_users)} users with top predicted score for cold item")

    # === Step 2: 找出 seed user（與 cold item 有交互的 user） ===
    tgt_u, tgt_i = data.target_train_edge_index


    # seed_users = [u.item() for u, i in zip(tgt_u, tgt_i) if i.item() == cold_item_id]
    # seed_users = list(set(seed_users))
    seed_users =124
    if not seed_users:
        logger.warning(f"[inject_fake_edges] No seed users found for cold item {cold_item_id}")
        return source_edge_index, target_edge_index, [cold_item_id], selected_users
    seed_user = 124  # 假設只有一個 seed user
    logger.info(f"[inject_fake_edges] Seed user for cold item: {seed_user}")

    # === Step 3: 取得 seed user 在 source domain 的所有行為 ===
    su, si = source_edge_index[0].tolist(), source_edge_index[1].tolist()
    seed_source_items = [i for u, i in zip(su, si) if u == seed_user]
    seed_source_items = list(set(seed_source_items))
    logger.info(f"[inject_fake_edges] Seed user has {len(seed_source_items)} source domain items")

    # === Step 4: 建立 fake edge 給 target domain 攻擊用戶 ===
    fake_source_edges = [(u, i) for u in selected_users for i in seed_source_items]
    fake_target_edges = [(u, cold_item_id) for u in selected_users]

    # === Step 5: 合併 source 假邊（視條件注入） ===
    # === Step 5: 合併 source 假邊（視條件注入） ===
    if inject_source and fake_source_edges:
        # ✅ 隨機抽取指定比例的假邊
        if source_edge_fraction < 1.0:
            sample_size = int(len(fake_source_edges) * source_edge_fraction)
            fake_source_edges = random.sample(fake_source_edges, sample_size)
            logger.info(f"[inject_fake_edges] 僅使用 {sample_size} 條 source 假邊 (比例={source_edge_fraction})")

        su, si = zip(*fake_source_edges)
        su_tensor = torch.tensor(su, dtype=torch.long, device=device)
        si_tensor = torch.tensor(si, dtype=torch.long, device=device)
        new_source_edge_index = torch.cat(
            [source_edge_index, torch.stack([su_tensor, si_tensor], dim=0)], dim=1
        )
        logger.info(f"[inject_fake_edges] 注入 {len(fake_source_edges)} 條 source 假邊 (copy seed user 行為)")
    else:
        new_source_edge_index = source_edge_index
        logger.info(f"[inject_fake_edges] 未注入 source 假邊")

    # === Step 6: 合併 target 假邊（視條件注入） ===
    if inject_target and fake_target_edges:
        tu, ti = zip(*fake_target_edges)
        tu_tensor = torch.tensor(tu, dtype=torch.long, device=device)
        ti_tensor = torch.tensor(ti, dtype=torch.long, device=device)
        new_target_edge_index = torch.cat(
            [target_edge_index, torch.stack([tu_tensor, ti_tensor], dim=0)], dim=1
        )
        logger.info(f"[inject_fake_edges] 注入 {len(fake_target_edges)} 條 target 假邊 (cold item)")
    else:
        new_target_edge_index = target_edge_index
        logger.info(f"[inject_fake_edges] 未注入 target 假邊")

    logger.info(f"[inject_fake_edges] 新 source_edge_index 邊數: {new_source_edge_index.shape[1]}")
    logger.info(f"[inject_fake_edges] 新 target_edge_index 邊數: {new_target_edge_index.shape[1]}")

    return new_source_edge_index, new_target_edge_index, [cold_item_id], selected_users

def search(args):
    args.search = True

    wandb.init(project="BiGNAS", config=args)
    set_seed(args.seed)
    set_logging()

    logging.info(f"args: {args}")

    dataset = CrossDomain(
        root=args.root,
        categories=args.categories,
        target=args.target,
        use_source=args.use_source,
    )

    data = dataset[0]
    args.num_users = data.num_users
    args.num_source_items = data.num_source_items
    args.num_target_items = data.num_target_items
    logging.info(f"data: {data}")

    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
    args.model_path = os.path.join(
        args.model_dir,
        f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
    )

    model = Model(args)
    perceptor = Perceptor(args)
    logging.info(f"model: {model}")

    # 取出 source 與 target 的 edge_index
    source_edge_index = data.source_link.to(args.device)  # 不是 source_edge_index，是 source_link

    target_edge_index = data.target_train_edge_index.to(args.device)

    # 你想要注入的冷門商品，可以改成你自己的ID列表
    # cold_item_ids  = args.cold_item_ids
    # 處理冷門商品 ID 列表（來自 argparse）
    cold_item_ids = args.cold_item_ids if isinstance(args.cold_item_ids, list) else [args.cold_item_ids]
    cold_item_id = cold_item_ids[0]  # 目前只支援單一 cold item


    # 注入 fake edges
    # fake_source_edge_index, fake_target_edge_index, cold_items, sim_users = inject_fake_edges(
    #     data=data,
    #     source_edge_index=source_edge_index,
    #     target_edge_index=target_edge_index,
    #     user_fraction=0.01,
    #     item_fraction=1,
    #     cold_item_ids=cold_item_ids,
    #     seed_user=124,
    #     log_detail=True,
    #     args=args,
    # )
    fake_source_edge_index, fake_target_edge_index, cold_items, sim_users = inject_fake_edges(
        data=data,
        model=model,
        source_edge_index=source_edge_index,
        target_edge_index=target_edge_index,
        cold_item_id=cold_item_id,
        user_fraction=0.01,
        device=args.device,
        args=args,
        inject_source=True,
        inject_target=True,
        source_edge_fraction=1
    )






    # 使用篡改後的 source_edge_index 進行訓練
    train(
        model,
        perceptor,
        data,
        args,
        source_edge_index=fake_source_edge_index,
        target_edge_index=fake_target_edge_index,  # ✅ 加上這行
    )


##############################
# def search(args):
#     args.search = True

#     wandb.init(project="BiGNAS", config=args)
#     set_seed(args.seed)
#     set_logging()

#     logging.info(f"args: {args}")

#     dataset = CrossDomain(
#         root=args.root,
#         categories=args.categories,
#         target=args.target,
#         use_source=args.use_source,
#     )

#     data = dataset[0]
#     args.num_users = data.num_users
#     args.num_source_items = data.num_source_items
#     args.num_target_items = data.num_target_items
#     logging.info(f"data: {data}")

#     DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
#     args.model_path = os.path.join(
#         args.model_dir,
#         f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
#     )

#     model = Model(args)
#     perceptor = Perceptor(args)
#     logging.info(f"model: {model}")
#     train(model, perceptor, data, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device & mode settings
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--search", default=False, action="store_true")
    parser.add_argument("--use-meta", default=False, action="store_true")
    parser.add_argument("--use-source", default=False, action="store_true")

    # dataset settings
    parser.add_argument(
        "--categories", type=str, nargs="+", default=["Electronic", "Clothing"]
    )
    parser.add_argument("--target", type=str, default="Clothing")
    parser.add_argument("--root", type=str, default="data/")

    # model settings
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./save/")

    # supernet settings
    parser.add_argument(
        "--space",
        type=str,
        nargs="+",
        default=["gcn", "gatv2", "sage", "lightgcn", "linear"],
    )
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--entropy", type=float, default=0.0)

    # training settings
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=15, help="Top-K for hit ratio evaluation")
    parser.add_argument("--cold_item_ids", type=int, default=None, help="冷門商品的 target item ID")

    # meta settings
    parser.add_argument("--meta-interval", type=int, default=50)
    parser.add_argument("--meta-num-layers", type=int, default=2)
    parser.add_argument("--meta-hidden-dim", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--conv-lr", type=float, default=1)
    parser.add_argument("--hpo-lr", type=float, default=0.01)
    parser.add_argument("--descent-step", type=int, default=10)
    parser.add_argument("--meta-op", type=str, default="gat")

    args = parser.parse_args()
    search(args)
