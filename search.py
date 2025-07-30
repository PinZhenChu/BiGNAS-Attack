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


def inject_fake_edges( 
    data,
    source_edge_index,
    target_edge_index,
    user_fraction=0.01,        # 注入多少比例的用戶（raw_overlap_users中選取比例）
    item_fraction=0.5,        # 每個用戶模仿多少比例種子用戶行為
    id2asin=None,
    cold_item_ids=None,
    seed_user=None,
    log_detail=False,
    args=None
):
    import torch
    import random
    from collections import defaultdict, Counter
    import logging

    device = source_edge_index.device
    logger = logging.getLogger(__name__)
    logger.info(f"[inject_fake_edges] Before injection: source_edge_index has {source_edge_index.shape[1]} edges")

    # 確保 cold_item_ids 為列表
    if isinstance(cold_item_ids, int):
        cold_item_ids = [cold_item_ids]

    if args is not None:
        args.cold_item_ids = cold_item_ids

    # 印出冷門商品資訊
    if log_detail:
        logger.info("===== 冷門商品列表 =====")
        for item_id in cold_item_ids:
            asin = id2asin[item_id] if id2asin and item_id in id2asin else f"ID:{item_id}"
            logger.info(f"[RANKING] 冷門商品 ID: {item_id}, ASIN: {asin}")

    # STEP 1: 決定 cold users（或使用指定 seed_user）
    user2cold_items = defaultdict(list)  # 預先宣告，避免引用錯誤

    if seed_user is not None:
        cold_users = [seed_user]
        logger.info(f"[inject_fake_edges] ✅ Using manually specified seed user: {seed_user}")
    else:
        for u, i in zip(target_edge_index[0].tolist(), target_edge_index[1].tolist()):
            if i in cold_item_ids:
                user2cold_items[u].append(i)
        cold_users = list(user2cold_items.keys())
        logger.info(f"[inject_fake_edges] Found {len(cold_users)} cold users: {cold_users}")

    # 除錯印 cold users 行為
    if log_detail:
        logger.info(f"[inject_fake_edges] Cold users and their items:")
        if seed_user is None:
            for cu in cold_users:
                items = user2cold_items.get(cu, [])
                logger.info(f"  User {cu}: Items {items}")
        else:
            # seed_user 指定時顯示其 source domain 行為
            seed_items_debug = list(defaultdict(set).get(seed_user, []))  # 空集合，改用下面user_item_dict
            logger.info(f"  Seed user {seed_user} items (source domain): Not shown yet (will show later)")

    # STEP 2: 建立 source domain 使用者-物品對應表
    user_item_dict = defaultdict(set)
    for u, i in zip(source_edge_index[0].tolist(), source_edge_index[1].tolist()):
        user_item_dict[u].add(i)

    if log_detail:
        logger.info(f"[inject_fake_edges] Sample of user-item mappings (前5用戶):")
        count_show = 0
        for u, items in user_item_dict.items():
            logger.info(f"  User {u}: {list(items)[:10]} (共{len(items)}項目)")
            count_show += 1
            if count_show >= 5:
                break

        # 如有 seed_user，印出其 source 行為
        if seed_user is not None:
            seed_user_items = user_item_dict.get(seed_user, set())
            logger.info(f"[inject_fake_edges] Seed user {seed_user} 行為數量: {len(seed_user_items)}，範例: {list(seed_user_items)[:10]}")

    # STEP 3: 從 raw_overlap_users 選擇 user_fraction 比例用戶注入假邊
    overlap_users_list = data.raw_overlap_users.tolist()
    num_overlap_users = len(overlap_users_list)
    num_fake_users = max(1, int(num_overlap_users * user_fraction))
    logger.info(f"[inject_fake_edges] Total overlap users: {num_overlap_users}")
    logger.info(f"[inject_fake_edges] Selecting {num_fake_users} users (~{user_fraction * 100:.2f}%) to inject fake edges.")
    
    candidate_users = random.sample(overlap_users_list, num_fake_users)
    logger.info(f"[inject_fake_edges] 選中用戶範例（最多前10個）: {candidate_users[:10]}")

    fake_edges = []
    sim_users = set()

    for u2 in candidate_users:
        sim_users.add(u2)
        # 從 cold_users 中隨機選 seed user 以模仿行為
        seed_user_for_injection = random.choice(cold_users)
        seed_items = list(user_item_dict.get(seed_user_for_injection, []))

        # fallback：若該 seed user 無行為，使用 source domain 熱門物品
        if not seed_items:
            top_items = [i for i, _ in Counter(source_edge_index[1].tolist()).most_common(10)]
            seed_items = top_items
            logger.warning(f"[inject_fake_edges] Seed user {seed_user_for_injection} 無 source 行為，改用熱門商品 {top_items}")

        # 按 item_fraction 抽樣種子商品行為
        num_items_to_inject = max(1, int(len(seed_items) * item_fraction))
        sampled_items = random.sample(seed_items, num_items_to_inject)

        if log_detail:
            logger.info(f"[inject_fake_edges] 用戶 {u2} 注入 {num_items_to_inject} 件商品，模仿種子用戶 {seed_user_for_injection}")

        # 假邊加入清單
        for item in sampled_items:
            fake_edges.append((u2, item))

    # STEP 4: 合併假邊進原始邊集
    if fake_edges:
        fake_u, fake_i = zip(*fake_edges)
        fake_u_tensor = torch.tensor(fake_u, dtype=torch.long).to(device)
        fake_i_tensor = torch.tensor(fake_i, dtype=torch.long).to(device)
        fake_edge_index = torch.stack([fake_u_tensor, fake_i_tensor], dim=0)

        new_edge_index = torch.cat([source_edge_index, fake_edge_index], dim=1)
        logger.info(f"[inject_fake_edges] 注入了 {len(fake_edges)} 條假邊")
    else:
        logger.warning("[inject_fake_edges] 未產生任何假邊")
        new_edge_index = source_edge_index

    logger.info(f"[inject_fake_edges] 注入後 source_edge_index 邊數: {new_edge_index.shape[1]}")
    logger.info(f"[inject_fake_edges] 總模擬用戶數: {len(sim_users)}")

    return new_edge_index, cold_item_ids, list(sim_users)

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
    cold_item_ids = [2286]

    # 注入 fake edges
    fake_source_edge_index, cold_items, sim_users = inject_fake_edges(
        data=data,
        source_edge_index=source_edge_index,
        target_edge_index=target_edge_index,
        user_fraction=0.01,      # 注入全部重疊用戶
        item_fraction=1,      # 每個用戶完整模仿種子用戶行為
        cold_item_ids=2286,     # 單一冷門商品ID（int會自動轉list）
        seed_user=124,
        log_detail=True,
        args=args,
    )



    # 使用篡改後的 source_edge_index 進行訓練
    train(model, perceptor, data, args, source_edge_index=fake_source_edge_index)

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
