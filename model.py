import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from ops import op
from supernet import Supernet


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.space = args.space
        self.bn = args.bn
        self.aggr = args.aggr
        self.num_users = args.num_users
        self.num_source_items = args.num_source_items
        self.num_target_items = args.num_target_items

        self.embedding_dim = args.embedding_dim
        self.user_embedding = nn.Embedding(self.num_users, args.embedding_dim)
        self.source_item_embedding = nn.Embedding(
            self.num_source_items, args.embedding_dim
        )
        self.target_item_embedding = nn.Embedding(
            self.num_target_items, args.embedding_dim
        )

        self.source_supernet = Supernet(
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            self.space,
            self.bn,
            self.aggr,
        )
        self.target_supernet = Supernet(
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            self.space,
            self.bn,
            self.aggr,
        )

        self.user_mix_linear = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
        )

        self.pred_input_dim = (
            self.embedding_dim * 2 + self.hidden_dim * self.num_layers * 2
        )
        self.source_preds = nn.ModuleList([nn.Linear(self.pred_input_dim, 1)])
        self.target_preds = nn.ModuleList([nn.Linear(self.pred_input_dim, 1)])
        self.replace_target_preds = nn.ModuleList([nn.Linear(self.pred_input_dim, 1)])

        self.init_parameters()

    def forward(self, source_edge_index, target_edge_index, link, is_source=True):
        source_x = torch.cat(
            [self.user_embedding.weight, self.source_item_embedding.weight], dim=0
        )
        target_x = torch.cat(
            [self.user_embedding.weight, self.target_item_embedding.weight], dim=0
        )

        source_embs = [source_x]
        target_embs = [target_x]
        for i in range(self.num_layers):
            source_x = self.source_supernet.convs[i](source_x, source_edge_index)
            target_x = self.target_supernet.convs[i](target_x, target_edge_index)
            user_emb = self.user_mix_linear[i](
                torch.cat(
                    [source_x[: self.num_users], target_x[: self.num_users]], dim=1
                )
            )
            source_x = torch.cat([user_emb, source_x[self.num_users :]], dim=0)
            target_x = torch.cat([user_emb, target_x[self.num_users :]], dim=0)

            source_embs.append(source_x)
            target_embs.append(target_x)

        source_embs = torch.cat(source_embs, dim=1)
        target_embs = torch.cat(target_embs, dim=1)

        user_embs = source_embs[link[0, :]]
        item_embs = source_embs[link[1, :]] if is_source else target_embs[link[1, :]]
        x = torch.cat([user_embs, item_embs], dim=1)

        preds = self.source_preds if is_source else self.target_preds
        for lin in preds:
            x = lin(x)
            x = F.leaky_relu(x)
        out = x.sigmoid()

        return out

    def meta_prediction(self, source_edge_index, target_edge_index, link):
        source_x = torch.cat(
            [self.user_embedding.weight, self.source_item_embedding.weight]
        )
        target_x = torch.cat(
            [self.user_embedding.weight, self.target_item_embedding.weight]
        )

        source_embs = [source_x]
        target_embs = [target_x]
        for i in range(self.num_layers):
            source_x = self.source_supernet.convs[i](source_x, source_edge_index)
            target_x = self.target_supernet.convs[i](target_x, target_edge_index)
            user_emb = self.user_mix_linear[i](
                torch.cat(
                    [source_x[: self.num_users], target_x[: self.num_users]], dim=1
                )
            )
            source_x = torch.cat([user_emb, source_x[self.num_users :]], dim=0)
            target_x = torch.cat([user_emb, target_x[self.num_users :]], dim=0)

            source_embs.append(source_x)
            target_embs.append(target_x)

        source_embs = torch.cat(source_embs, dim=1)
        target_embs = torch.cat(target_embs, dim=1)

        user_embs = source_embs[link[0, :]]
        item_embs = target_embs[link[1, :]]
        x = torch.cat([user_embs, item_embs], dim=1)

        preds = self.replace_target_preds
        for lin in preds:
            x = lin(x)
            x = F.leaky_relu(x)
        out = x.sigmoid()

        return out

    def init_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.source_item_embedding.weight, std=0.01)
        nn.init.normal_(self.target_item_embedding.weight, std=0.01)

    def print_alpha(self):
        logging.info("source supernet alpha:")
        self.source_supernet.print_alpha()
        logging.info("target supernet alpha:")
        self.target_supernet.print_alpha()
        
    # === CL / Encoder utilities ===
    def encode_full(self, edge_index_1, edge_index_2=None, return_norm=False):
        """
        支援兩種模式：
        1. encode_full(full_edge_index, return_norm=True) → 跨域整圖模式
        2. encode_full(source_edge_index, target_edge_index, return_norm=True) → 分域模式
        """
        num_users = self.num_users
        num_src_items = self.num_source_items
        num_tgt_items = self.num_target_items

        # 1️⃣ 初始化節點嵌入
        source_x = torch.cat([self.user_embedding.weight, self.source_item_embedding.weight], dim=0)
        target_x = torch.cat([self.user_embedding.weight, self.target_item_embedding.weight], dim=0)

        if edge_index_2 is None:
            # === 跨域整圖模式 ===
            full_edge_index = edge_index_1
            # 將 source_x 與 target_x 合併成同一張圖的初始特徵
            x = torch.cat([
                self.user_embedding.weight,
                self.source_item_embedding.weight,
                self.target_item_embedding.weight
            ], dim=0)

            for i in range(self.num_layers):
                x = self.source_supernet.convs[i](x, full_edge_index)  # 用同一套 GNN 跑整圖

            if return_norm:
                x = F.normalize(x, dim=1)

            user_emb = x[:num_users]
            src_item_emb = x[num_users:num_users + num_src_items]
            tgt_item_emb = x[num_users + num_src_items:]
            return user_emb, src_item_emb, tgt_item_emb

        else:
            # === 分域模式 ===
            source_edge_index = edge_index_1
            target_edge_index = edge_index_2

            for i in range(self.num_layers):
                source_x = self.source_supernet.convs[i](source_x, source_edge_index)
                target_x = self.target_supernet.convs[i](target_x, target_edge_index)

                user_emb = self.user_mix_linear[i](
                    torch.cat([source_x[:num_users], target_x[:num_users]], dim=1)
                )

                source_x = torch.cat([user_emb, source_x[num_users:]], dim=0)
                target_x = torch.cat([user_emb, target_x[num_users:]], dim=0)

            if return_norm:
                source_x = F.normalize(source_x, dim=1)
                target_x = F.normalize(target_x, dim=1)

            user_emb = source_x[:num_users]
            src_item_emb = source_x[num_users:]
            tgt_item_emb = target_x[num_users:]
            return user_emb, src_item_emb, tgt_item_emb


    def encode_views(self, se_view1, te_view1, se_view2, te_view2, return_norm=True):
        # view1
        source_x1 = torch.cat([self.user_embedding.weight, self.source_item_embedding.weight], dim=0)
        target_x1 = torch.cat([self.user_embedding.weight, self.target_item_embedding.weight], dim=0)
        for i in range(self.num_layers):
            source_x1 = self.source_supernet.convs[i](source_x1, se_view1)
            target_x1 = self.target_supernet.convs[i](target_x1, te_view1)
        user_v1 = target_x1[:self.num_users]
        source_item_v1 = source_x1[self.num_users:]
        target_item_v1 = target_x1[self.num_users:]

        # view2
        source_x2 = torch.cat([self.user_embedding.weight, self.source_item_embedding.weight], dim=0)
        target_x2 = torch.cat([self.user_embedding.weight, self.target_item_embedding.weight], dim=0)
        for i in range(self.num_layers):
            source_x2 = self.source_supernet.convs[i](source_x2, se_view2)
            target_x2 = self.target_supernet.convs[i](target_x2, te_view2)
        user_v2 = target_x2[:self.num_users]
        source_item_v2 = source_x2[self.num_users:]
        target_item_v2 = target_x2[self.num_users:]

        if return_norm:
            user_v1 = F.normalize(user_v1, dim=1)
            source_item_v1 = F.normalize(source_item_v1, dim=1)
            target_item_v1 = F.normalize(target_item_v1, dim=1)
            user_v2 = F.normalize(user_v2, dim=1)
            source_item_v2 = F.normalize(source_item_v2, dim=1)
            target_item_v2 = F.normalize(target_item_v2, dim=1)

        return (user_v1, source_item_v1, target_item_v1), (user_v2, source_item_v2, target_item_v2)



class Perceptor(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.domain_prior = nn.Parameter(torch.ones(1, 1))
        self.item_prior_convs = nn.ModuleList()
        for _ in range(args.meta_num_layers):
            self.item_prior_convs.append(op(args.meta_op, args.meta_hidden_dim))
        self.item_prior_linear = nn.Linear(args.meta_hidden_dim, 1)

    def forward(self, item, edge_index, ref_model):
        x = torch.cat(
            [ref_model.user_embedding.weight, ref_model.source_item_embedding.weight]
        )
        for conv in self.item_prior_convs:
            x = conv(x, edge_index)
        item_prior = self.item_prior_linear(x)[item]
        return torch.relu(self.domain_prior) * torch.softmax(item_prior, dim=0)
