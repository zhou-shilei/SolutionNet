# -*- coding: utf-8 -*-
from __future__ import annotations

"""SolutionNet network definition compatible with the deployed checkpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool


class SolutionNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_channels = hidden_channels if hidden_channels % 2 == 0 else hidden_channels + 1
        self.dropout = float(dropout)

        # Intra-molecule GAT
        self.intra_conv1 = GATConv(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            edge_dim=edge_dim,
            heads=2,
            concat=True,
            dropout=self.dropout,
        )
        self.intra_conv2 = GATConv(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels,
            edge_dim=edge_dim,
            heads=1,
            concat=False,
            dropout=self.dropout,
        )

        # Force / H-bond
        self.force_proj = nn.Linear(6, self.hidden_channels)
        self.hbond_proj = nn.Linear(4, self.hidden_channels)
        self.feat_fusion = nn.Linear(self.hidden_channels * 2, self.hidden_channels)

        # Inter-molecule GAT
        self.inter_conv = GATConv(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            edge_dim=self.hidden_channels,
            heads=2,
            concat=False,
            dropout=self.dropout,
            add_self_loops=False,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels * 6, self.hidden_channels * 4),
            nn.SiLU(),
            nn.BatchNorm1d(self.hidden_channels * 4),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels * 4, self.hidden_channels * 2),
            nn.SiLU(),
            nn.BatchNorm1d(self.hidden_channels * 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels * 2, 1),
            nn.Sigmoid(),
        )

        # Residual
        self.residual_intra = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.residual_inter = nn.Linear(self.hidden_channels, self.hidden_channels)

    @staticmethod
    def build_interaction_edges(batch_size: int, device: torch.device) -> torch.Tensor:
        edges = []
        interaction_pairs = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (1, 2), (2, 1),
        ]
        for i in range(batch_size):
            offset = i * 3
            for u, v in interaction_pairs:
                edges.append([offset + u, offset + v])
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def forward(self, batch_A, batch_B, batch_C, batch_i, batch_H):
        device = next(self.parameters()).device
        batch_size = int(batch_A.num_graphs)

        # Intra feature extractor
        def process_intra(batch):
            x = batch.x.float().to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.float().to(device)

            x1 = F.silu(self.intra_conv1(x, edge_index, edge_attr))      # [n, 2*hidden]
            x2 = F.silu(self.intra_conv2(x1, edge_index, edge_attr))     # [n, hidden]

            add_pool = global_add_pool(x2, batch.batch.to(device))
            mean_pool = global_mean_pool(x2, batch.batch.to(device))
            max_pool = global_max_pool(x2, batch.batch.to(device))
            x_pool = add_pool + mean_pool + max_pool                      # [bs, hidden]

            return x_pool + self.residual_intra(x1.mean(dim=0, keepdim=True))

        x_A = process_intra(batch_A)
        x_B = process_intra(batch_B)
        x_C = process_intra(batch_C)

        # Force features
        dist_keys = ['AB', 'AC', 'BC', 'AA', 'BB', 'CC']
        dist = self.extract_dist_features(batch_i, dist_keys, batch_size, device)   # [bs, 6]
        force_feat = F.silu(self.force_proj(dist))                                  # [bs, hidden]

        # H-bond features (4 dims: HAB only)
        hbond_feat = self.extract_hbond_features(batch_H, batch_size, device)       # [bs, 24]
        hbond_feat = F.silu(self.hbond_proj(hbond_feat))                             # [bs, hidden]

        combined_feat = F.silu(self.feat_fusion(torch.cat([force_feat, hbond_feat], dim=1)))  # [bs, hidden]

        # Inter graph
        inter_nodes = torch.cat([x_A, x_B, x_C], dim=0)                              # [3*bs, hidden]
        inter_edge_index = self.build_interaction_edges(batch_size, device)         # [2, 6*bs]

        # Edge attributes (match your posted logic)
        edge_attr_list = []
        for i in range(batch_size):
            for j in range(3):
                spec_force = F.silu(force_feat[i, j].unsqueeze(0) * combined_feat[i])  # scalar*hidden -> [hidden]
                edge_attr = F.silu(spec_force + combined_feat[i])                      # [hidden]
                edge_attr_list.extend([edge_attr, edge_attr])

        inter_edge_attr = torch.stack(edge_attr_list, dim=0).to(device)              # [6*bs, hidden]

        inter_out = F.silu(self.inter_conv(inter_nodes, inter_edge_index, inter_edge_attr))
        inter_out = inter_out + self.residual_inter(inter_nodes)

        x_A_inter = inter_out[::3]
        x_B_inter = inter_out[1::3]
        x_C_inter = inter_out[2::3]

        combined = torch.cat([x_A, x_A_inter, x_B, x_B_inter, x_C, x_C_inter], dim=1)
        return self.classifier(combined)

    def extract_hbond_features(self, batch_H, batch_size: int, device: torch.device) -> torch.Tensor:
        feature_names = [
            'HAB_H_num', 'HAB_H_aver', 'HAB_H_min', 'HAB_H_max',
        ]
        features = []
        for i in range(batch_size):
            sample_feats = []
            for feat_name in feature_names:
                if hasattr(batch_H, feat_name):
                    val = getattr(batch_H, feat_name)[i].item()
                else:
                    val = 0.0
                sample_feats.append(torch.tensor(val, device=device, dtype=torch.float32))
            features.append(torch.stack(sample_feats))
        return torch.stack(features)

    def extract_dist_features(self, feature_dict, keys, batch_size: int, device: torch.device) -> torch.Tensor:
        features = torch.zeros(batch_size, 6, device=device, dtype=torch.float32)
        for i, key in enumerate(keys):
            if hasattr(feature_dict, 'get'):
                val = feature_dict.get(key, 0.0)
            else:
                val = getattr(feature_dict, key, 0.0)

            if isinstance(val, torch.Tensor):
                v = val.to(device).float().view(-1)
                if v.numel() > 1:
                    std = v.std(unbiased=False)
                    if std > 1e-8:
                        v = (v - v.mean()) / (std + 1e-8)
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

                if v.numel() >= batch_size:
                    features[:, i] = v[:batch_size]
                elif v.numel() > 0:
                    features[:, i] = v.mean().repeat(batch_size)
                else:
                    features[:, i] = 0.0
            else:
                features[:, i] = float(val)

        return features
