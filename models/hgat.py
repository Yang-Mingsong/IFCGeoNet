import torch
import torch.nn as nn
from dgl.nn.pytorch import HeteroGraphConv, GATConv


class HeteroGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes, num_heads):
        super(HeteroGATLayer, self).__init__()
        self.hetero_conv = HeteroGraphConv({
            etype: GATConv(in_feats, out_feats, num_heads, allow_zero_in_degree=True)
            for etype in etypes
        })

    def forward(self, g, inputs):
        # inputs 是一个字典，键是节点类型，值是对应的特征张量
        h = self.hetero_conv(g, inputs)
        # 对多头注意力输出进行平均
        for ntype in h:
            h[ntype] = h[ntype].mean(dim=1)
        return h


class HeteroGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, etypes, num_heads):
        super(HeteroGAT, self).__init__()
        self.layer1 = HeteroGATLayer(in_feats, hidden_feats, etypes, num_heads)
        self.layer2 = HeteroGATLayer(hidden_feats, hidden_feats, etypes, num_heads)
        self.layer3 = HeteroGATLayer(hidden_feats, out_feats, etypes, num_heads)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, g, node_feature):
        h = self.layer1(g, node_feature)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}

        h = self.layer2(g, h)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}

        h = self.layer3(g, h)

        return h
