import os
import torch
import yaml
import random
import torch.nn as nn
from collections import defaultdict
from data.custom_dataset import IfcFileGraphsDataset


# ===== MMD计算 =====
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
    total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(L2_distance.data) / (total.size(0)**2 - total.size(0))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    return sum([torch.exp(-L2_distance / bw) for bw in bandwidth_list]) / len(bandwidth_list)


def compute_mmd_loss(source, target):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    return torch.mean(XX + YY - XY - YX)


# ===== 随机下采样函数 =====
def subsample_tensor(tensor, num_samples):
    if tensor.size(0) > num_samples:
        indices = torch.randperm(tensor.size(0))[:num_samples]
        return tensor[indices]
    else:
        return tensor


# ===== 可训练的映射器 =====
class FeatureMapper(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===== 加载配置 =====
with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
ifg_Config = config['RGCN']

# ===== 加载 ISG 特征 =====
isg_data = torch.load(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\IfcSchemaGraph_embedding\55_REVERSE_IFCSchemaGraph_HGAT_128_512_128_nce.bin")  # 假设保存的是 dict
isg_class = torch.stack(list(isg_data['class_node'].values()))
isg_attr = torch.stack(list(isg_data['attribute_node'].values()))
isg_type = torch.stack(list(isg_data['type_node'].values()))
isg_value = torch.stack(list(isg_data['value_node'].values()))

print(f"ISG - class: {isg_class.shape}, attr: {isg_attr.shape}")

# ===== 加载 IFG 数据集 =====
dataset = IfcFileGraphsDataset(
    name='train_dataset',
    root=ifg_Config['data_dir'],
    data_type='train',
    c_dir=ifg_Config['ifc_file_graph_cache'],
    isg=ifg_Config['isg_name'],
    edge_direction=ifg_Config['edge_direction']
)

# ===== 按类采样 5个图 =====
label_to_indices = defaultdict(list)
for idx, label in enumerate(dataset.labels):
    label_to_indices[label].append(idx)

sampled_indices = []
for label, indices in label_to_indices.items():
    sampled_indices.extend(random.sample(indices, min(5, len(indices))))

# ===== 提取 IFG 中 class_node 与 attribute_node 特征 =====
ifg_class_feats, ifg_attr_feats = [], []
ifg_type_feats, ifg_value_feats = [], []
for idx in sampled_indices:
    g, label = dataset[idx]
    if 'class_node' in g.ntypes and g.num_nodes('class_node') > 0:
        ifg_class_feats.append(g.nodes['class_node'].data['features'])
    if 'attribute_node' in g.ntypes and g.num_nodes('attribute_node') > 0:
        ifg_attr_feats.append(g.nodes['attribute_node'].data['features'])
    if 'type_node' in g.ntypes and g.num_nodes('type_node') > 0:
        ifg_class_feats.append(g.nodes['type_node'].data['features'])
    if 'value_node' in g.ntypes and g.num_nodes('value_node') > 0:
        ifg_attr_feats.append(g.nodes['value_node'].data['features'])

ifg_class = torch.cat(ifg_class_feats, dim=0)
ifg_attr = torch.cat(ifg_attr_feats, dim=0)
ifg_type = torch.cat(ifg_class_feats, dim=0)
ifg_value = torch.cat(ifg_attr_feats, dim=0)

print(f"IFG - class: {ifg_class.shape}, attr: {ifg_attr.shape}")

# ===== 训练 MMD 映射器 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mapper = FeatureMapper(input_dim=isg_class.shape[1]).to(device)
optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-3)

# ===== 拼接并统一下采样大小 =====
source_feats = torch.cat([isg_class, isg_attr, isg_type, isg_value], dim=0)
target_feats = torch.cat([ifg_class, ifg_attr, ifg_type, ifg_value], dim=0)

sample_size = 1000  # 可调节大小
min_len = min(source_feats.size(0), target_feats.size(0), sample_size)

source_feats = subsample_tensor(source_feats, min_len).to(device)
target_feats = subsample_tensor(target_feats, min_len).to(device)

print(f"Sampled for MMD - source: {source_feats.shape}, target: {target_feats.shape}")

# ===== 开始训练 =====
for epoch in range(1001):
    mapper.train()
    mapped = mapper(source_feats)
    loss = compute_mmd_loss(mapped, target_feats)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] MMD Loss: {loss.item():.4f}")

# ===== 保存映射器 =====
save_path = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\4domain_mapper.pth"
torch.save(mapper.state_dict(), save_path)
print(f"Saved domain mapper to {save_path}")
