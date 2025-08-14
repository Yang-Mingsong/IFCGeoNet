import os
import torch
import yaml
import torch.nn as nn
from torch.backends import cudnn
from data.custom_dataset import IfcFileGraphsDataset
from models.rgcn import RGCN


# ===== 可训练的映射器结构（与训练用保持一致） =====
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


# ===== 特征替换逻辑 =====
@torch.no_grad()
def replace_features_with_mapped(g, isg_class, isg_attr, isg_type, isg_value, mapper, device):
    if 'class_node' in g.ntypes:
        ids = g.nodes('class_node').tolist()
        old_feats = g.nodes['class_node'].data['features']
        new_feats = []
        for i, nid in enumerate(ids):
            key = str(nid)
            if key in isg_class:
                src_feat = isg_class[key].unsqueeze(0).to(device)
                mapped = mapper(src_feat).squeeze(0).cpu()
                new_feats.append(mapped)
            else:
                new_feats.append(old_feats[i])
        g.nodes['class_node'].data['features'] = torch.stack(new_feats)

    if 'attribute_node' in g.ntypes:
        ids = g.nodes('attribute_node').tolist()
        old_feats = g.nodes['attribute_node'].data['features']
        new_feats = []
        for i, nid in enumerate(ids):
            key = str(nid)
            if key in isg_attr:
                src_feat = isg_attr[key].unsqueeze(0).to(device)
                mapped = mapper(src_feat).squeeze(0).cpu()
                new_feats.append(mapped)
            else:
                new_feats.append(old_feats[i])
        g.nodes['attribute_node'].data['features'] = torch.stack(new_feats)

    if 'type_node' in g.ntypes:
        ids = g.nodes('type_node').tolist()
        old_feats = g.nodes['type_node'].data['features']
        new_feats = []
        for i, nid in enumerate(ids):
            key = str(nid)
            if key in isg_type:
                src_feat = isg_type[key].unsqueeze(0).to(device)
                mapped = mapper(src_feat).squeeze(0).cpu()
                new_feats.append(mapped)
            else:
                new_feats.append(old_feats[i])
        g.nodes['type_node'].data['features'] = torch.stack(new_feats)

    if 'value_node' in g.ntypes:
        ids = g.nodes('value_node').tolist()
        old_feats = g.nodes['value_node'].data['features']
        new_feats = []
        for i, nid in enumerate(ids):
            key = str(nid)
            if key in isg_value:
                src_feat = isg_value[key].unsqueeze(0).to(device)
                mapped = mapper(src_feat).squeeze(0).cpu()
                new_feats.append(mapped)
            else:
                new_feats.append(old_feats[i])
        g.nodes['value_node'].data['features'] = torch.stack(new_feats)

    return g


# ===== 主提取逻辑 =====
@torch.no_grad()
def extract_graph_features(model, dataset, mapper, isg_class, isg_attr, isg_type, isg_value,  device, save_path):
    model.eval()
    features, labels = [], []
    for g, label in dataset:
        g = replace_features_with_mapped(g, isg_class, isg_attr, isg_type, isg_value,  mapper, device)
        g = g.to(device)
        graph_embedding = model(g)
        features.append(graph_embedding.cpu())
        labels.append(label)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    torch.save({'features': features, 'labels': labels}, save_path)
    print(f"Saved domain-initialized graph features to {save_path}")


# ===== Main Entry Point =====
if __name__ == "__main__":
    with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    ifg_Config = config['RGCN']

    os.environ["CUDA_VISIBLE_DEVICES"] = ifg_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    # ===== 加载 val 数据集 =====
    val_dataset = IfcFileGraphsDataset(
        name='test_dataset',
        root=ifg_Config['data_dir'],
        data_type='val',
        c_dir=ifg_Config['ifc_file_graph_cache'],
        isg=ifg_Config['isg_name'],
        edge_direction=ifg_Config['edge_direction']
    )
    classes = val_dataset.classes
    metagraph = val_dataset.re_metagraph

    print('Creating model...')
    model = RGCN(
        in_dim=ifg_Config['in_feats'],
        hidden_dim=ifg_Config['hidden_feats'],
        n_classes=len(classes),
        rel_names=metagraph
    ).to(device)

    # ===== 加载 ISG 特征与训练好的 Mapper =====
    isg_data = torch.load(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\IfcSchemaGraph_embedding\55_REVERSE_IFCSchemaGraph_HGAT_128_512_128_nce.bin")
    isg_class = isg_data['class_node']
    isg_attr = isg_data['attribute_node']
    isg_type = isg_data['type_node']
    isg_value = isg_data['value_node']

    mapper = FeatureMapper(input_dim=ifg_Config['in_feats']).to(device)
    mapper.load_state_dict(torch.load(r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\4domain_mapper.pth"))
    mapper.eval()

    # ===== 执行提取并保存 =====
    save_path = r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\4val_domaininit_features.pt'
    extract_graph_features(model, val_dataset, mapper, isg_class, isg_attr, isg_type, isg_value, device, save_path)
