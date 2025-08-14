import torch
import yaml
import os
from torch.backends import cudnn
from data.custom_dataset import IfcFileGraphsDataset
from models.rgcn import RGCN


# Load config
with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
ifg_Config = config.get('RGCN')


@torch.no_grad()
def extract_graph_features(model, dataset, device, save_path):
    model.eval()
    features = []
    labels = []
    for g, label in dataset:
        g = g.to(device)
        graph_embedding = model(g)
        features.append(graph_embedding.cpu())
        labels.append(label)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    torch.save({'features': features, 'labels': labels}, save_path)
    print(f"Saved meta-initialized graph features to {save_path}")

# ===== Main Entry Point =====
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ifg_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    print('Creating graph...')
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
    print('Creating model...', ifg_Config['edge_direction'])
    model = RGCN(
        in_dim=ifg_Config['in_feats'],
        hidden_dim=ifg_Config['hidden_feats'],
        n_classes=len(classes),
        rel_names=metagraph
    ).to(device)

    extract_graph_features(model, val_dataset, device, save_path=r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\val_random_features.pt')
