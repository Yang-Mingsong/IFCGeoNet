import os, yaml, torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
import wandb
from data.custom_dataset import IfcFileGraphsDataset
from models.rgcn import RGCN
from utils.early_stopping import EarlyStopping
from scripts.train import RGCNTrainer
from scripts.test import  RGCNTester  # 你的现有训练/测试封装

CFG_PATH = r"Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml"

def deterministic_cos_sin_codes(n_nodes: int, dim: int, device='cpu'):
    """基于局部 gid=0..n-1 的确定性编码（无学习、无表），返回 (n, dim)。"""
    if n_nodes == 0: return torch.empty((0, dim), device=device)
    half = dim // 2
    gids = torch.arange(n_nodes, dtype=torch.float32, device=device).unsqueeze(1)        # (n,1)
    ks   = torch.arange(half, dtype=torch.float32, device=device).unsqueeze(0)           # (1,half)
    M = float(n_nodes + 1)  # 频率归一
    angles = torch.pi * (gids + 1) * (ks + 1) / M                                        # (n,half)
    even = torch.cos(angles)
    odd  = torch.sin(angles)
    out  = torch.empty((n_nodes, dim), dtype=torch.float32, device=device)
    out[:, 0:2*half:2] = even
    out[:, 1:2*half:2] = odd
    if dim % 2 == 1:
        out[:, -1] = torch.cos(torch.pi * (gids.squeeze(1) + 1) * (half + 1) / M)
    return out

class OneHotInitDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, feat_dim):
        self.base = base_ds
        self.feat_dim = feat_dim
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        g, y = self.base[idx]
        # 覆盖所有节点类型特征为确定性编码
        for ntype in g.ntypes:
            n = g.num_nodes(ntype)
            g.nodes[ntype].data['features'] = deterministic_cos_sin_codes(n, self.feat_dim)
        return g, y

def main():
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['RGCN']

    # Initialize wandb
    wandb.init(project=cfg['project_name'], name=cfg['run_name'], config=cfg, mode="offline")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    train_base = IfcFileGraphsDataset('train_dataset', cfg['data_dir'], 'train',
                                      cfg['ifc_file_graph_cache'], cfg['isg_name'], cfg['edge_direction'])
    val_base  = IfcFileGraphsDataset('val_dataset',  cfg['data_dir'], 'val',
                                      cfg['ifc_file_graph_cache'], cfg['isg_name'], cfg['edge_direction'])

    feat_dim = cfg['in_feats']
    train_ds = OneHotInitDataset(train_base, feat_dim)
    val_ds  = OneHotInitDataset(val_base,  feat_dim)

    train_loader = GraphDataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    test_loader  = GraphDataLoader(val_ds, batch_size=cfg['val_batch_size'], shuffle=False, drop_last=False)

    classes = train_base.classes
    metagraph = train_base.re_metagraph if cfg['edge_direction'] == 'REVERSE' else train_base.metagraph
    model = RGCN(cfg['in_feats'], cfg['hidden_feats'], len(classes), metagraph).to(device)

    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, cfg['num_epochs'], eta_min=cfg['eta_min'])
    early_stopping = EarlyStopping(save_path=cfg['save_path'] + '\\early_stopping', patience=cfg['early_stopping'])

    trainer = RGCNTrainer(model, train_loader, optimizer, criterion)
    tester  = RGCNTester(model, test_loader, criterion)

    print('Start training (Deterministic One-hot init).')
    for epoch in range(1, cfg['num_epochs'] + 1):
        trainer.train(epoch, device, cfg['batch_size'])
        scheduler.step()
        if epoch % cfg['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, cfg['test_batch_size'], cfg['save_path'])
            if early_stopping.early_stop:
                print("Early stopping"); break
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()