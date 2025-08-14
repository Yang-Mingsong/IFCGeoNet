# -*- coding: utf-8 -*-
import os, yaml, torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn
from dgl.dataloading import GraphDataLoader
import wandb
from data.custom_dataset import IfcFileGraphsDataset
from models.rgcn import RGCN
from utils.early_stopping import EarlyStopping
from scripts.train import RGCNTrainer
from scripts.test import  RGCNTester  # 你的现有训练/测试封装

# -------- 配置 --------
CFG_PATH = r"/configs/config.yaml"
META_WEIGHTS = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\train_meta_init.pth"
# ---------------------

def main():
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['RGCN']
    # Initialize wandb
    wandb.init(project=cfg['project_name'], name=cfg['run_name'], config=cfg, mode="offline")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    train_dataset = IfcFileGraphsDataset('train_dataset', cfg['data_dir'], 'train',
                                         cfg['ifc_file_graph_cache'], cfg['isg_name'], cfg['edge_direction'])
    val_dataset  = IfcFileGraphsDataset('val_dataset',  cfg['data_dir'], 'val',
                                         cfg['ifc_file_graph_cache'], cfg['isg_name'], cfg['edge_direction'])

    train_loader = GraphDataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    val_loader  = GraphDataLoader(val_dataset,  batch_size=cfg['val_batch_size'], shuffle=False, drop_last=False)

    classes = train_dataset.classes
    metagraph = train_dataset.re_metagraph if cfg['edge_direction'] == 'REVERSE' else train_dataset.metagraph
    model = RGCN(cfg['in_feats'], cfg['hidden_feats'], len(classes), metagraph).to(device)

    # 加载 meta 预训练权重
    try:
        sd = torch.load(META_WEIGHTS, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(META_WEIGHTS, map_location=device)
    model.load_state_dict(sd, strict=False)
    print("Loaded meta-initialized weights.")

    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, cfg['num_epochs'], eta_min=cfg['eta_min'])
    early_stopping = EarlyStopping(save_path=cfg['save_path'] + '\\early_stopping', patience=cfg['early_stopping'])

    trainer = RGCNTrainer(model, train_loader, optimizer, criterion)
    tester  = RGCNTester(model, val_loader, criterion)

    print('Start training (Meta init).')
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
