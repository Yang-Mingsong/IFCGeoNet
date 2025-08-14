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

# ------------ 配置 ------------
CFG_PATH = r"Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml"
MAPPER_PATH = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\4domain_mapper.pth"
# ----------------------------

# 与你训练 MMD 时一致的映射器
class FeatureMapper(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x): return self.net(x)

def ensure_gid_present(g):
    # 若数据集中已写入则不会覆盖；否则给个可复现的局部 id
    for ntype in g.ntypes:
        if '_gid' not in g.nodes[ntype].data:
            g.nodes[ntype].data['_gid'] = torch.arange(g.num_nodes(ntype), dtype=torch.long)

# 包装一个数据集：在 __getitem__ 里对图做 MMD 特征映射
class MMDInitDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, mapper_path, feat_dim):
        self.base = base_ds
        self.feat_dim = feat_dim
        self.mapper = FeatureMapper(feat_dim, 128).eval()
        # 安全加载（避免 warning）
        try:
            sd = torch.load(mapper_path, map_location='cpu', weights_only=True)
        except TypeError:
            sd = torch.load(mapper_path, map_location='cpu')
        self.mapper.load_state_dict(sd)
        for p in self.mapper.parameters(): p.requires_grad_(False)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        g, y = self.base[idx]
        # 仅映射 class_node 与 attribute_node，其它保持原 ISG 初始化
        with torch.no_grad():
            for ntype in ('class_node', 'attribute_node', "type_node", "value_node"):
                if ntype in g.ntypes and g.num_nodes(ntype) > 0:
                    f = g.nodes[ntype].data['features'].float()
                    g.nodes[ntype].data['features'] = self.mapper(f).float()
        return g, y

def main():
    # 读配置
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['RGCN']

    # Initialize wandb
    wandb.init(project=cfg['project_name'], name=cfg['run_name'], config=cfg,  mode="offline")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    # 数据
    train_base = IfcFileGraphsDataset('train_dataset', cfg['data_dir'], 'train',
                                      cfg['ifc_file_graph_cache'], cfg['isg_name'], cfg['edge_direction'])
    val_base  = IfcFileGraphsDataset('val_dataset',  cfg['data_dir'], 'val',
                                      cfg['ifc_file_graph_cache'], cfg['isg_name'], cfg['edge_direction'])
    feat_dim = cfg['in_feats']

    train_ds = MMDInitDataset(train_base, MAPPER_PATH, feat_dim)
    val_ds  = MMDInitDataset(val_base,  MAPPER_PATH, feat_dim)

    train_loader = GraphDataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    val_loader  = GraphDataLoader(val_ds, batch_size=cfg['val_batch_size'], shuffle=False, drop_last=False)

    # 模型
    classes = train_base.classes
    metagraph = train_base.re_metagraph if cfg['edge_direction'] == 'REVERSE' else train_base.metagraph
    model = RGCN(cfg['in_feats'], cfg['hidden_feats'], len(classes), metagraph).to(device)

    # 训练组件
    optimizer = Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, cfg['num_epochs'], eta_min=cfg['eta_min'])
    early_stopping = EarlyStopping(save_path=cfg['save_path'] + '\\early_stopping', patience=cfg['early_stopping'])

    trainer = RGCNTrainer(model, train_loader, optimizer, criterion)
    tester  = RGCNTester(model, val_loader, criterion)

    # 训练
    print('Start training (MMD init).')
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
