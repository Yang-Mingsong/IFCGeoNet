import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from data.custom_dataset import IfcFileGraphsDataset
from models.rgcn import RGCN
import torch.nn as nn

# ----------------------- 可配置区 -----------------------
CFG_PATH = r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml'

# 确定性 one-hot 的全局嵌入表（可选）
DET_GID_TABLE = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\deterministic_gid_embeddings.pt"

# MMD 学到的映射器（可选）
MMD_MAPPER_PATH = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\4domain_mapper.pth"

# 元学习保存的权重（必需用于 Meta 可视化）
META_MODEL_PATH = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\train_meta_init.pth"

# 输出图片目录
OUT_DIR = r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\figs_ntype"
os.makedirs(OUT_DIR, exist_ok=True)

# t-SNE 参数
MAX_PER_TYPE = 30000
TSNE_PERPLEXITY = 30
TSNE_SEED = 42

# 选择一个 IFG 图的 index（val 集）
IFG_INDEX = 0
# ------------------------------------------------------


# 简单的 MMD 映射器（需与你训练时的定义一致）
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


def load_cfg():
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['RGCN']


def sample_per_type(feats_by_type, max_per_type=MAX_PER_TYPE, seed=TSNE_SEED):
    """ feats_by_type: dict[ntype] -> (N_i, D) tensor """
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    type_to_id = {'class_node':0,'attribute_node':1,'type_node':2,'value_node':3}
    for ntype, feats in feats_by_type.items():
        if feats is None or feats.numel() == 0:
            continue
        n = feats.shape[0]
        if n > max_per_type:
            idx = rng.choice(n, max_per_type, replace=False)
            sel = feats[idx]
        else:
            sel = feats
        X_list.append(sel)
        y_list.append(torch.full((sel.shape[0],), type_to_id.get(ntype, -1), dtype=torch.long))
    if not X_list:
        return None, None
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def run_tsne_and_save(X, y, title, outpath):
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init='pca',
                learning_rate='auto', random_state=TSNE_SEED)
    X_2d = tsne.fit_transform(X_np)
    plt.figure(figsize=(8, 7))


    cmap = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red'}
    labels = {0:'class',1:'attribute',2:'type',3:'value'}
    for k in np.unique(y_np):
        mask = (y_np == k)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=20, alpha=0.9, c=cmap.get(k, 'gray'), label=labels.get(k,str(k)))
    plt.legend(markerscale=2)
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
    print(f"Saved: {outpath}")


def ensure_gid_present(g):
    for ntype in g.ntypes:
        if '_gid' not in g.nodes[ntype].data:
            g.nodes[ntype].data['_gid'] = torch.arange(g.num_nodes(ntype), dtype=torch.long, device=g.device)


def get_ifg_graph(cfg, index=0, split='val'):
    dataset = IfcFileGraphsDataset(
        name=f'{split}_dataset',
        root=cfg['data_dir'],
        data_type=split,
        c_dir=cfg['ifc_file_graph_cache'],
        isg=cfg['isg_name'],
        edge_direction=cfg['edge_direction']
    )
    g, lable = dataset[index]
    print(lable)
    return g


def collect_feats_by_ntype(g):
    out = {}
    for ntype in g.ntypes:
        if 'features' in g.nodes[ntype].data and g.num_nodes(ntype)>0:
            out[ntype] = g.nodes[ntype].data['features'].float().cpu()
        else:
            out[ntype] = torch.empty(0,0)
    return out


def init_random_features(g, dim=128, seed=2024):
    gen = torch.Generator(device=g.device)
    gen.manual_seed(seed)
    for ntype in g.ntypes:
        n = g.num_nodes(ntype)
        if n>0:
            g.nodes[ntype].data['features'] = torch.randn(n, dim, generator=gen, device=g.device)


def init_onehot_deterministic(g, table_path=DET_GID_TABLE, dim=128):
    """
    确定性 one-hot（降维后）：
    - 若存在预先生成的全局表（建议），用其；
    - 否则针对**当前图**按 _gid 即时用 cos/sin 规则生成（确定性、轻量）。
    """
    ensure_gid_present(g)
    if os.path.exists(table_path):
        table = torch.load(table_path)  # {ntype: {gid: tensor(dim)}}
        for ntype in g.ntypes:
            gids = g.nodes[ntype].data['_gid'].tolist()
            vecs = []
            default = torch.zeros(dim)
            ttab = table.get(ntype, {})
            for gid in gids:
                v = ttab.get(int(gid), default)
                if not torch.is_tensor(v):
                    v = torch.tensor(v, dtype=torch.float32)
                vecs.append(v)
            g.nodes[ntype].data['features'] = torch.stack(vecs, dim=0).to(g.device)
    else:
        # 图内即时构造（确定性）
        half = dim // 2
        for ntype in g.ntypes:
            gids = g.nodes[ntype].data['_gid'].long()
            M = float(max(int(gids.max().item()) + 2, 2))
            ks = torch.arange(half, device=g.device, dtype=torch.float32) + 1.0
            emb_list = []
            for gid in gids.tolist():
                gidf = float(gid + 1)
                angle = (torch.pi * gidf * ks) / M  # (half,)
                even = torch.cos(angle)
                odd  = torch.sin(angle)
                vec = torch.empty(dim, device=g.device, dtype=torch.float32)
                vec[0:2*half:2] = even
                vec[1:2*half:2] = odd
                if dim % 2 == 1:
                    vec[-1] = torch.cos(torch.pi * gidf * (half+1) / M)
                emb_list.append(vec)
            g.nodes[ntype].data['features'] = torch.stack(emb_list, dim=0)


def apply_mmd_mapper_on_graph(g, mapper_path=MMD_MAPPER_PATH):
    if not os.path.exists(mapper_path):
        print(f"[Skip mmd] mapper not found: {mapper_path}")
        return
    # 读取维度
    any_ntype = next((nt for nt in g.ntypes if g.num_nodes(nt)>0), None)
    if any_ntype is None:
        return
    dim = g.nodes[any_ntype].data['features'].shape[1]
    mapper = FeatureMapper(input_dim=dim, hidden_dim=128).to(g.device)
    mapper.load_state_dict(torch.load(mapper_path, map_location=g.device))
    mapper.eval()
    # 仅对 class / attribute 做映射
    for ntype in ['class_node','attribute_node']:
        if ntype in g.ntypes and g.num_nodes(ntype)>0:
            with torch.no_grad():
                f = g.nodes[ntype].data['features']
                g.nodes[ntype].data['features'] = mapper(f)


def apply_meta_rgcn_and_collect(g, cfg):
    """
    加载元学习权重，跑前向，取**节点级嵌入**（四类节点），用于 t-SNE。
    要求 RGCN.forward 支持 return_feats="node"；否则自动跳过。
    """
    try:
        classes = IfcFileGraphsDataset(
            name='tmp',
            root=cfg['data_dir'],
            data_type='val',
            c_dir=cfg['ifc_file_graph_cache'],
            isg=cfg['isg_name'],
            edge_direction=cfg['edge_direction']
        ).classes
        metagraph = IfcFileGraphsDataset(
            name='tmp2',
            root=cfg['data_dir'],
            data_type='val',
            c_dir=cfg['ifc_file_graph_cache'],
            isg=cfg['isg_name'],
            edge_direction=cfg['edge_direction']
        ).re_metagraph
        model = RGCN(
            in_dim=cfg['in_feats'],
            hidden_dim=cfg['hidden_feats'],
            n_classes=len(classes),
            rel_names=metagraph
        ).to(g.device)
        model.load_state_dict(torch.load(META_MODEL_PATH, map_location=g.device))
        model.eval()
        with torch.no_grad():
            out = model(g, return_feats="node")  # 期望返回 (logits, node_feats) 或 node_feats
        # 兼容两种返回
        if isinstance(out, tuple) and len(out) == 2:
            _, node_feats = out
        elif isinstance(out, dict):
            node_feats = out.get("node_feats", None)
        else:
            # 不支持返回节点特征，直接跳过
            print("[Meta] RGCN.forward 未返回节点级特征，跳过 Meta 可视化")
            return None
        # node_feats: dict[ntype]->(N,D)
        feats_by_type = {}
        for ntype in g.ntypes:
            ft = node_feats.get(ntype, None)
            if ft is None or ft.numel() == 0:
                feats_by_type[ntype] = torch.empty(0,0)
            else:
                feats_by_type[ntype] = ft.detach().float().cpu()
        return feats_by_type
    except Exception as e:
        print(f"[Meta] 加载/前向失败：{e}")
        return None


def plot_ifg_under_inits(cfg, index=IFG_INDEX, split='val'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g0 = get_ifg_graph(cfg, index=index, split=split).to(device)
    ensure_gid_present(g0)

    # 读取当前特征的维度（若没有，就默认 128）
    dim = None
    for nt in g0.ntypes:
        if g0.num_nodes(nt)>0 and 'features' in g0.nodes[nt].data:
            dim = g0.nodes[nt].data['features'].shape[1]
            break
    if dim is None:
        dim = cfg['in_feats']

    # 1) ISG 初始化（直接用数据集载入的初始特征）
    g = g0.clone()
    feats0 = collect_feats_by_ntype(g)
    X, y = sample_per_type(feats0)
    if X is not None:
        run_tsne_and_save(X, y, f"IFG#{index} ISG-init (t-SNE)", os.path.join(OUT_DIR, f"IFG{index}_isg_tsne.png"))

    # 2) Random 初始化
    g_rand = g0.clone()
    init_random_features(g_rand, dim=dim)
    feats_r = collect_feats_by_ntype(g_rand)
    X, y = sample_per_type(feats_r)
    if X is not None:
        run_tsne_and_save(X, y, f"IFG#{index} Random-init (t-SNE)", os.path.join(OUT_DIR, f"IFG{index}_random_tsne.png"))

    # # 3) One-hot 确定性（全局表优先；否则图内即时生成）
    # g_one = g0.clone()
    # init_onehot_deterministic(g_one, table_path=DET_GID_TABLE, dim=dim)
    # feats_o = collect_feats_by_ntype(g_one)
    # X, y = sample_per_type(feats_o)
    # if X is not None:
    #     run_tsne_and_save(X, y, f"IFG#{index} OneHot-Det (t-SNE)", os.path.join(OUT_DIR, f"IFG{index}_onehot_det_tsne.png"))
    #
    # # 4) MMD（在 ISG 初始化的基础上对 class/attr 变换）
    # if os.path.exists(MMD_MAPPER_PATH):
    #     g_mmd = g0.clone()
    #     apply_mmd_mapper_on_graph(g_mmd, mapper_path=MMD_MAPPER_PATH)
    #     feats_m = collect_feats_by_ntype(g_mmd)
    #     X, y = sample_per_type(feats_m)
    #     if X is not None:
    #         run_tsne_and_save(X, y, f"IFG#{index} MMD-mapped (t-SNE)", os.path.join(OUT_DIR, f"IFG{index}_mmd_tsne.png"))
    # else:
    #     print(f"[Skip mmd] mapper not found: {MMD_MAPPER_PATH}")
    #
    # # 5) Meta-RGCN：加载元学习权重，输出**节点级嵌入**再可视化
    # if os.path.exists(META_MODEL_PATH):
    #     g_meta = g0.clone()
    #     # 使用 ISG 或 Onehot 都可，这里跟随**原始(加载)特征**更贴近你的训练设置
    #     feats_meta = apply_meta_rgcn_and_collect(g_meta, cfg)
    #     if feats_meta is not None:
    #         X, y = sample_per_type(feats_meta)
    #         if X is not None:
    #             run_tsne_and_save(X, y, f"IFG#{index} Meta-RGCN node embeddings (t-SNE)",
    #                               os.path.join(OUT_DIR, f"IFG{index}_meta_tsne.png"))
    # else:
    #     print(f"[Skip meta] meta weights not found: {META_MODEL_PATH}")


def main():
    cfg = load_cfg()
    plot_ifg_under_inits(cfg, index=IFG_INDEX, split='val')


if __name__ == "__main__":
    main()
