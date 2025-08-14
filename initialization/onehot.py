import os
import torch
import yaml
from collections import defaultdict
from tqdm import tqdm
from data.custom_dataset import IfcFileGraphsDataset

# ---------- Config ----------
CFG_PATH = r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml'
OUTPUT_DIR = r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization'
DETERMINISTIC_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'val_deterministic_onehot_features.pt')
# --------------------------------

def load_config():
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['RGCN']

def ensure_gid_present(g):
    for ntype in g.ntypes:
        if '_gid' not in g.nodes[ntype].data:
            n_nodes = g.num_nodes(ntype)
            g.nodes[ntype].data['_gid'] = torch.arange(n_nodes, dtype=torch.long)

def build_global_gid_sets(dataset):
    gid_sets = defaultdict(set)
    for g, _ in tqdm(dataset, desc="Scanning graphs for global gid sets"):
        ensure_gid_present(g)
        for ntype in g.ntypes:
            gids = g.nodes[ntype].data['_gid'].tolist()
            gid_sets[ntype].update(gids)
    # sort for determinism
    gid_list = {ntype: sorted(list(gids)) for ntype, gids in gid_sets.items()}
    return gid_list  # {ntype: [gid1, gid2, ...]}

def make_deterministic_embedding_table(gid_list, target_dim):
    """
    Create deterministic embeddings for each gid using cos/sin positional-style encoding.
    Returns {ntype: {gid: tensor(target_dim)}}
    """
    embedding_tables = {}
    for ntype, gids in gid_list.items():
        if len(gids) == 0:
            embedding_tables[ntype] = {}
            continue
        # scale factor M to control frequency; using number of unique gids for stability
        M = float(len(gids)) + 1.0
        half = target_dim // 2
        table = {}
        for gid in gids:
            # vectorized: build angles for k=0..half-1
            ks = torch.arange(half, dtype=torch.float32)
            angle = (torch.pi * (gid + 1) * (ks + 1)) / M  # shape (half,)
            even = torch.cos(angle)  # for positions 0,2,4,...
            odd = torch.sin(angle)   # for positions 1,3,5,...
            vec = torch.empty(target_dim, dtype=torch.float32)
            vec[0:2*half:2] = even
            vec[1:2*half:2] = odd
            if target_dim % 2 == 1:
                # last element if odd dimension, reuse cos with next k
                k_extra = half
                angle_extra = (torch.pi * (gid + 1) * (k_extra + 1)) / M
                vec[-1] = torch.cos(angle_extra)
            table[gid] = vec
        embedding_tables[ntype] = table
    return embedding_tables

@torch.no_grad()
def extract_graph_features_with_onehot_deterministic(cfg, mode='val', save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = IfcFileGraphsDataset(
        name=f'{mode}_dataset',
        root=cfg['data_dir'],
        data_type=mode,
        c_dir=cfg['ifc_file_graph_cache'],
        isg=cfg['isg_name'],
        edge_direction=cfg['edge_direction']
    )

    # step1: build global gid list and embedding tables
    gid_list = build_global_gid_sets(dataset)
    embedding_tables = make_deterministic_embedding_table(gid_list, target_dim=cfg['in_feats'])

    features, labels = [], []
    for g, label in tqdm(dataset, desc="Applying deterministic one-hot init"):
        ensure_gid_present(g)
        # assign features per node type
        for ntype in g.ntypes:
            gids = g.nodes[ntype].data['_gid']
            emb_list = []
            table = embedding_tables.get(ntype, {})
            for gid in gids.tolist():
                emb = table.get(gid)
                if emb is None:
                    # fallback zero vector
                    emb = torch.zeros(cfg['in_feats'], dtype=torch.float32)
                emb_list.append(emb)
            feat = torch.stack(emb_list)  # (N_nodes, in_feats)
            g.nodes[ntype].data['features'] = feat  # keep on CPU

        # graph-level embedding: concat mean over each node type present
        graph_feat = torch.cat([
            g.nodes[ntype].data['features'].mean(dim=0)
            for ntype in g.ntypes if g.num_nodes(ntype) > 0
        ])
        features.append(graph_feat)
        labels.append(label)

    features = torch.stack(features)
    labels = torch.tensor(labels)
    torch.save({'features': features, 'labels': labels}, save_path)
    print(f"Saved deterministic one-hot graph features to {save_path}")

def main():
    cfg = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extract_graph_features_with_onehot_deterministic(cfg, mode='val', save_path=DETERMINISTIC_FEATURES_PATH)

if __name__ == '__main__':
    main()
