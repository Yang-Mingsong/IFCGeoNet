import torch
import torch.nn.functional as F
import random
import copy
import yaml
import os
from torch.backends import cudnn
from data.custom_dataset import IfcFileGraphsDataset
from models.rgcn import RGCN


# Load config
with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
ifg_Config = config.get('RGCN')

# ===== Graph Task Sampler =====
class GraphTaskSampler:
    def __init__(self, dataset, n_way, n_support, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.label_to_indices = self._build_label_index()

    def _build_label_index(self):
        label_dict = {}
        for idx, label in enumerate(self.dataset.labels):
            label_dict.setdefault(int(label), []).append(idx)
        return label_dict

    def sample_task(self):
        selected_classes = random.sample(list(self.label_to_indices.keys()), self.n_way)
        support, query = [], []
        for cls in selected_classes:
            indices = random.sample(self.label_to_indices[cls], self.n_support + self.n_query)
            support.extend(indices[:self.n_support])
            query.extend(indices[self.n_support:])
        return support, query

# ===== Reptile Meta-Training =====
def reptile_meta_train(model, dataset, device, meta_epochs=100, inner_steps=3, lr_inner=0.0001, meta_step_size=0.05):
    sampler = GraphTaskSampler(dataset, n_way=20, n_support=3, n_query=2)
    for epoch in range(meta_epochs):
        model_inner = copy.deepcopy(model).to(device)
        optimizer = torch.optim.SGD(model_inner.parameters(), lr=lr_inner)
        support_ids, _ = sampler.sample_task()
        model_inner.train()
        for step in range(inner_steps):
            loss = 0
            for idx in support_ids:
                g, label = dataset[idx]
                g = g.to(device)
                label = torch.tensor([label], dtype=torch.long).to(device)
                pred = model_inner(g)

                # Debug: 检查是否有NaN
                if torch.isnan(pred).any():
                    print(f"NaN detected in prediction at epoch {epoch}, step {step}, sample {idx}")
                    continue

                # Debug: 如果预测结果是一维的，加一个 batch 维度
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)

                # Debug: 打印当前预测维度与标签
                print(f"Epoch {epoch}, Step {step}, Sample {idx}, Pred shape: {pred.shape}, Label: {label}")

                try:
                    loss += F.cross_entropy(pred, label)
                except Exception as e:
                    print(f"Loss computation failed at epoch {epoch}, step {step}, sample {idx}: {e}")
                    continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reptile meta-update
        for param, meta_param in zip(model.parameters(), model_inner.parameters()):
            param.data = param.data + meta_step_size * (meta_param.data - param.data)

        if epoch % 10 == 0:
            print(f"[Meta Epoch {epoch}] Done")

# ===== Feature Extraction =====
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
    train_dataset = IfcFileGraphsDataset(
        name='train_dataset',
        root=ifg_Config['data_dir'],
        data_type='train',
        c_dir=ifg_Config['ifc_file_graph_cache'],
        isg=ifg_Config['isg_name'],
        edge_direction=ifg_Config['edge_direction']
    )

    classes = train_dataset.classes
    metagraph = train_dataset.re_metagraph
    print('Creating model...', ifg_Config['edge_direction'])
    model = RGCN(
        in_dim=ifg_Config['in_feats'],
        hidden_dim=ifg_Config['hidden_feats'],
        n_classes=len(classes),
        rel_names=metagraph
    ).to(device)

    print("Start meta-training...")
    reptile_meta_train(model, train_dataset, device)
    torch.save(model.state_dict(), r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\meta_init.pth")

    test_dataset = IfcFileGraphsDataset(
        name='test_dataset',
        root=ifg_Config['data_dir'],
        data_type='val',
        c_dir=ifg_Config['ifc_file_graph_cache'],
        isg=ifg_Config['isg_name'],
        edge_direction=ifg_Config['edge_direction']
    )
    extract_graph_features(model, test_dataset, device, save_path=r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\val_meta_features.pt')
