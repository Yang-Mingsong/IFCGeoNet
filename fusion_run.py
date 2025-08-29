# fusion_fewshot_with_static_robust.py
# -*- coding: utf-8 -*-
import os
import random

import pandas as pd
import yaml
import wandb
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import confusion_matrix
# ===== 你的工程内模块 =====
from data.custom_dataset import IfcFileGraphsDataset, MultiViewDataSet
from models.mvcnn import SVCNN, MVCNN
from models.rgcn import RGCN


# =========================
# 工具 & 评测
# =========================
def eval_all_metrics(labels, preds):
    return {
        "overall_acc": metrics.accuracy_score(labels, preds),
        "balanced_acc": metrics.balanced_accuracy_score(labels, preds),
        "precision": metrics.precision_score(labels, preds, average='macro', zero_division=0),
        "f1": metrics.f1_score(labels, preds, average='macro', zero_division=0),
        "recall": metrics.recall_score(labels, preds, average='macro', zero_division=0),
    }


def few_shot_split_indices(dataset, shots_per_class=5, seed=42):
    """
    基于 RGCN 的 dataset 做分层采样（每类 K 个 few-shot 训练，剩余评估）。
    返回: (train_idx, eval_idx)
    """
    random.seed(seed)
    by_cls = {}
    for i in range(len(dataset)):
        _, y = dataset[i]
        y = int(y)
        by_cls.setdefault(y, []).append(i)
    train_idx, eval_idx = [], []
    for c, idxs in by_cls.items():
        assert len(idxs) >= shots_per_class, f"class {c} 样本不足 {shots_per_class}"
        random.shuffle(idxs)
        train_idx += idxs[:shots_per_class]
        eval_idx  += idxs[shots_per_class:]
    train_idx.sort(); eval_idx.sort()
    return train_idx, eval_idx


def freeze_backbone_mvcnn_get_head(mvcnn: nn.Module):
    # 冻结特征提取部分
    for p in mvcnn.net_1.parameters():
        p.requires_grad = False
    # 只训练分类层
    if isinstance(mvcnn.net_2, nn.Sequential):
        for m in mvcnn.net_2[:-1].parameters():
            m.requires_grad = False
        head = mvcnn.net_2[-1]
    else:
        head = mvcnn.net_2  # resnet 情况
    assert isinstance(head, nn.Linear), "MVCNN 的分类头应为 nn.Linear"
    return head


def freeze_backbone_rgcn_get_head(rgcn: nn.Module):
    head = None
    for m in rgcn.modules():
        if isinstance(m, nn.Linear):
            head = m
    assert head is not None, "未找到 RGCN 的分类头线性层"
    for p in rgcn.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    return head


def train_classifier_head_mvcnn(mvcnn, dataloader, device, head_params, epochs=20, lr=5e-3, wd=1e-4):
    mvcnn.to(device).train()
    opt = torch.optim.Adam(head_params, lr=lr, weight_decay=wd)
    for ep in range(1, epochs + 1):
        tloss, n = 0.0, 0
        for (image_batches, labels) in dataloader:
            imgs_np = np.stack(image_batches, axis=1).astype('float32')  # (B,V,C,H,W)
            img_inputs = torch.from_numpy(imgs_np)
            B, V, C, H, W = img_inputs.size()
            img_inputs = img_inputs.view(-1, C, H, W).to(device)
            y = labels.to(device)

            mvcnn.train()
            logits = mvcnn(img_inputs)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

            tloss += loss.item() * y.size(0); n += y.size(0)
        print(f"[MVCNN-Head] Epoch {ep}/{epochs} loss={tloss/max(n,1):.4f}")
    mvcnn.eval()
    return mvcnn


def train_classifier_head_rgcn(rgcn, dataloader, device, head_params, epochs=20, lr=5e-3, wd=1e-4):
    rgcn.to(device).train()
    opt = torch.optim.Adam(head_params, lr=lr, weight_decay=wd)
    for ep in range(1, epochs + 1):
        tloss, n = 0.0, 0
        for (bg, labels) in dataloader:
            bg = bg.to(device); y = labels.to(device)
            logits = rgcn(bg)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item() * y.size(0); n += y.size(0)
        print(f"[RGCN-Head]  Epoch {ep}/{epochs} loss={tloss/max(n,1):.4f}")
    rgcn.eval()
    return rgcn


def compute_softmax_outputs(mvcnn, rgcn, device, mvcnn_loader, rgcn_loader):
    softmax_mvcnn, softmax_rgcn, labels = [], [], []
    total_iter = min(len(mvcnn_loader), len(rgcn_loader))
    for (image_batches, img_label_batches), (graph_batches, gra_label_batches) in tqdm(
        zip(mvcnn_loader, rgcn_loader), total=total_iter
    ):
        with torch.no_grad():
            imgs_np = np.stack(image_batches, axis=1).astype('float32')  # (B,V,C,H,W)
            img_inputs = torch.from_numpy(imgs_np)
            B, V, C, H, W = img_inputs.size()
            img_inputs = img_inputs.view(-1, C, H, W).to(device)

            batched_graph = graph_batches.to(device)

            if torch.equal(img_label_batches, gra_label_batches):
                targets = img_label_batches.to(device)
            else:
                print("[WARN] 图像与图标签不一致，已以图标签为准。")
                targets = gra_label_batches.to(device)

            image_logits = mvcnn(img_inputs)
            graph_logits = rgcn(batched_graph)

            softmax_mvcnn.append(torch.softmax(image_logits, dim=1).cpu().numpy())
            softmax_rgcn.append(torch.softmax(graph_logits, dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())

    return (np.concatenate(softmax_mvcnn, axis=0),
            np.concatenate(softmax_rgcn, axis=0),
            np.concatenate(labels, axis=0))


# ===== per-class 指标（ΔF1 的替代特征会用到） =====
def per_class_f1(labels, preds, num_classes):
    return metrics.f1_score(labels, preds, average=None, labels=list(range(num_classes)))

def per_class_recall(labels, preds, num_classes):
    return metrics.recall_score(labels, preds, average=None, labels=list(range(num_classes)), zero_division=0)

def per_class_balanced_accuracy(labels, preds, num_classes):
    # BA_c = (TPR_c + TNR_c)/2, 其中针对 class c 做一对多二分类
    ba = np.zeros(num_classes, dtype=np.float32)
    y = np.asarray(labels)
    p = np.asarray(preds)
    for c in range(num_classes):
        y_c = (y == c).astype(int)
        p_c = (p == c).astype(int)
        # 混淆
        tp = np.sum((y_c == 1) & (p_c == 1))
        tn = np.sum((y_c == 0) & (p_c == 0))
        fp = np.sum((y_c == 0) & (p_c == 1))
        fn = np.sum((y_c == 1) & (p_c == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ba[c] = 0.5 * (tpr + tnr)
    return ba


# ===== 动态融合（支持传入自定义 delta） =====
def grid_search_group_weights_with_delta(delta, softmax_mvcnn, softmax_rgcn, labels, num_classes,
                                         k_values=9, w_steps=11, random_state=0):
    """
    给定 delta (C,1)，做 KMeans 分组 + 组内权重网格搜索。
    返回:
      best_weights: dict[g] -> np.array (1,2,g)
      best_cluster_labels: dict[g] -> np.array (C,)
    """
    if isinstance(k_values, int):
        k_values = [k_values]

    best_weights = {}
    best_cluster_labels = {}

    for g in k_values:
        km = KMeans(n_clusters=g, random_state=random_state, n_init='auto')
        km.fit(delta)
        cls2cluster = km.labels_.copy()  # (C,)

        w_grid = np.linspace(0.0, 1.0, w_steps)
        best_w_per_cluster = []
        for cluster_id in range(g):
            cls_in_cluster = np.where(cls2cluster == cluster_id)[0]
            mask = np.isin(labels, cls_in_cluster)
            if mask.sum() == 0:
                best_w_per_cluster.append((0.5, 0.5))
                continue
            sm_i = softmax_mvcnn[mask]; sm_g = softmax_rgcn[mask]; y = labels[mask]
            best_f1, best_w = -1.0, (0.5, 0.5)
            for w1 in w_grid:
                w2 = 1.0 - w1
                fused = w1 * sm_i + w2 * sm_g
                pred  = np.argmax(fused, axis=1)
                f1 = metrics.f1_score(y, pred, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1, best_w = f1, (float(w1), float(w2))
            best_w_per_cluster.append(best_w)

        W = np.array(best_w_per_cluster, dtype=np.float32).T  # (2,g)
        W = W[None, ...]  # (1,2,g)
        best_weights[g] = W
        best_cluster_labels[g] = cls2cluster

    return best_weights, best_cluster_labels


def grid_search_group_weights(softmax_mvcnn, softmax_rgcn, labels, num_classes, k_values=9, w_steps=11, random_state=0):
    # 标准 ΔF1
    preds_img = np.argmax(softmax_mvcnn, axis=1)
    preds_gra = np.argmax(softmax_rgcn, axis=1)
    f1_img = per_class_f1(labels, preds_img, num_classes)  # (C,)
    f1_gra = per_class_f1(labels, preds_gra, num_classes)  # (C,)
    delta = (f1_img - f1_gra).reshape(-1, 1)               # (C,1)
    return grid_search_group_weights_with_delta(delta, softmax_mvcnn, softmax_rgcn, labels,
                                                num_classes, k_values, w_steps, random_state)


def test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_weights, best_cluster_labels, g, tag="dynamic"):
    best_w = best_weights[g][0]              # (2, g)
    cluster_labels = best_cluster_labels[g]  # (C,)

    final_preds = np.zeros_like(labels)
    for cluster in range(g):
        cls_in_cluster = np.where(cluster_labels == cluster)[0]
        class_indices = np.isin(labels, cls_in_cluster)
        if class_indices.sum() == 0:
            continue
        w1, w2 = best_w[:, cluster]
        fused_probs = w1 * softmax_mvcnn[class_indices] + w2 * softmax_rgcn[class_indices]
        final_preds[class_indices] = np.argmax(fused_probs, axis=1)

    # ---- metrics ----
    m = eval_all_metrics(labels, final_preds)
    print(f"==[{tag} Fusion]==")
    print("Overall:{:.6f}  Balanced:{:.6f}  Precision:{:.6f}  F1:{:.6f}  Recall:{:.6f}"
          .format(m["overall_acc"], m["balanced_acc"], m["precision"], m["f1"], m["recall"]))
    wandb.log({f"{tag}/"+k: v for k, v in m.items()})

    return m


# ===== 静态融合（全局网格搜索） =====
def static_fusion_grid_search(softmax_mvcnn, softmax_rgcn, labels, w_steps=11):
    w_grid = np.linspace(0.0, 1.0, w_steps)
    best = {"f1": -1.0}
    best_w1, best_w2 = 0.5, 0.5

    for w1 in w_grid:
        w2 = 1.0 - w1
        fused = w1 * softmax_mvcnn + w2 * softmax_rgcn
        preds = np.argmax(fused, axis=1)
        m = eval_all_metrics(labels, preds)
        if m["f1"] > best["f1"]:
            best, best_w1, best_w2 = m, float(w1), float(w2)

    return best_w1, best_w2, best


def static_fusion_eval(softmax_mvcnn, softmax_rgcn, labels, w1=0.5):
    fused = w1 * softmax_mvcnn + (1.0 - w1) * softmax_rgcn
    preds = np.argmax(fused, axis=1)
    return eval_all_metrics(labels, preds)


# ===== 鲁棒性小实验：替代特征 & 加噪声 =====
def build_delta_variant(labels, softmax_mvcnn, softmax_rgcn, num_classes, mode="f1", noise_sigma=None, seed=0):
    """mode in {'f1','recall','ba'}，可选对 Δ 向量加高斯噪声。"""
    preds_img = np.argmax(softmax_mvcnn, axis=1)
    preds_gra = np.argmax(softmax_rgcn, axis=1)

    if mode == "f1":
        m_img = per_class_f1(labels, preds_img, num_classes)
        m_gra = per_class_f1(labels, preds_gra, num_classes)
    elif mode == "recall":
        m_img = per_class_recall(labels, preds_img, num_classes)
        m_gra = per_class_recall(labels, preds_gra, num_classes)
    elif mode == "ba":
        m_img = per_class_balanced_accuracy(labels, preds_img, num_classes)
        m_gra = per_class_balanced_accuracy(labels, preds_gra, num_classes)
    else:
        raise ValueError("mode must be in {'f1','recall','ba'}")

    delta = (m_img - m_gra).astype(np.float32).reshape(-1, 1)
    if noise_sigma is not None and noise_sigma > 0:
        rng = np.random.RandomState(seed)
        delta = delta + rng.normal(0.0, noise_sigma, size=delta.shape).astype(np.float32)
    return delta


def robustness_feature_substitution(softmax_mvcnn, softmax_rgcn, labels, num_classes, g_eval, w_steps=11):
    """替代 ΔF1 的特征进行聚类：Recall-diff、BA-diff。返回两组 F1"""
    results = {}
    for mode in ["recall", "ba"]:
        delta_alt = build_delta_variant(labels, softmax_mvcnn, softmax_rgcn, num_classes, mode=mode)
        best_w, best_cls = grid_search_group_weights_with_delta(delta_alt, softmax_mvcnn, softmax_rgcn,
                                                                labels, num_classes, k_values=g_eval, w_steps=w_steps)
        m = test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_w, best_cls, g=g_eval, tag=f"dynamic_{mode}")
        results[mode] = m["f1"]
    return results  # dict: {'recall': f1, 'ba': f1}


def robustness_noise_perturbation(softmax_mvcnn, softmax_rgcn, labels, num_classes, g_eval, base_delta=None,
                                  sigma=0.005, trials=5, w_steps=11, seed=0):
    """在 ΔF1 上加噪声，多次试验，返回 F1 的 mean/std。"""
    if base_delta is None:
        base_delta = build_delta_variant(labels, softmax_mvcnn, softmax_rgcn, num_classes, mode="f1")
    f1_list = []
    for t in range(trials):
        delta_noisy = base_delta + np.random.RandomState(seed + t).normal(0.0, sigma, size=base_delta.shape).astype(np.float32)
        best_w, best_cls = grid_search_group_weights_with_delta(delta_noisy, softmax_mvcnn, softmax_rgcn,
                                                                labels, num_classes, k_values=g_eval, w_steps=w_steps,
                                                                random_state=seed + t)
        m = test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_w, best_cls, g=g_eval, tag=f"dynamic_noisy_t{t}")
        f1_list.append(m["f1"])
    return float(np.mean(f1_list)), float(np.std(f1_list))



# =========================
# 主流程
# =========================
def fusion_main():
    # 读取配置
    with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    gwf = config.get('GroupedWeightFusion', {})

    # wandb
    wandb.init(project=gwf.get('project_name', 'GroupedWeightFusion'),
               name=gwf.get('run_name', 'fewshot_head_adapt'),
               config=gwf, mode=gwf.get('wandb_mode', 'offline'))

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gwf.get('gpu_device', '0'))
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    transform = transforms.Compose([
        transforms.Resize(gwf.get('resize', 224)),
        transforms.ToTensor()
    ])

    print('Creating data...')

    if gwf.get('ENABLE_HEAD_ADAPT'):
        # ====== 原来的 few-shot 适配路径（保留，将来可开） ======
        rgcn_test_dataset = IfcFileGraphsDataset(
            name='test_dataset',
            root=gwf['graph_data_dir'],
            data_type='test',
            c_dir=gwf['ifc_file_graph_cache'],
            isg=gwf['isg_name'],
            edge_direction=gwf['edge_direction']
        )
        mvcnn_test_dataset = MultiViewDataSet(
            gwf['png_data_dir'],
            data_type='test',
            view_name=gwf['view_name'],
            data_transform=transform
        )
        shots = int(gwf.get('shots_per_class', 5))
        seed = int(gwf.get('seed', 42))
        train_idx, eval_idx = few_shot_split_indices(rgcn_test_dataset, shots_per_class=shots, seed=seed)

        rgcn_train = Subset(rgcn_test_dataset, train_idx)
        rgcn_eval = Subset(rgcn_test_dataset, eval_idx)
        mvcnn_train = Subset(mvcnn_test_dataset, train_idx)
        mvcnn_eval = Subset(mvcnn_test_dataset, eval_idx)

        rgcn_train_loader = GraphDataLoader(rgcn_train, batch_size=int(gwf.get('train_batch_size', 16)),
                                            drop_last=False, shuffle=True)
        rgcn_eval_loader = GraphDataLoader(rgcn_eval, batch_size=int(gwf.get('test_batch_size', 32)),
                                           drop_last=False, shuffle=False)
        mvcnn_train_loader = DataLoader(mvcnn_train, num_workers=int(gwf.get('num_workers', 0)),
                                        batch_size=int(gwf.get('train_batch_size', 16)), shuffle=True, drop_last=False)
        mvcnn_eval_loader = DataLoader(mvcnn_eval, num_workers=int(gwf.get('num_workers', 0)),
                                       batch_size=int(gwf.get('test_batch_size', 32)), shuffle=False, drop_last=False)
    else:
        # ====== 旧数据集“零适配”静态融合路径 ======
        # 用你旧数据集上的验证/测试划分：如果你有 val/test，就在 val 上搜静态权重，test 上汇报；
        # 如果只有 test，就在 test 上直接做（需要在论文里说明）
        rgcn_eval_dataset = IfcFileGraphsDataset(
            name='eval_dataset',
            root=gwf['graph_data_dir'],
            data_type=gwf.get('eval_split', 'test'),  # 可在 config 里设 'val' 或 'test'
            c_dir=gwf['ifc_file_graph_cache'],
            isg=gwf['isg_name'],
            edge_direction=gwf['edge_direction']
        )
        mvcnn_eval_dataset = MultiViewDataSet(
            gwf['png_data_dir'],
            data_type=gwf.get('eval_split', 'test'),
            view_name=gwf['view_name'],
            data_transform=transform
        )

        rgcn_eval_loader = GraphDataLoader(rgcn_eval_dataset, batch_size=int(gwf.get('test_batch_size', 32)),
                                           drop_last=False, shuffle=False)
        mvcnn_eval_loader = DataLoader(mvcnn_eval_dataset, num_workers=int(gwf.get('num_workers', 0)),
                                       batch_size=int(gwf.get('test_batch_size', 32)), shuffle=False, drop_last=False)

    # ===== 模型 =====
    metagraph = (rgcn_eval_dataset if not gwf.get('ENABLE_HEAD_ADAPT') else rgcn_test_dataset).re_metagraph
    classes = (rgcn_eval_dataset if not gwf.get('ENABLE_HEAD_ADAPT') else rgcn_test_dataset).classes
    num_classes = len(classes)

    print("Loading models...")
    svcnn = SVCNN(num_classes=num_classes, pretraining=True, cnn_name=gwf['svcnn'])
    mvcnn = MVCNN(svcnn_model=svcnn, num_classes=num_classes, num_views=int(gwf['num_views'])).to(device)
    del svcnn
    try:
        mvcnn.load_state_dict(torch.load(gwf['MVCNN_Pretrained_model_path'], weights_only=True))
    except TypeError:
        mvcnn.load_state_dict(torch.load(gwf['MVCNN_Pretrained_model_path']))
    mvcnn.eval()

    rgcn = RGCN(int(gwf['in_feats']), int(gwf['hidden_feats']), num_classes, metagraph).to(device)
    try:
        rgcn.load_state_dict(torch.load(gwf['RGCN_Pretrained_model_path'], weights_only=True))
    except TypeError:
        rgcn.load_state_dict(torch.load(gwf['RGCN_Pretrained_model_path']))
    rgcn.eval()

    # ===== 只有当需要 few-shot 适配时，才训练分类头 =====
    if gwf.get('ENABLE_HEAD_ADAPT'):
        head_mv = freeze_backbone_mvcnn_get_head(mvcnn)
        head_rg = freeze_backbone_rgcn_get_head(rgcn)

        print(f"Training classifier heads with {shots}-shot per class ...")
        mvcnn = train_classifier_head_mvcnn(
            mvcnn, mvcnn_train_loader, device, head_mv.parameters(),
            epochs=int(gwf.get('head_epochs', 20)),
            lr=float(gwf.get('head_lr', 5e-3)),
            wd=float(gwf.get('head_wd', 1e-4))
        )
        rgcn = train_classifier_head_rgcn(
            rgcn, rgcn_train_loader, device, head_rg.parameters(),
            epochs=int(gwf.get('head_epochs', 20)),
            lr=float(gwf.get('head_lr', 5e-3)),
            wd=float(gwf.get('head_wd', 1e-4))
        )

    # ===== 评估：计算 softmax =====
    print("Computing Softmax outputs on eval split...")
    softmax_mvcnn, softmax_rgcn, labels = compute_softmax_outputs(
        mvcnn, rgcn, device, mvcnn_eval_loader, rgcn_eval_loader
    )

    # ===== 单模型基线 =====
    pred_img = np.argmax(softmax_mvcnn, axis=1)
    pred_gra = np.argmax(softmax_rgcn, axis=1)
    m_img = eval_all_metrics(labels, pred_img)
    m_gra = eval_all_metrics(labels, pred_gra)
    print("[Single] MVCNN:", m_img)
    print("[Single] RGCN  :", m_gra)
    wandb.log({f"single/mv_{k}": v for k, v in m_img.items()})
    wandb.log({f"single/rg_{k}": v for k, v in m_gra.items()})

    # ===== 静态融合：全局网格搜索 =====
    w_steps = int(gwf.get('static_w_steps', 11))  # 0.0~1.0，步数=21 即间隔0.05
    best_w1, best_w2, m_static = static_fusion_grid_search(softmax_mvcnn, softmax_rgcn, labels, w_steps=w_steps)
    print(f"[StaticFusion] best w1={best_w1:.3f}, w2={best_w2:.3f} | {m_static}")
    wandb.log({"static/best_w1": best_w1, "static/best_w2": best_w2})
    wandb.log({f"static/{k}": v for k, v in m_static.items()})
    # 可选：固定 0.5/0.5 也报一下
    m_static_0505 = static_fusion_eval(softmax_mvcnn, softmax_rgcn, labels, w1=0.5)
    print(f"[StaticFusion-0.5] {m_static_0505}")
    wandb.log({f"static0505/{k}": v for k, v in m_static_0505.items()})

    # ===== 动态融合（标准 ΔF1） =====
    os.makedirs(gwf.get('save_path', './outputs'), exist_ok=True)
    k_values = gwf.get('k_values', 9)
    if not isinstance(k_values, (list, tuple)):
        try:
            k_values = int(k_values)
        except:
            k_values = 9
    g_eval = k_values if isinstance(k_values, int) else (k_values[0] if len(k_values) > 0 else 9)

    print("Running grid search for best fusion weights (dynamic, ΔF1)...")
    best_weights, best_cluster_labels = grid_search_group_weights(
        softmax_mvcnn, softmax_rgcn, labels, num_classes, k_values=g_eval, w_steps=int(gwf.get('dyn_w_steps', 11))
    )
    np.save(os.path.join(gwf['save_path'], 'best_weights.npy'), best_weights)
    np.save(os.path.join(gwf['save_path'], 'best_cluster_labels.npy'), best_cluster_labels)

    m_dynamic = test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_weights, best_cluster_labels, g=g_eval, tag="dynamic")


    # ===== 鲁棒性：替代特征 =====
    print("Robustness: feature substitution (Recall-diff / BA-diff)")
    alt_res = robustness_feature_substitution(softmax_mvcnn, softmax_rgcn, labels, num_classes, g_eval,
                                              w_steps=int(gwf.get('dyn_w_steps', 11)))
    for mode, f1 in alt_res.items():
        diff = f1 - m_dynamic["f1"]
        print(f"[FeatureSub] {mode}-diff F1={f1:.6f} (Δ vs ΔF1: {diff:+.4%})")
        wandb.log({f"robust_feature/{mode}_f1": f1, f"robust_feature/{mode}_delta": diff})

    # ===== 鲁棒性：ΔF1 加噪声 =====
    print("Robustness: ΔF1 + Gaussian noise (σ≈0.5%)")
    base_delta = build_delta_variant(labels, softmax_mvcnn, softmax_rgcn, num_classes, mode="f1")
    sigma = float(gwf.get('noise_sigma', 0.005))  # ~0.5%
    trials = int(gwf.get('noise_trials', 5))
    mean_f1, std_f1 = robustness_noise_perturbation(softmax_mvcnn, softmax_rgcn, labels, num_classes,
                                                    g_eval, base_delta=base_delta, sigma=sigma, trials=trials,
                                                    w_steps=int(gwf.get('dyn_w_steps', 11)), seed=int(gwf.get('seed', 42)))
    print(f"[Noise] mean F1={mean_f1:.6f}, std={std_f1:.6f} (Δ vs ΔF1: {mean_f1 - m_dynamic['f1']:+.4%})")
    wandb.log({"robust_noise/mean_f1": mean_f1, "robust_noise/std_f1": std_f1,
               "robust_noise/delta_vs_base": mean_f1 - m_dynamic["f1"], "robust_noise/sigma": sigma})

    wandb.finish()


if __name__ == '__main__':
    fusion_main()
