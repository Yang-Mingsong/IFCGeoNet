import numpy as np
import os
import wandb
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.custom_dataset import IfcSchemaGraphDataSet, IfcFileGraphsDataset, MultiViewDataSet
from torch.backends import cudnn
from torch import nn
from torch.utils.data import DataLoader
from models.mvcnn import SVCNN, MVCNN
from models.rgcn import RGCN
from scripts.test import RGCNTester, MVCNNTester
from scripts.train import HGATTrainer, RGCNTrainer, MVCNNTrainer, compute_softmax_outputs
from utils.early_stopping import EarlyStopping
from models.hgat import HeteroGAT
from dgl.dataloading import GraphDataLoader
import torchvision.transforms as transforms
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report

# configs
with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
gwf_Config = config.get('GroupedWeightFusion')


def fusion_main():

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = gwf_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    # data
    print('Creating data...')
    rgcn_test_dataset = IfcFileGraphsDataset(name='test_dataset', root=gwf_Config['graph_data_dir'], data_type='test', c_dir=gwf_Config['ifc_file_graph_cache'],
                                             isg=gwf_Config['isg_name'], edge_direction=gwf_Config['edge_direction'])
    rgcn_test_dataloader = GraphDataLoader(rgcn_test_dataset, batch_size=gwf_Config['test_batch_size'], drop_last=False, shuffle=False)
    mvcnn_test_dataset = MultiViewDataSet(gwf_Config['png_data_dir'], data_type='test', view_name=gwf_Config['view_name'], data_transform=transform)
    mvcnn_test_dataloader = DataLoader(mvcnn_test_dataset, num_workers=0, batch_size=gwf_Config['test_batch_size'], shuffle=False, drop_last=False)

    metagraph = rgcn_test_dataset.re_metagraph
    classes = rgcn_test_dataset.classes

    # model
    print("Loading models...")
    # 加载预训练MVCNN权重
    svcnn = SVCNN(num_classes=len(classes), pretraining=True, cnn_name=gwf_Config['svcnn'])
    mvcnn = MVCNN(svcnn_model=svcnn, num_classes=len(classes), num_views=gwf_Config['num_views']).to(device)
    del svcnn

    pretrained_weights = torch.load(gwf_Config['MVCNN_Pretrained_model_path'], weights_only=True)  # 预训练权重文件
    mvcnn.load_state_dict(pretrained_weights)  # strict=False 允许加载部分权重
    for param in mvcnn.parameters():
        param.requires_grad = False

    # 加载预训练RGCN权重
    rgcn = RGCN(gwf_Config['in_feats'], gwf_Config['hidden_feats'], len(classes), metagraph).to(device)
    pretrained_weights = torch.load(gwf_Config['RGCN_Pretrained_model_path'], weights_only=True)  # 预训练权重文件
    rgcn.load_state_dict(pretrained_weights)
    for param in rgcn.parameters():
        param.requires_grad = False
    mvcnn.eval()
    rgcn.eval()

    # Computing
    print("Computing Softmax outputs...")
    softmax_mvcnn, softmax_rgcn, labels = compute_softmax_outputs(mvcnn, rgcn, device, mvcnn_test_dataloader, rgcn_test_dataloader)

    pred_mvcnn = np.argmax(softmax_mvcnn, axis=1)
    pred_rgcn = np.argmax(softmax_rgcn, axis=1)

    mvcnn_cm = confusion_matrix(labels, pred_mvcnn)
    mvcnn_precision_per_class = precision_score(labels, pred_mvcnn, average=None, zero_division=0)
    mvcnn_recall_per_class = recall_score(labels, pred_mvcnn, average=None, zero_division=0)
    mvcnn_f1_score_per_class = f1_score(labels, pred_mvcnn, average=None, zero_division=0)
    df_mvcnn_cm = pd.DataFrame(mvcnn_cm, index=classes, columns=classes)
    df_mvcnn_cm.to_csv(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\confusion_matrix\mvcnn_cm.csv", index=True)
    df_mvcnn = pd.DataFrame({
        "Precision": mvcnn_precision_per_class,
        "Recall": mvcnn_recall_per_class,
        "F1 Score": mvcnn_f1_score_per_class
    })
    df_mvcnn.to_csv(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\confusion_matrix\mvcnn.csv", index=True)

    rgcn_cm = confusion_matrix(labels, pred_rgcn)
    rgcn_precision_per_class = precision_score(labels, pred_rgcn, average=None, zero_division=0)
    rgcn_recall_per_class = recall_score(labels, pred_rgcn, average=None, zero_division=0)
    rgcn_f1_score_per_class = f1_score(labels, pred_rgcn, average=None, zero_division=0)
    df_rgcn_cm = pd.DataFrame(rgcn_cm, index=classes, columns=classes)
    df_rgcn_cm.to_csv(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\confusion_matrix\rgcn_cm.csv", index=True)
    df_rgcn = pd.DataFrame({
        "Precision": rgcn_precision_per_class,
        "Recall": rgcn_recall_per_class,
        "F1 Score": rgcn_f1_score_per_class
    })
    df_rgcn.to_csv(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\confusion_matrix\rgcn.csv", index=True)

    best_weights = np.load(os.path.join(gwf_Config['save_path'], 'best_weights.npy'), allow_pickle=True).item()
    best_cluster_labels = np.load(os.path.join(gwf_Config['save_path'], 'best_cluster_labels.npy'), allow_pickle=True).item()
    pred_ifcgeonet = test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_weights, best_cluster_labels, g=gwf_Config['k_values'])
    ifcgeonet_cm = confusion_matrix(labels, pred_ifcgeonet)
    ifcgeonet_precision_per_class = precision_score(labels, pred_ifcgeonet, average=None, zero_division=0)
    ifcgeonet_recall_per_class = recall_score(labels, pred_ifcgeonet, average=None, zero_division=0)
    ifcgeonet_f1_score_per_class = f1_score(labels, pred_ifcgeonet, average=None, zero_division=0)
    df_ifcgeonet_cm = pd.DataFrame(ifcgeonet_cm, index=classes, columns=classes)
    df_ifcgeonet_cm.to_csv(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\confusion_matrix\ifcgeonet_cm.csv", index=True)
    df_ifcgeonet = pd.DataFrame({
        "Precision": ifcgeonet_precision_per_class,
        "Recall": ifcgeonet_recall_per_class,
        "F1 Score": ifcgeonet_f1_score_per_class
    })
    df_ifcgeonet.to_csv(r"Q:\pychem_project\XUT-GeoIFCNet-Master\results\confusion_matrix\ifcgeonet.csv", index=True)


def test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_weights, best_cluster_labels, g):
    best_w = best_weights[g][0]  # 获取对应的权重
    cluster_labels = best_cluster_labels[g]  # 直接使用 grid_search_group_weights 计算的类别分组
    # 计算加权 softmax 结果
    final_preds = np.zeros_like(labels)
    for cluster in range(g):
        # class_indices = np.where(cluster_labels == cluster)[0]
        class_indices = np.isin(labels, np.where(cluster_labels == cluster)[0])
        if class_indices.sum() == 0:
            continue
        w1, w2 = best_w[:, cluster]
        fused_probs = w1 * softmax_mvcnn[class_indices] + w2 * softmax_rgcn[class_indices]
        final_preds[class_indices] = np.argmax(fused_probs, axis=1)
    return final_preds


if __name__ == '__main__':
    fusion_main()
