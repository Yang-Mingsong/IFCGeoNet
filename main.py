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
from scripts.test import RGCNTester, MVCNNTester, test_fusion_model
from scripts.train import HGATTrainer, RGCNTrainer, MVCNNTrainer, compute_softmax_outputs
from utils.early_stopping import EarlyStopping
from models.hgat import HeteroGAT
from dgl.dataloading import GraphDataLoader
import torchvision.transforms as transforms
import numpy as np

# configs
with open(r'Q:\pychem_project\XUT-GeoIFCNet-Master\configs\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
isg_Config = config.get('HGAT')
ifg_Config = config.get('RGCN')
img_Config = config.get('MVCNN')
gwf_Config = config.get('GroupedWeightFusion')


def ifc_schema_graph_main():
    # Initialize wandb
    wandb.init(project=isg_Config['project_name'], config=isg_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = isg_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    early_stopping = EarlyStopping(isg_Config['save_path'] + '\\early_stopping', patience=8)

    # data
    print('Creating graph...')
    ifc_schema_data = IfcSchemaGraphDataSet("IFCSCHEMAGRAPH", isg_Config['edge_csv'], isg_Config['node_csv'],
                                            isg_Config['cache_path'], isg_Config['in_feat'], isg_Config['edge_direction'], isg_Config["feat_seed"])  # 默认是正向的
    ifc_schema_graph = ifc_schema_data.g.to(device)
    print(isg_Config['edge_direction'], ifc_schema_graph.canonical_etypes)

    # model
    print('Creating model...')
    isg_hgat = HeteroGAT(in_feats=isg_Config['in_feat'], hidden_feats=isg_Config['hidden_feats'], out_feats=isg_Config['out_feats'],
                         num_heads=isg_Config['num_heads'], etypes=ifc_schema_graph.canonical_etypes).to(device)

    # Optimizer
    optimizer = Adam(isg_hgat.parameters(), lr=isg_Config['learning_rate'], weight_decay=isg_Config['weight_decay'], betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, isg_Config['num_epochs'], eta_min=isg_Config['eta_min'])

    # train
    trainer = HGATTrainer(isg_hgat, ifc_schema_graph, optimizer, device, isg_Config['edge_direction'], isg_Config["seed"])

    print('start training...')
    for epoch in range(1, isg_Config['num_epochs'] + 1):
        trainer.train(epoch, early_stopping, isg_Config["in_feat"], isg_Config['temperature'], isg_Config['max_norm'], isg_Config["save_path"])
        scheduler.step()
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练;

    # 获取最终的节点嵌入表示
    isg_hgat.eval()
    with torch.no_grad():
        result = isg_hgat(ifc_schema_graph, trainer.node_features)

    node_embeddings = {}
    # 遍历 result 中的每个节点类型
    for node_type in result.keys():
        node_embeddings[node_type] = {}
        # 遍历 result[node_type] 中的每个节点嵌入向量
        for node_id, vector in enumerate(result[node_type]):
            # 获取节点名称
            node_name = ifc_schema_data.node_types[node_type][node_id]
            # 将节点名称和向量添加到字典中
            vector = vector.detach().to('cpu')
            node_embeddings[node_type][node_name] = vector

    save_path = os.path.join(isg_Config['save_path'], isg_Config['edge_direction'] + '_' + isg_Config["project_name"] + '.bin')
    torch.save(node_embeddings, str(save_path))

    result = torch.load(str(save_path), weights_only=False)
    print(result)


def ifc_file_graph_main():
    # Initialize wandb
    wandb.init(project=ifg_Config['project_name'], name=ifg_Config['run_name'], config=ifg_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ifg_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    early_stopping = EarlyStopping(save_path=ifg_Config['save_path'] + '\\early_stopping', patience=ifg_Config['early_stopping'])

    # data

    print('Creating graph...')
    train_dataset = IfcFileGraphsDataset(name='train_dataset', root=ifg_Config['data_dir'], data_type='train', c_dir=ifg_Config['ifc_file_graph_cache'],
                                         isg=ifg_Config['isg_name'], edge_direction=ifg_Config['edge_direction'])
    train_loader = GraphDataLoader(train_dataset, batch_size=ifg_Config['batch_size'], drop_last=False, shuffle=True)
    # 修改data_type
    test_dataset = IfcFileGraphsDataset(name='test_dataset', root=ifg_Config['data_dir'], data_type='test', c_dir=ifg_Config['ifc_file_graph_cache'],
                                        isg=ifg_Config['isg_name'], edge_direction=ifg_Config['edge_direction'])
    # 修改batch_size
    test_loader = GraphDataLoader(test_dataset, batch_size=ifg_Config['test_batch_size'], drop_last=False, shuffle=False)

    # model
    classes = train_dataset.classes
    if ifg_Config['edge_direction'] == 'REVERSE':
        metagraph = train_dataset.re_metagraph
    else:
        metagraph = train_dataset.metagraph
    print('Creating model...', ifg_Config['edge_direction'])
    model = RGCN(ifg_Config['in_feats'], ifg_Config['hidden_feats'], len(classes), metagraph).to(device)

    # Loss and Optimizer
    optimizer = Adam(model.parameters(), lr=ifg_Config['learning_rate'], weight_decay=ifg_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, ifg_Config['num_epochs'], eta_min=ifg_Config['eta_min'])

    # train and tes
    trainer = RGCNTrainer(model, train_loader, optimizer, loss)
    tester = RGCNTester(model, test_loader, loss)

    print('start training.')

    for epoch in range(1, ifg_Config['num_epochs'] + 1):
        trainer.train(epoch, device, ifg_Config['batch_size'])
        scheduler.step()
        if epoch % ifg_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, ifg_Config['test_batch_size'], ifg_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练

            torch.cuda.empty_cache()


def mvcnn_main():
    # Initialize wandb
    wandb.init(project=img_Config['project_name'], name=img_Config['svcnn'], config=img_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = img_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    early_stopping = EarlyStopping(save_path=img_Config['save_path'] + '\\early_stopping', patience=img_Config['early_stopping'])

    # data
    train_dataset = MultiViewDataSet(img_Config['data_dir'], data_type='train', view_name=img_Config['view_name'], data_transform=transform)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=img_Config['batch_size'], shuffle=True, drop_last=False)

    tset_test = MultiViewDataSet(img_Config['data_dir'], data_type='test', view_name=img_Config['view_name'], data_transform=transform)
    test_loader = DataLoader(tset_test, num_workers=0, batch_size=img_Config['test_batch_size'], shuffle=False, drop_last=False)

    # model
    print('Creating model...')
    classes = train_dataset.classes
    svcnn = SVCNN(num_classes=len(classes), pretraining=img_Config['pretraining'], cnn_name=img_Config['svcnn']).to(device)
    mvcnn = MVCNN(svcnn_model=svcnn, num_classes=len(classes), num_views=img_Config['num_views']).to(device)
    del svcnn

    # Loss and Optimizer
    optimizer = Adam(mvcnn.parameters(), lr=img_Config['learning_rate'], weight_decay=img_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, img_Config['num_epochs'], img_Config['eta_min'])

    # train and test
    trainer = MVCNNTrainer(mvcnn, train_loader, optimizer, loss)
    tester = MVCNNTester(mvcnn, test_loader, loss)

    print('start training.')
    for epoch in range(1, img_Config['num_epochs'] + 1):
        trainer.train(epoch, device, img_Config['batch_size'])
        scheduler.step()
        if epoch % img_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, img_Config['test_batch_size'], img_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练


def fusion_main():
    # Initialize wandb
    wandb.init(project=gwf_Config['project_name'], name=gwf_Config['run_name'], config=gwf_Config)

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

    # # Search
    # print("Running grid search for best fusion weights...")
    # best_weights, best_cluster_labels = grid_search_group_weights(softmax_mvcnn, softmax_rgcn, labels, len(classes))
    # np.save(os.path.join(gwf_Config['save_path'], 'best_weights.npy'), best_weights)
    # np.save(os.path.join(gwf_Config['save_path'], 'best_cluster_labels.npy'), best_cluster_labels)
    # print(best_weights)

    # Test
    print("Evaluating fusion model on test set...")
    best_weights = np.load(os.path.join(gwf_Config['save_path'], 'best_weights.npy'), allow_pickle=True).item()
    best_cluster_labels = np.load(os.path.join(gwf_Config['save_path'], 'best_cluster_labels.npy'), allow_pickle=True).item()
    test_fusion_model(softmax_mvcnn, softmax_rgcn, labels, best_weights, best_cluster_labels, g=gwf_Config['k_values'])

    wandb.finish()


if __name__ == '__main__':
    # ifc_schema_graph_main()
    # ifc_file_graph_main()
    # mvcnn_main()
    fusion_main()

