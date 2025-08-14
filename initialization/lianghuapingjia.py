import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def compute_clustering_metrics(file_path, name=''):
    # 读取特征和标签
    data = torch.load(file_path)
    features = data['features'].squeeze(1)  # [N, D] 去掉多余维度
    labels = data['labels']

    X = features.numpy()
    y = labels.numpy()

    # 计算三个指标
    silhouette = silhouette_score(X, y)
    ch_score = calinski_harabasz_score(X, y)
    db_score = davies_bouldin_score(X, y)

    print(f"{name} Embeddings:")
    print(f"Silhouette Score:       {silhouette:.4f}")
    print(f"Calinski-Harabasz Score:{ch_score:.4f}")
    print(f"Davies-Bouldin Score:   {db_score:.4f}")
    print("-" * 40)




# 示例：对三个初始化进行比较
compute_clustering_metrics(r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\val_random_features.pt', name='Random')
compute_clustering_metrics(r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\val_our_features.pt', name='Our')
compute_clustering_metrics(r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\test_meta_features.pt', name='Meta')
compute_clustering_metrics(r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\val_domaininit_features.pt', name='MMD')
compute_clustering_metrics(r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\val_deterministic_onehot_features.pt', name='onehot')
