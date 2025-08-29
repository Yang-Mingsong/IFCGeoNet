import torch
import wandb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
OMP_NUM_THREADS = 1


def grid_search_group_weights(softmax_mvcnn, softmax_rgcn, labels, num_classes, k_values=None):
    if k_values is None:
        k_values = list(range(1, num_classes + 1))
        print("k_values={}".format(k_values))

    results = {}
    best_cluster_labels = {}


    preds_m = np.argmax(softmax_mvcnn, axis=1)
    preds_r = np.argmax(softmax_rgcn, axis=1)

    labels_idx = np.arange(num_classes)

    f1_mvcnn = f1_score(labels, preds_m, average=None, labels=labels_idx, zero_division=0)
    f1_rgcn = f1_score(labels, preds_r, average=None, labels=labels_idx, zero_division=0)

    delta_f1 = f1_mvcnn - f1_rgcn

    # （可选）整体 Macro-F1
    macro_m = f1_score(labels, preds_m, average='macro', zero_division=0)
    macro_r = f1_score(labels, preds_r, average='macro', zero_division=0)

    print("mvcnnF1 per-class:", f1_mvcnn)
    print("rgcnF1  per-class:", f1_rgcn)
    print("ΔF1 per-class:", delta_f1)
    print("Macro-F1 MVCNN=", macro_m, " RGCN=", macro_r, " Δ=", macro_m - macro_r)

    # 逐个 K 值进行搜索
    for K in k_values:
        print("Running K-means clustering with K={}...".format(K))

        # **Step 2: K-means 聚类**
        delta_f1_reshaped = delta_f1.reshape(-1, 1)  # 变为二维数组
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(delta_f1_reshaped)

        print("K-means cluster assignments: {}".format(cluster_labels))
        wandb.log({
            "K": K,
            "cluster_labels": cluster_labels
        })

        # 计算每个类别组的权重
        best_weights = torch.zeros(2, K)
        total_f1 = 0.0

        for cluster in range(K):
            cluster_mask = np.isin(labels, np.where(cluster_labels == cluster)[0])
            if cluster_mask.sum() == 0:
                print("Cluster {} has no samples, skipping.".format(cluster))
                continue
            cluster_labels_real = labels[cluster_mask]

            # 获取该类别组的 softmax 结果
            softmax_mvcnn_group = softmax_mvcnn[cluster_mask]
            softmax_rgcn_group = softmax_rgcn[cluster_mask]

            best_w1, best_w2 = 0.5, 0.5
            best_f1 = -1

            # **Step 4: 网格搜索 w1, w2**
            for w1 in torch.arange(0, 1.1, 0.1):
                w2 = 1 - w1
                fused = w1 * softmax_mvcnn_group + w2 * softmax_rgcn_group
                preds = torch.argmax(fused, dim=1)

                if len(cluster_labels_real) != len(preds):
                    print("Cluster {} mismatch: labels={}, preds={}".format(cluster, len(cluster_labels_real), len(preds)))
                    continue

                f1 = f1_score(cluster_labels_real, preds, average='macro', zero_division=1)
                print("Cluster {}, w1={}, w2={}, f1={}".format(cluster, w1.item(), w2.item(), f1))

                if f1 > best_f1:
                    best_w1, best_w2 = w1.item(), w2.item()
                    best_f1 = f1

            if best_f1 == -1:
                print("No valid F1-score found for cluster {}, using default (0.5, 0.5)".format(cluster))
                best_f1 = 0.0

            best_weights[:, cluster] = torch.tensor([best_w1, best_w2])
            total_f1 += best_f1

        # 计算该 group_size 下的最终final_f1-score
        final_preds = np.zeros_like(labels)

        for cluster in range(K):
            cluster_mask = np.isin(labels, np.where(cluster_labels == cluster)[0])
            if cluster_mask.sum() == 0:
                continue

            w1, w2 = best_weights[:, cluster]
            fused_probs = w1 * softmax_mvcnn[cluster_mask] + w2 * softmax_rgcn[cluster_mask]
            final_preds[cluster_mask] = np.argmax(fused_probs, axis=1)

        final_f1 = f1_score(labels, final_preds, average='macro', zero_division=1)
        results[K] = (best_weights, final_f1)
        best_cluster_labels[K] = cluster_labels  # 存储最佳 K 值对应的 cluster_labels
    return results, best_cluster_labels
