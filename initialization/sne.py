import torch
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免 PyCharm 图形显示报错
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载特征文件
data = torch.load(r'Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\4val_domaininit_features.pt')
features = data['features']
labels = data['labels']

# 去除多余维度
if features.dim() == 3 and features.shape[1] == 1:
    features = features.squeeze(1)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# 转为 numpy
X = features.numpy()
y = labels.numpy()

# 进行 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
X_2d = tsne.fit_transform(X)

# 绘图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab20', s=10)
plt.colorbar(scatter, ticks=range(len(set(y))))
plt.title("t-SNE of Meta-Initialized Graph Features")
plt.tight_layout()
plt.grid(True)

# 保存图像
plt.savefig(r"Q:\pychem_project\XUT-GeoIFCNet-Master\initialization\result\4_tsne_mmd_features.png", dpi=300)
print("t-SNE 图像已保存为 tsne_mmd_features.png")
