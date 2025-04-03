from thop import profile
import torch
from models.mvcnn import MVCNN, SVCNN


svcnn = SVCNN(num_classes=20, pretraining=True, cnn_name="resnet34")
mvcnn = MVCNN(svcnn_model=svcnn, num_classes=20, num_views=12)
del svcnn

# 创建一个示例输入
input_tensor = torch.randn(12, 3, 224, 224)  # 假设输入的形状为 (batch_size*view_num, channels, height, width)

flops, params = profile(mvcnn, inputs=(input_tensor,))
print(f'Total FLOPs: {flops}')
print(f'Total parameters: {params}')
