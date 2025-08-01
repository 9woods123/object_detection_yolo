from ultralytics import YOLO
import torch
from pathlib import Path
print(torch.cuda.is_available())  # 应该返回 True
print(torch.version.cuda)         # 应该显示 CUDA 版本
print(torch.cuda.get_device_name(0))  # 应该显示 "NVIDIA GeForce RTX 5080"

# model = YOLO('yolo11.yaml')
model = YOLO('yolo11n.pt')

model.train(data='train_cfg.yaml', epochs=100,
               device=[0])



