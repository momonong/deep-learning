# 1. 匯入必要套件
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import os

# 2. 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用設備：", device)

# 3. 定義訓練相關超參數
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
IMG_SIZE = 224  # 將圖片 resize 到固定大小（常用在 CNN）
