import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
IMG_SIZE = 224