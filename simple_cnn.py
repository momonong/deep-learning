import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (3,224,224) → (16,224,224)
        self.pool = nn.MaxPool2d(2, 2)                            # (16,224,224) → (16,112,112)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (32,112,112)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)                   # Flatten 後進入 Linear
        self.fc2 = nn.Linear(128, num_classes)                   # 輸出類別數

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))               # fc1 + relu
        x = self.fc2(x)                       # fc2 (輸出 logits)
        return x
