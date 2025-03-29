from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms


# 1. 掃描資料夾中的所有樣本
dataset = ImageFolder("data/train")
all_samples = dataset.samples  # List of (image_path, label)
label_names = dataset.classes  # e.g. ['0_spaghetti', '1_ramen', '2_udon']

# 2. 用 stratify 確保每一類都有切分到
train_samples, val_samples = train_test_split(
    all_samples,
    test_size=0.2,             # 驗證資料佔 20%
    stratify=[s[1] for s in all_samples],
    random_state=42           # 確保每次切分一樣
)
from collections import Counter

train_labels = [s[1] for s in train_samples]
val_labels = [s[1] for s in val_samples]

print("Train label distribution:", Counter(train_labels))
print("Val label distribution:", Counter(val_labels))

print(f"Total samples: {len(all_samples)}")
print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")



class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CustomImageDataset(train_samples, transform=transform)
val_dataset = CustomImageDataset(val_samples, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape)

print("範例 label 分布：", torch.bincount(y))
