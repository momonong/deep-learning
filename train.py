import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from dataset import CustomImageDataset, train_samples, val_samples
from tqdm import tqdm
from augment import train_transform, val_transform


# ⚙️ 設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
num_epochs = 30
batch_size = 128
learning_rate = 3e-4

# 📦 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 📂 Dataset & Loader
train_dataset = CustomImageDataset(train_samples, transform=train_transform)
val_dataset = CustomImageDataset(val_samples, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 🧠 模型：ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 🎯 Loss / Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 📊 記錄用
train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []
best_acc = 0

# 🔁 訓練迴圈
for epoch in range(num_epochs):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds += out.argmax(dim=1).tolist()
        train_labels += y.tolist()

    train_acc = accuracy_score(train_labels, train_preds)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # 🔍 驗證
    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} [Val]"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            val_preds += out.argmax(dim=1).tolist()
            val_labels += y.tolist()

    val_acc = accuracy_score(val_labels, val_preds)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.3f} | Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.3f} | Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved new best model!")

# 📈 畫圖：loss & acc
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
print("📊 training_plot.png 已儲存！")
