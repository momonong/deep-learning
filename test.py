import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from dataset import CustomImageDataset
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ⚙️ 設定
batch_size = 128
num_classes = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🧠 模型：ResNet50（結構須與 train 時一致）
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
model.eval()

# 📁 測試資料集
print("\n🔍 使用最佳模型進行測試...")
test_dir = "data/test"
answer_path = "data/answer.csv"

if os.path.exists(answer_path):
    answer_df = pd.read_csv(answer_path)
    test_samples = []
    for _, row in answer_df.iterrows():
        img_filename = f"test_{int(row['ID']):04d}.jpg"
        img_path = os.path.join(test_dir, img_filename)
        if os.path.exists(img_path):
            test_samples.append((img_path, row['Target']))
        else:
            print(f"⚠️ 找不到圖片：{img_path}，已略過")

    test_dataset = CustomImageDataset(test_samples, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_preds += out.argmax(dim=1).tolist()
            test_labels += y.tolist()

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"🎯 Test Accuracy: {test_acc:.4f}")

    # 📊 混淆矩陣
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spaghetti", "ramen", "udon"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig("confusion_matrix.png")
    print("🖼️ confusion_matrix.png 已儲存！")

else:
    print("⚠️ 找不到測試集答案檔案，跳過 test 評估。")
