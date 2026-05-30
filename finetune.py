import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1. Архитектура (СТРОГО под 128x128, чтобы веса подошли)
class ParkingCNN(nn.Module):
    def __init__(self):
        super(ParkingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 128 -> 64 -> 32 -> 16. 64 * 16 * 16 = 16384
        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Датасет с относительными путями
class YOLOParkingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Используем относительные пути
        images_dir = os.path.join(root_dir, 'ParkingDataset-1')
        labels_dir = os.path.join(root_dir, 'Labels_Output')

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Не нашел папку с картинками по пути: {images_dir}")

        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    rel_path = os.path.relpath(img_path, images_dir)
                    label_path = os.path.join(labels_dir, os.path.splitext(rel_path)[0] + '.txt')

                    if os.path.exists(label_path):
                        self._process_image(img_path, label_path)

    def _process_image(self, img_path, label_path):
        img = cv2.imread(img_path)
        if img is None: return
        h, w, _ = img.shape
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    cls, x_c, y_c, bw, bh = map(float, parts)
                    left = int((x_c - bw/2) * w)
                    top = int((y_c - bh/2) * h)
                    right = int((x_c + bw/2) * w)
                    bottom = int((y_c + bh/2) * h)
                    patch = img[max(0, top):min(h, bottom), max(0, left):min(w, right)]
                    if patch.size > 0:
                        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                        self.samples.append((Image.fromarray(patch), int(cls)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        patch, target = self.samples[idx]
        if self.transform: patch = self.transform(patch)
        return patch, target

# 3. Основная функция тренировки
def train_and_save():
    # ВКЛЮЧАЕМ MPS (ДЛЯ MAC)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"Используем устройство: {DEVICE}")

    # Относительный путь к датасету
    DATA_PATH = "my_dataset_ba_new"
    OLD_WEIGHTS = "parking_cnn_weights.pth"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = YOLOParkingDataset(DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ParkingCNN().to(DEVICE)

    # Загружаем старые веса
    if os.path.exists(OLD_WEIGHTS):
        model.load_state_dict(torch.load(OLD_WEIGHTS, map_location=DEVICE))
        print("Веса из PKLot успешно загружены. Начинаем дообучение...")
    else:
        print("ВНИМАНИЕ: Старые веса не найдены, обучение начнется с нуля!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Низкий LR для тонкой настройки

    model.train()
    for epoch in range(5): # Для начала 5 эпох хватит
        running_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Эпоха {epoch+1}, Loss: {running_loss/len(loader):.4f}")

    # ПОЛНОЕ СОХРАНЕНИЕ
    torch.save(model, 'full_parking_model_tuned_ba.pth')
    print("Дообученная модель сохранена целиком в full_parking_model_tuned.pth")

if __name__ == "__main__":
    train_and_save()