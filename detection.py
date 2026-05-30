import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import csv
import time
import os
import requests                          # ← добавлено
from datetime import datetime

# --- CONFIG ---
BASE_PATH = "C:/ParkingDataset"
MODEL_PATH = 'full_parking_model_tuned_ba_new.pth'
CSV_PATH = 'parking_grid_summer_ba_fall.csv'
CAM_INDEX = 1
W, H = 2048, 1536
CHECK_INTERVAL = 120
SAVE_INTERVAL = 600
DISPLAY_SCALE = 0.5

# --- СЕРВЕР (измените под свои данные) ---
SERVER_IP  = "100.75.12.72"       # ← Tailscale IP сервера
SERVER_PORT = 5000
PARKING_ID  = "parking_2"      # ← "parking_2" на втором компьютере


# --- ARCHITECTURE ---
class ParkingCNN(nn.Module):
    def __init__(self):
        super(ParkingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
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


def load_parking_grid(csv_path):
    spots = []
    if not os.path.exists(csv_path): return spots
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= 8:
                try:
                    pts = np.array([[int(row[0]), int(row[1])], [int(row[2]), int(row[3])],
                                    [int(row[4]), int(row[5])], [int(row[6]), int(row[7])]], dtype=np.int32)
                    spots.append((f"Spot_{i + 1}", pts))
                except ValueError:
                    continue
    return spots


# --- ОТПРАВКА НА СЕРВЕР ---
def send_to_server(free: int, occupied: int):
    url = f"http://{SERVER_IP}:{SERVER_PORT}/update"
    try:
        resp = requests.post(url, json={
            "parking_id": PARKING_ID,
            "free":       free,
            "occupied":   occupied
        }, timeout=10)
        print(f"  → Сервер: {resp.status_code} | free={free}, occupied={occupied}")
    except requests.exceptions.ConnectionError:
        print("  → Сервер недоступен (ConnectionError)")
    except requests.exceptions.Timeout:
        print("  → Сервер не ответил (Timeout)")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()

    original_spots = load_parking_grid(CSV_PATH)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    last_save_time = 0
    print(f"Система запущена. Парковка: {PARKING_ID} | Сервер: {SERVER_IP}:{SERVER_PORT}")

    while True:
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        time.sleep(10)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Ошибка камеры. Пробую через 30 сек...")
            time.sleep(30)
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        now_ts = time.time()
        now_dt = datetime.now()

        # 1. СОХРАНЕНИЕ
        if now_ts - last_save_time >= SAVE_INTERVAL:
            folder = os.path.join(BASE_PATH, now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H"))
            os.makedirs(folder, exist_ok=True)
            filename = f"{now_dt.strftime('%M-%S')}_heartbeat.jpg"
            cv2.imwrite(os.path.join(folder, filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"[{now_dt.strftime('%H:%M:%S')}] RAW сохранен.")
            last_save_time = now_ts

        # 2. АНАЛИЗ
        cam_h, cam_w = frame.shape[:2]
        scale_x, scale_y = cam_w / W, cam_h / H
        display_frame = frame.copy()
        occupied, empty = 0, 0

        with torch.no_grad():
            for spot_id, pts in original_spots:
                curr_pts = (pts.astype(np.float32) * [scale_x, scale_y]).astype(np.int32)
                x, y, w_p, h_p = cv2.boundingRect(curr_pts)
                patch = frame[max(0, y):y + h_p, max(0, x):x + w_p]
                if patch.size == 0: continue

                pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                tensor_img = transform(pil_img).unsqueeze(0).to(device)
                is_occupied = torch.max(model(tensor_img), 1)[1].item() == 1

                color = (0, 0, 255) if is_occupied else (0, 255, 0)
                if is_occupied:
                    occupied += 1
                else:
                    empty += 1
                cv2.polylines(display_frame, [curr_pts], True, color, 4)

        # 3. ПРЕВЬЮ + ОТПРАВКА
        print(f"[{now_dt.strftime('%H:%M:%S')}] Свободно: {empty}, Занято: {occupied}")
        send_to_server(free=empty, occupied=occupied)   # ← единственная новая строка

        preview_frame = cv2.resize(display_frame, (int(cam_w * DISPLAY_SCALE), int(cam_h * DISPLAY_SCALE)))
        cv2.imshow('Parking Monitor', preview_frame)

        # 4. УМНОЕ ОЖИДАНИЕ
        start_wait = time.time()
        while time.time() - start_wait < CHECK_INTERVAL:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    main()
