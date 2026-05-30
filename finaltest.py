import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import csv
import time

# 1. Архитектура модели (должна присутствовать для корректной работы torch.load, если сохраняли класс)
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
    """
    Загружает координаты из CSV.
    Ожидаемый формат без заголовка (или пропусти первую строку, если он есть):
    id_места, x1, y1, x2, y2, x3, y3, x4, y4
    """
    spots = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        # Если в файле есть заголовки, раскомментируй следующую строку:
        # next(reader)
        for row in reader:
            if len(row) >= 9:
                spot_id = row[0]
                pts = np.array([
                    [int(row[1]), int(row[2])],
                    [int(row[3]), int(row[4])],
                    [int(row[5]), int(row[6])],
                    [int(row[7]), int(row[8])]
                ], dtype=np.int32)
                spots.append((spot_id, pts))
    return spots

def extract_patch(image, pts):
    """Вырезает прямоугольник, описывающий четырехугольник места"""
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    patch = image[y:y+h, x:x+w]
    return patch

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Загрузка модели
    model_path = 'full_parking_model_tuned.pth'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Загрузка разметки
    csv_path = 'parking_grid_spring.csv'
    spots = load_parking_grid(csv_path)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Подключение к камере.
    # 0 - встроенная вебкамера. Для IP-камеры вставь URL, например: 'rtsp://admin:pass@192.168.1.10:554/stream'
    cap = cv2.VideoCapture(0)

    print("Система запущена. Проверка каждые 2 минуты. Нажми 'q' в окне камеры для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка получения кадра с камеры. Повторная попытка...")
            time.sleep(5)
            continue

        display_frame = frame.copy()
        occupied_count = 0
        empty_count = 0

        print(f"\n--- Отчет от {time.strftime('%H:%M:%S')} ---")

        with torch.no_grad():
            for spot_id, pts in spots:
                # 1. Вырезаем патч
                patch = extract_patch(frame, pts)

                if patch.size == 0:
                    continue

                # 2. Подготовка для модели
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(patch_rgb)
                tensor_img = transform(pil_img).unsqueeze(0).to(device)

                # 3. Инференс
                outputs = model(tensor_img)
                _, predicted = torch.max(outputs, 1)
                is_occupied = predicted.item() == 1

                # 4. Отрисовка
                if is_occupied:
                    color = (0, 0, 255) # Красный - занято
                    occupied_count += 1
                    status_text = "Zanyato"
                else:
                    color = (0, 255, 0) # Зеленый - свободно
                    empty_count += 1
                    status_text = "Svobodno"

                # Рисуем сам четырехугольник
                cv2.polylines(display_frame, [pts], isClosed=True, color=color, thickness=2)

                # Подпись ID места
                cv2.putText(display_frame, f"ID:{spot_id}", (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                print(f"Место {spot_id}: {status_text}")

        print(f"Итого: {empty_count} свободных, {occupied_count} занятых.")

        # Вывод изображения на экран
        cv2.imshow('Parking Monitor', display_frame)

        # Ожидание 2 минуты (120 000 миллисекунд).
        # Разбито на частые проверки, чтобы окно не "зависало" и можно было нажать 'q'
        wait_time_seconds = 120
        exit_flag = False
        for _ in range(wait_time_seconds * 10):
            if cv2.waitKey(100) & 0xFF == ord('q'):
                exit_flag = True
                break

        if exit_flag:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()