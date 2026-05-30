import cv2
import csv
import os
import numpy as np
import json

# Настройки
GRID_CSV = "parking_grid_summer_ba_new.csv"
IMAGES_DIR = "my_dataset_ba_test/ParkingDataset-1"
LABELS_BASE_DIR = "my_dataset_ba_test/Labels_Output"
CHECKPOINT_FILE = "progress_qqq_new.json"

# Загружаем полигоны
polygons = []
try:
    with open(GRID_CSV, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                pts = list(map(int, row))
                polygons.append([(pts[i], pts[i+1]) for i in range(0, 8, 2)])
except FileNotFoundError:
    print("Ошибка: Сначала создай сетку!")
    exit()

# Загружаем чекпоинт
last_processed_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        last_processed_idx = json.load(f).get("last_idx", 0)

status = [1] * len(polygons)
last_confirmed_status = [1] * len(polygons)

def toggle_status(event, x, y, flags, param):
    global status
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, poly in enumerate(polygons):
            if cv2.pointPolygonTest(np.array(poly, np.int32), (x, y), False) >= 0:
                status[i] = 1 if status[i] == 0 else 0
                draw_current_state()
                break

def draw_current_state():
    global current_image, clone_img
    current_image = clone_img.copy()
    for i, poly in enumerate(polygons):
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        color = (0, 0, 255) if status[i] == 1 else (0, 255, 0)
        cv2.polylines(current_image, [pts], True, color, 2)
        cv2.putText(current_image, str(i), poly[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imshow("Annotator", current_image)

# Собираем и СОРТИРУЕМ пути
image_paths = []
for root, _, files in os.walk(IMAGES_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'):
            image_paths.append(os.path.join(root, f))
image_paths.sort() # Сортировка для порядка

cv2.namedWindow("Annotator")
cv2.setMouseCallback("Annotator", toggle_status)

# Начинаем с чекпоинта
for idx in range(last_processed_idx, len(image_paths)):
    img_path = image_paths[idx]

    nparr = np.fromfile(img_path, np.uint8)
    clone_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if clone_img is None: continue

    status = [1] * len(polygons)
    draw_current_state()

    print(f"[{idx+1}/{len(image_paths)}] Обработка: {img_path}")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            status = last_confirmed_status.copy()
            draw_current_state()
        elif key == ord(' '):
            last_confirmed_status = status.copy()

            # Сохранение структуры папок
            rel_path = os.path.relpath(img_path, IMAGES_DIR)
            target_dir = os.path.join(LABELS_BASE_DIR, os.path.dirname(rel_path))
            if not os.path.exists(target_dir): os.makedirs(target_dir)

            save_path = os.path.join(target_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

            # Сохраняем в YOLO формате
            with open(save_path, "w") as f:
                img_h, img_w = clone_img.shape[:2]
                for i, poly in enumerate(polygons):
                    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                    # Расчет YOLO координат как в Train2.ipynb
                    x_c = ((min(xs) + max(xs)) / 2) / img_w
                    y_c = ((min(ys) + max(ys)) / 2) / img_h
                    w = (max(xs) - min(xs)) / img_w
                    h = (max(ys) - min(ys)) / img_h
                    f.write(f"{status[i]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            # Обновляем чекпоинт
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"last_idx": idx + 1}, f)
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()