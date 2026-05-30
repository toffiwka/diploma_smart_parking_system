import cv2
import csv
import numpy as np

# Настройки
REFERENCE_IMAGE = "56-55_heartbeat.jpg"  # Подставь путь к любому фото пустой парковки
OUTPUT_CSV = "parking_grid_summer_pa_fall.csv"

points = []
polygons = []
image = cv2.imread(REFERENCE_IMAGE)
clone = image.copy() if image is not None else None

def draw_polygon(event, x, y, flags, param):
    global points, polygons, image

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1) # Рисуем точку

        # Когда поставили 4 точки - замыкаем фигуру
        if len(points) == 4:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            polygons.append(points.copy())
            points.clear()

        cv2.imshow("Grid Maker", image)

if clone is None:
    print("Ошибка: Не удалось загрузить изображение.")
else:
    cv2.namedWindow("Grid Maker")
    cv2.setMouseCallback("Grid Maker", draw_polygon)

    print("ИНСТРУКЦИЯ:")
    print("- Кликай 4 раза, чтобы выделить парковочное место.")
    print("- Нажми 'C', чтобы отменить последнюю фигуру (Clear last).")
    print("- Нажми 'S', чтобы сохранить сетку и выйти.")

    cv2.imshow("Grid Maker", image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"): # Сохранить и выйти
            with open(OUTPUT_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                for poly in polygons:
                    # Сохраняем как x1,y1, x2,y2, x3,y3, x4,y4
                    flat_list = [coord for pt in poly for coord in pt]
                    writer.writerow(flat_list)
            print(f"Сохранено {len(polygons)} мест в {OUTPUT_CSV}")
            break
        elif key == ord("c"): # Отменить последнюю фигуру
            if polygons:
                polygons.pop()
                image = clone.copy() # Очищаем и перерисовываем
                points.clear()
                for poly in polygons:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imshow("Grid Maker", image)

    cv2.destroyAllWindows()