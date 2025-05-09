import cv2
import numpy as np
from shapely.geometry import Polygon

def apply_fisheye(
    image: np.ndarray,
    k1_range: tuple = (0.1, 0.3),  # Случайный коэффициент из диапазона
    k2_range: tuple = (0.01, 0.1),
    center_var: float = 0.1  # Смещение центра искажения (для реализма)
) -> tuple[np.ndarray, dict]:
    """Применяет fisheye-эффект с вариативными параметрами.
    Возвращает искаженное изображение и параметры (k1, k2, center)."""
    h, w = image.shape[:2]

    # Случайные параметры
    k1 = np.random.uniform(*k1_range)
    k2 = np.random.uniform(*k2_range)
    center = (w/2 + np.random.uniform(-center_var, center_var) * w,
              h/2 + np.random.uniform(-center_var, center_var) * h)

    # Векторизованные вычисления (быстрее в 100+ раз)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    nx = (x - center[0]) / (w / 2)
    ny = (y - center[1]) / (h / 2)
    r = np.sqrt(nx**2 + ny**2)
    distortion = 1 + k1 * r**2 + k2 * r**4

    map_x = (nx * distortion + 1) * (w / 2)
    map_y = (ny * distortion + 1) * (h / 2)

    # Исправление выхода за границы
    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    fisheye_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    params = {'k1': k1, 'k2': k2, 'center': center}
    return fisheye_img, params

def apply_fisheye_to_bbox(bbox: list, img_size: tuple, params: dict) -> list:
    """Пересчет BBox с учетом параметров искажения."""
    x, y, w, h = bbox
    img_w, img_h = img_size
    corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    distorted_corners = []
    for (px, py) in corners:
        nx = (px - params['center'][0]) / (img_w / 2)
        ny = (py - params['center'][1]) / (img_h / 2)
        r = np.sqrt(nx**2 + ny**2)
        distortion = 1 + params['k1'] * r**2 + params['k2'] * r**4
        dx = nx * distortion * (img_w / 2) + params['center'][0]
        dy = ny * distortion * (img_h / 2) + params['center'][1]
        distorted_corners.append([dx, dy])

    polygon = Polygon(distorted_corners)
    bounds = polygon.bounds
    new_bbox = [
        max(0, bounds[0]),
        max(0, bounds[1]),
        min(img_w, bounds[2]) - max(0, bounds[0]),
        min(img_h, bounds[3]) - max(0, bounds[1])
    ]
    return [round(coord, 2) for coord in new_bbox]
