import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from urllib.request import urlretrieve
import urllib.error

# Параметры камеры "рыбий глаз" (примерные, требуют калибровки)
focal_length = 350  # Параметр f, упомянутый в статье
image_size = (512, 512)  # Размер изображения
aspect_ratio = (1.0, 1.0)  # (a_x, a_y)
poly_coeffs = [0.0001, 0.00005, 0.00001, 0.000001]  # Коэффициенты полинома 4-й степени

# 1. Функция обратной проекции (2D → 3D)
def unproject(image, focal_length, aspect_ratio):
    h, w = image.shape[:2]
    u = np.linspace(-w/2, w/2, w)
    v = np.linspace(-h/2, h/2, h)
    u, v = np.meshgrid(u, v)
    
    # Нормализация координат
    u = u / (w/2) * aspect_ratio[0]
    v = v / (h/2) * aspect_ratio[1]
    
    # Вычисление расстояния ρ
    rho = np.sqrt(u**2 + v**2)
    
    # Полиномиальная модель для угла θ
    theta = np.zeros_like(rho)
    for i, coeff in enumerate(poly_coeffs):
        theta += coeff * rho**(i+1)
    
    # Преобразование в 3D координаты на единичной сфере
    x = np.cos(theta) * u / rho
    y = np.cos(theta) * v / rho
    z = np.sin(theta)
    
    # Для ρ=0, чтобы избежать деления на ноль
    mask = rho == 0
    x[mask] = 0
    y[mask] = 0
    z[mask] = 1
    
    return np.stack([x, y, z], axis=-1)

# 2. Функция трансформации (поворот/смещение)
def transform_points(points_3d, rotation_matrix, translation_vector):
    # Применяем поворот
    transformed = np.einsum('ij,...j->...i', rotation_matrix, points_3d)
    # Применяем смещение (для единичной сферы смещение менее релевантно)
    transformed += translation_vector
    # Нормализуем обратно на единичную сферу
    norm = np.linalg.norm(transformed, axis=-1, keepdims=True)
    transformed /= norm + 1e-6  # Избегаем деления на ноль
    return transformed

# 3. Функция повторной проекции (3D → 2D)
def reproject(points_3d, focal_length, aspect_ratio, image_size):
    h, w = image_size
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    
    # Вычисляем угол θ
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    # Полиномиальная модель для ρ
    rho = np.zeros_like(theta)
    for i, coeff in enumerate(poly_coeffs):
        rho += coeff * theta**(i+1)
    
    # Преобразуем в пиксельные координаты
    u = x / np.sqrt(x**2 + y**2) * rho * (w/2) / aspect_ratio[0]
    v = y / np.sqrt(x**2 + y**2) * rho * (h/2) / aspect_ratio[1]
    
    # Для z=0, чтобы избежать деления на ноль
    mask = np.sqrt(x**2 + y**2) == 0
    u[mask] = 0
    v[mask] = 0
    
    # Смещаем координаты к центру изображения
    u = u + w/2
    v = v + h/2
    
    return u, v

# 4. Полная аугментация изображения
def augment_viewpoint(image, rotation_matrix, translation_vector):
    # Обратная проекция
    points_3d = unproject(image, focal_length, aspect_ratio)
    
    # Трансформация
    transformed_3d = transform_points(points_3d, rotation_matrix, translation_vector)
    
    # Повторная проекция
    u, v = reproject(transformed_3d, focal_length, aspect_ratio, image_size)
    
    # Интерполяция изображения
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    augmented = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return augmented

# 5. Загрузка датасета (KITTI sample with fallback)
def download_sample_image():
    # Primary URL: KITTI raw data
    kitti_url = "http://www.cvlibs.net/datasets/kitti/raw_data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000005.png"
    # Fallback URL: BDD100K sample
    fallback_url = "https://github.com/ucbdrive/bdd100k/raw/master/images/10k/train/0000f77c-6257be58.jpg"
    filename = "sample_image.png"
    
    if not os.path.exists(filename):
        for url in [kitti_url, fallback_url]:
            try:
                print(f"Downloading sample image from {url}")
                urlretrieve(url, filename)
                # Verify the downloaded file is a valid image
                img = cv2.imread(filename)
                if img is None:
                    print(f"Downloaded file {filename} is not a valid image.")
                    os.remove(filename)  # Remove invalid file
                    continue
                print(f"Successfully downloaded {filename}")
                return filename
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"Failed to download from {url}: {e}")
                continue
        raise ValueError("Failed to download a valid image from all URLs. Please provide a local image or check the URLs.")
    return filename

# 6. Упрощённая модель ERFNet (заглушка для демонстрации)
class ERFNet(nn.Module):
    def __init__(self, num_classes=19):
        super(ERFNet, self).__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x

# 7. Основная функция для обработки и визуализации
def main():
    # Загрузка изображения
    image_path = download_sample_image()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check if the file is a valid image.")
    image = cv2.resize(image, image_size)  # Ресайз под размер модели
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Пример трансформации: поворот на 10 градусов вокруг оси Z
    theta = np.radians(10)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    translation_vector = np.array([0.1, 0.1, 0.1])
    
    # Аугментация
    augmented_image = augment_viewpoint(image, rotation_matrix, translation_vector)
    
    # Подготовка изображения для модели
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(augmented_image).unsqueeze(0)
    
    # Инициализация модели (заглушка)
    model = ERFNet(num_classes=19)  # Using 19 classes for consistency
    model.eval()
    
    # Получение предсказания
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).squeeze().numpy()
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Augmented Image (Fisheye)")
    plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Segmentation Mask")
    plt.imshow(pred, cmap='jet')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fisheye_augmentation_result.png')
    plt.show()

if __name__ == "__main__":
    main()
