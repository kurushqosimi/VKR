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
https://www.researchgate.net/publication/370853808_Surround-view_Fisheye_Camera_Viewpoint_Augmentation_for_Image_Semantic_Segmentation/fulltext/64664f4870202663165678ed/Surround-View-Fisheye-Camera-Viewpoint-Augmentation-for-Image-Semantic-Segmentation.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19&__cf_chl_tk=leUy6A19ggXi6g.bkqrxjMQauj3SGJ.ZAcKq1Kv7_q0-1746106779-1.0.1.1-eC7yYUXPqNyWsfi45jRNWdIc_jWTnJRuSgfqhepgWTQ




https://www.researchgate.net/publication/370853808_Surround-view_Fisheye_Camera_Viewpoint_Augmentation_for_Image_Semantic_Segmentation



Привет. Меня зовут Куруш. Я студент 4 курса направлении Прикладной математики и Информатики. В этом году я выпускаюсь, и тема моей выпускной квалификационной работы это Определение объекта на изображении с широкоугольного объектива. И было мною поставлено следующая гипотеза: Добавление в процесс обучения специальной ‘fisheye’-аугментации (искусственного искажения изображений под широкоугольную оптику) позволит значительно повысить точность детекции объектов на снимках с подобными искажениями, при этом не окажет существенного негативного влияния на качество распознавания обычных (неискажённых) изображений. Я поработал с YOLOv8n попробовал следующие пропорции 10% по 50%, т.е в обучении было использовано 10% искаженных изображений, далее 20%, и так до 50%. Чтоб не создавать самому датасет я использовал готовый датасет COCO2017 и написал свою аугментацию для искажения изображения, само собой координаты bbox-ов также поменялись. Итогом стало то что 20% дало самый актуальный результат по точности и простых изображений, и искаженных. Что я считаю неправильным выводом из-за неправильной работы аугментации. Можете посмотреть и улучшить работу моей аугментации, а до этого скачать больше готовых датасет как COCO2017. Вот код до аугментации: !pip install ultralytics albumentations opencv-python # Скачивание изображений и аннотаций (~1Гб)
!wget http://images.cocodataset.org/zips/val2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Распаковка
!unzip val2017.zip -d coco
!unzip annotations_trainval2017.zip -d coco
# Скачиваем инструмент от Ultralytics
!git clone https://github.com/ultralytics/JSON2YOLO.git
!pip install pycocotools !pip install -U sahi pycocotools from sahi.utils.coco import Coco

# Реальные пути на Colab:
coco_annotation_file = '/content/coco/annotations/instances_val2017.json'
coco_images_dir = '/content/coco/val2017'
yolo_output_dir = '/content/coco2017_yolo'

# Создание объекта Coco
coco = Coco.from_coco_dict_or_path(
    coco_annotation_file,
    image_dir=coco_images_dir
)

# Конвертация в YOLO-формат (80% изображения используются в train)
coco.export_as_yolo(
    output_dir=yolo_output_dir,
    train_split_rate=0.8
) import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import yaml
import math
from scipy.stats import truncnorm

# Пути к исходным изображениям и labels
images_dir = '/content/coco2017_yolo/train'
labels_dir = '/content/coco2017_yolo/train'

# Путь к новым данным
fisheye_images_dir = '/content/coco2017_yolo_fisheye/train'
fisheye_labels_dir = '/content/coco2017_yolo_fisheye/train'

os.makedirs(fisheye_images_dir, exist_ok=True)
os.makedirs(fisheye_labels_dir, exist_ok=True)

# Функция для создания более реалистичного эффекта fisheye
def apply_fisheye(image, strength=2.0, ellipticity=1.0, add_vignette=False, add_chromatic_aberration=False):
    """
    Создает реалистичный эффект fisheye с дополнительными эффектами
    Args:
        image: исходное изображение
        strength: сила искажения (1.0 - умеренное, 2.0 - сильное)
        ellipticity: эллиптичность искажения (1.0 - круговое, >1 - эллиптическое)
        add_vignette: добавлять ли виньетирование
        add_chromatic_aberration: добавлять ли хроматические аберрации
    Returns:
        искаженное изображение
    """
    height, width = image.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Нормализация координат
    x = (x - width/2) / (width/2)
    y = (y - height/2) / (height/2)

    # Применяем эллиптичность
    r = np.sqrt((x**2) + ((y*ellipticity)**2))
    theta = np.arctan2(y*ellipticity, x)

    # Искажение типа "рыбий глаз"
    r_new = np.arctan(r * strength) / (np.pi/2)

    # Преобразование обратно в декартовы координаты
    x_new = r_new * np.cos(theta) * width/2 + width/2
    y_new = r_new * np.sin(theta) / ellipticity * height/2 + height/2

    # Ограничение координат
    x_new = np.clip(x_new, 0, width-1).astype(np.float32)
    y_new = np.clip(y_new, 0, height-1).astype(np.float32)

    # Применяем искажение
    distorted = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if add_vignette:
        # Виньетирование
        rows, cols = distorted.shape[:2]  # Определяем размеры изображения
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        for c in range(3):
            distorted[:, :, c] = distorted[:, :, c] * mask

    if add_chromatic_aberration:
        # Хроматическая аберрация
        rows, cols = distorted.shape[:2]  # Определяем размеры изображения
        for c in range(3):
            shift = np.random.uniform(-2, 2)  # Случайный сдвиг для каждого канала
            M = np.float32([[1, 0, shift], [0, 1, shift]])
            distorted[:, :, c] = cv2.warpAffine(distorted[:, :, c], M, (cols, rows))

    # Маска для круглого вида
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width//2, height//2), int(min(width, height)/2), 255, -1)

    # Применение маски
    result = distorted.copy()
    for c in range(0, 3):
        result[:, :, c] = cv2.bitwise_and(result[:, :, c], mask)

    return result

# Функция для преобразования координат bbox в искаженное пространство
def transform_bbox_fisheye(bbox, width, height, strength=2.0, ellipticity=1.0):
    """
    Преобразует координаты bbox (формат YOLO) для соответствия искажению fisheye
    """
    x_center, y_center, w, h = bbox
    x1 = (x_center - w/2) * width
    y1 = (y_center - h/2) * height
    x2 = (x_center + w/2) * width
    y2 = (y_center + h/2) * height

    points = [[x1, y1], [x2, y1], [x1, y2], [x2, y2], [(x1+x2)/2, (y1+y2)/2]]
    transformed_points = []

    for x, y in points:
        x_norm = (x - width/2) / (width/2)
        y_norm = (y - height/2) / (height/2)
        r = math.sqrt(x_norm**2 + (y_norm*ellipticity)**2)
        theta = math.atan2(y_norm*ellipticity, x_norm)
        r_new = math.atan(r * strength) / (math.pi/2)
        x_new = r_new * math.cos(theta) * width/2 + width/2
        y_new = r_new * math.sin(theta) / ellipticity * height/2 + height/2
        x_new = min(max(0, x_new), width-1)
        y_new = min(max(0, y_new), height-1)
        transformed_points.append((x_new, y_new))

    xs = [p[0] for p in transformed_points[:4]]
    ys = [p[1] for p in transformed_points[:4]]
    new_center_x = transformed_points[4][0] / width
    new_center_y = transformed_points[4][1] / height
    new_width = (max(xs) - min(xs)) / width
    new_height = (max(ys) - min(ys)) / height

    return [
        min(max(0, new_center_x), 1),
        min(max(0, new_center_y), 1),
        min(max(0.01, new_width), 1),
        min(max(0.01, new_height), 1)
    ]

# Генерация случайной силы искажения с нормальным распределением
def get_random_strength():
    mean = 2.0
    std_dev = 0.5
    lower = 1.5
    upper = 3.0
    return truncnorm.rvs((lower - mean) / std_dev, (upper - mean) / std_dev, loc=mean, scale=std_dev)

# Обработка тренировочных данных
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(image_files, desc='Создание реалистичной fisheye-аугментации'):
    img_path = os.path.join(images_dir, img_name)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    lbl_path = os.path.join(labels_dir, label_name)

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    height, width = image.shape[:2]
    fisheye_strength = get_random_strength()  # Улучшенная генерация силы искажения
    ellipticity = np.random.uniform(0.9, 1.1)  # Случайная эллиптичность
    add_vignette = np.random.rand() < 0.5     # Вероятность добавления виньетирования
    add_chromatic_aberration = np.random.rand() < 0.5  # Вероятность добавления хроматической аберрации

    fisheye_image = apply_fisheye(
        image,
        strength=fisheye_strength,
        ellipticity=ellipticity,
        add_vignette=add_vignette,
        add_chromatic_aberration=add_chromatic_aberration
    )

    bboxes = []
    class_labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts[:5])
                    bboxes.append([x_center, y_center, bbox_width, bbox_height])
                    class_labels.append(int(class_id))
    else:
        continue

    transformed_bboxes = [
        transform_bbox_fisheye(bbox, width, height, strength=fisheye_strength, ellipticity=ellipticity)
        for bbox in bboxes
    ]

    aug_img_path = os.path.join(fisheye_images_dir, img_name)
    cv2.imwrite(aug_img_path, fisheye_image)

    aug_lbl_path = os.path.join(fisheye_labels_dir, label_name)
    with open(aug_lbl_path, 'w') as f:
        for bbox, cls in zip(transformed_bboxes, class_labels):
            x_center, y_center, bbox_width, bbox_height = bbox
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Обработка валидационных данных (аналогично тренировочным, но без случайности)
val_src = '/content/coco2017_yolo/val'
val_fisheye_dst = '/content/coco2017_yolo_fisheye/val'
os.makedirs(val_fisheye_dst, exist_ok=True)

val_image_files = [f for f in os.listdir(val_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(val_image_files, desc='Создание fisheye валидационных данных'):
    img_path = os.path.join(val_src, img_name)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    lbl_path = os.path.join(val_src, label_name)

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    height, width = image.shape[:2]
    fisheye_strength = 2.0  # Фиксированная сила искажения
    ellipticity = 1.0       # Без эллиптичности
    add_vignette = False    # Без виньетирования
    add_chromatic_aberration = False  # Без хроматических аберраций

    fisheye_image = apply_fisheye(
        image,
        strength=fisheye_strength,
        ellipticity=ellipticity,
        add_vignette=add_vignette,
        add_chromatic_aberration=add_chromatic_aberration
    )

    bboxes = []
    class_labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts[:5])
                    bboxes.append([x_center, y_center, bbox_width, bbox_height])
                    class_labels.append(int(class_id))
    else:
        continue

    transformed_bboxes = [
        transform_bbox_fisheye(bbox, width, height, strength=fisheye_strength, ellipticity=ellipticity)
        for bbox in bboxes
    ]

    aug_img_path = os.path.join(val_fisheye_dst, img_name)
    cv2.imwrite(aug_img_path, fisheye_image)

    aug_lbl_path = os.path.join(val_fisheye_dst, label_name)
    with open(aug_lbl_path, 'w') as f:
        for bbox, cls in zip(transformed_bboxes, class_labels):
            x_center, y_center, bbox_width, bbox_height = bbox
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Копируем и адаптируем YAML файл для новых данных
original_yaml = '/content/coco2017_yolo/data.yml'
new_yaml = '/content/coco2017_yolo_fisheye/data_fisheye.yml'

with open(original_yaml, 'r') as file:
    data_yaml = yaml.safe_load(file)

fisheye_data_yaml = data_yaml.copy()
fisheye_data_yaml['train'] = fisheye_images_dir
fisheye_data_yaml['val'] = val_fisheye_dst
fisheye_data_yaml['path'] = os.path.dirname(fisheye_images_dir)

with open(new_yaml, 'w') as file:
    yaml.dump(fisheye_data_yaml, file, default_flow_style=False)

print(f'Создание реалистичного fisheye датасета завершено!')
print(f'Тренировочные изображения: {fisheye_images_dir}')
print(f'Валидационные изображения: {val_fisheye_dst}')
print(f'YAML файл конфигурации: {new_yaml}')

# Функция для визуализации результатов аугментации
def visualize_augmentation_examples():
    import matplotlib.pyplot as plt
    import random

    sample_images = random.sample(image_files, 3)
    plt.figure(figsize=(15, 10))

    for i, img_name in enumerate(sample_images):
        orig_path = os.path.join(images_dir, img_name)
        orig_img = cv2.imread(orig_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        fish_path = os.path.join(fisheye_images_dir, img_name)
        fish_img = cv2.imread(fish_path)
        fish_img = cv2.cvtColor(fish_img, cv2.COLOR_BGR2RGB)

        plt.subplot(3, 2, i*2+1)
        plt.title(f"Оригинал {i+1}")
        plt.imshow(orig_img)
        plt.axis('off')

        plt.subplot(3, 2, i*2+2)
        plt.title(f"Fisheye {i+1}")
        plt.imshow(fish_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Запуск визуализации
visualize_augmentation_examples() Чтоб в визуализации после работы аугментации выходили изображения с ббоксами и там говорилось что сколько объектов было потеряно и тд. Тут думаю не аугментации а просто как функция фишай используется так как аугментация используется во время тренировки. Хотелось бы что была возможность уровень искажений увеличить или уменьшить, добавить или не добавлять виньетирование или хроматичность
