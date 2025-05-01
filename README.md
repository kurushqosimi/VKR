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
