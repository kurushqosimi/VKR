Вы учитесь на 4 курсе, и тема вашего ВКР Определение объекта на изображении с широкоугольного объектива. Вы написали вот такую аугментацию import cv2
import numpy as np
import os
import json
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random
from shapely.geometry import Polygon

# Настройки
DATA_DIR = "/content/coco"
IMAGES_DIR = os.path.join(DATA_DIR, "val2017")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations/instances_val2017.json")
OUTPUT_DIR = "/content/fisheye_coco"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Скачиваем COCO 2017 (Val)
!wget http://images.cocodataset.org/zips/val2017.zip -O /content/val2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /content/annotations.zip
!unzip -q /content/val2017.zip -d /content/coco
!unzip -q /content/annotations.zip -d /content/coco

# Инициализация COCO API
coco = COCO(ANNOTATIONS_FILE)

# Функция для fisheye-аугментации (с меньшими коэффициентами)
def apply_fisheye(image, k1=0.15, k2=0.05):  # Уменьшено для более мягкого эффекта
    h, w = image.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            x = (2 * j - w) / w
            y = (2 * i - h) / h
            r = np.sqrt(x**2 + y**2)
            distortion = 1 + k1 * r**2 + k2 * r**4
            map_x[i, j] = (x * distortion + 1) * w / 2
            map_y[i, j] = (y * distortion + 1) * h / 2

    fisheye_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return fisheye_img

# Функция для пересчёта bbox с обрезкой по границам
def apply_fisheye_to_bbox(bbox, img_width, img_height, k1=0.15, k2=0.05):
    x, y, w, h = bbox
    corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    
    distorted_corners = []
    for (px, py) in corners:
        nx = (2 * px - img_width) / img_width
        ny = (2 * py - img_height) / img_height
        r = np.sqrt(nx**2 + ny**2)
        distortion = 1 + k1 * r**2 + k2 * r**4
        dx = (nx * distortion + 1) * img_width / 2
        dy = (ny * distortion + 1) * img_height / 2
        distorted_corners.append([dx, dy])
    
    polygon = Polygon(distorted_corners)
    bounds = polygon.bounds
    new_bbox = [
        max(0, bounds[0]),          # x_min
        max(0, bounds[1]),          # y_min
        min(img_width, bounds[2]) - max(0, bounds[0]),  # width
        min(img_height, bounds[3]) - max(0, bounds[1])  # height
    ]
    return [round(coord, 2) for coord in new_bbox]

# Выбираем случайное изображение
cat_ids = coco.getCatIds(catNms=['person', 'dog', 'car'])
img_ids = coco.getImgIds(catIds=cat_ids)
random_img_id = random.choice(img_ids)
img_info = coco.loadImgs([random_img_id])[0]
img_path = os.path.join(IMAGES_DIR, img_info['file_name'])

# Загружаем данные
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ann_ids = coco.getAnnIds(imgIds=random_img_id, catIds=cat_ids)
annotations = coco.loadAnns(ann_ids)

# Применяем fisheye
fisheye_image = apply_fisheye(image)

# Обрабатываем bbox
print("Сравнение координат bounding boxes (с обрезкой по границам):")
print("{:<30} {:<30}".format("Оригинальные координаты", "Новые координаты"))
print("-"*65)

new_annotations = []
for i, ann in enumerate(annotations):
    original_bbox = [round(coord, 2) for coord in ann['bbox']]
    new_bbox = apply_fisheye_to_bbox(ann['bbox'], img_info['width'], img_info['height'])
    
    # Пропускаем "битые" bbox
    if new_bbox[2] <= 5 or new_bbox[3] <= 5:  # Ширина/высота < 5px
        print(f"BBox {i+1} пропущен (слишком мал после преобразования)")
        continue
    
    ann['bbox'] = new_bbox
    new_annotations.append(ann)
    print(f"BBox {i+1}:")
    print(f"  {original_bbox} → {new_bbox}")
    print("-"*65)

# Функция для рисования bbox с подписями
def draw_boxes(img, annotations, color=(0, 255, 0), thickness=2):
    for i, ann in enumerate(annotations):
        bbox = ann['bbox']
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        # Подпись с координатами
        label = f"{x},{y} {w}x{h}"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

# Визуализация
plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
plt.title(f"Original Image (ID: {random_img_id})")
original_with_boxes = draw_boxes(image.copy(), annotations, color=(255, 0, 0))
plt.imshow(original_with_boxes)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Fisheye Image with Corrected BBoxes")
if new_annotations:
    fisheye_with_boxes = draw_boxes(fisheye_image.copy(), new_annotations, color=(0, 255, 0))
else:
    fisheye_with_boxes = fisheye_image.copy()
    plt.text(100, 100, "No valid bboxes after transform", color='red')
plt.imshow(fisheye_with_boxes)
plt.axis('off')

plt.tight_layout()
plt.show()

# Сохраняем результаты
output_image_path = os.path.join(OUTPUT_DIR, img_info['file_name'])
cv2.imwrite(output_image_path, cv2.cvtColor(fisheye_image, cv2.COLOR_RGB2BGR))

output_annotations = {
    "info": coco.dataset['info'],
    "licenses": coco.dataset['licenses'],
    "images": [img_info],
    "annotations": new_annotations,
    "categories": coco.dataset['categories']
}

with open(os.path.join(OUTPUT_DIR, 'fisheye_annotations.json'), 'w') as f:
    json.dump(output_annotations, f)

print(f"\nРезультаты сохранены в {OUTPUT_DIR}")
print(f"Изображение: {output_image_path}")
print(f"Аннотации: {os.path.join(OUTPUT_DIR, 'fisheye_annotations.json')}") Теперь вам надо провести эксперимент с тремя разницами готовыми датасетами, и тремя разными нейронными сетями. Что вы делаете так это вы обучаете модели с 10%, 20%, 30%, 40%, 50% фишай изображениями, и проверяете насколько модели стали лучше в распознавании объектов в фишай изображениях, и насколько модели стали хуже в распознавании объектов в обычных изображениях. Сколько времени занимает каждый из полученных моделей при обучении. Т.е например взяли датасет коко2017 и взяли нейросеть yolov8n сначала тренировали его обычными изображениями и засекли время, проверили его работу на обычных изображениях, потом на искаженных. Сохранили эти метрики для так называемой yolov8n_original.pt модель тоже сохраняем. Потом сделали то же самое но только при тренировке использовали 10% аугментированных рисунков, то есть с фишай эффектами. Сохранили метрики и саму модель как yolov8n_10.pt. Далее 20%, итак до 50% повторяем тоже самое. Как вторую нейросеть берем модель нейросети faster r cnn. На этом же датасете делаем тоже самое что сделали для yolov8n. То есть будет удобно если мы создадим coco2017_10, coco2017_20 и тд до 50. И обучаем модели нейронных сетей, на этих датасетах и сохраняем метрики и модели. Ну и какую то еще современную нейронную сеть берем как третье. И все это повторяем для двух других датасетов которые вы выберете. Все модели и метрики сохраняем. Чтоб при тренировке с фишай изображениями было разной степени фишай эффекта. Также надо чтоб мы засекли сколько точность падает в зависимости от растояния объекта от центра. Ну как будто бы мерим точность у краев сколько падает у каждой модели. Вот такие требования к вашему ВКР. Вы поняли ваш ВКР? Думаю легче будет если мы разделим на шаги и по шагу пройдемся.
