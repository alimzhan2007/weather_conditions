import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil


SOURCE_DIR = "Railway_Dataset"
OUTPUT_DIR = "auto_labeled_dataset"
TARGET_SIZE = (640, 640)

def extract_histogram(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_resized = cv2.resize(img, TARGET_SIZE)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    return hist / (hist.sum() + 1e-7)


file_paths = []
features = []

print("Шаг 1: Сканирование изображений и извлечение признаков...")
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        full_path = os.path.join(SOURCE_DIR, filename)
        hist_features = extract_histogram(full_path)
        
        if hist_features is not None:
            file_paths.append(full_path)
            features.append(hist_features)

X = np.array(features)
print(f"Успешно обработано изображений: {len(X)}")


print("\nШаг 2: Группировка изображений на 6 категорий видимости...")
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)


print("\nШаг 3: Копирование файлов в новые папки...")
for path, label in zip(file_paths, cluster_labels):
    cluster_folder = os.path.join(OUTPUT_DIR, f"cluster_{label}")
    os.makedirs(cluster_folder, exist_ok=True)

    shutil.copy(path, cluster_folder)

print(f"\nГотово! Проверь папку '{OUTPUT_DIR}'. Картинки разделены на 6 кластеров.")