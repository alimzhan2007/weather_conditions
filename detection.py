import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class VisibilityEstimatorSVM:
    def __init__(self):
        self.svm_classifier = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
    
    def extract_features(self, image_path):
        """Извлечение нормализованной гистограммы градаций серого."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_resized = cv2.resize(img, (640, 640))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_normalized = hist / (hist.sum() + 1e-7)
        
        return hist_normalized

    def train_model(self, image_paths, labels):
        """Обучение модели на путях к изображениям."""
        X = []
        y = []
        
        print("Извлечение признаков для обучения...")
        for path, label in zip(image_paths, labels):
            features = self.extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)
                
        self.svm_classifier.fit(X, y)
        print("Обучение SVM успешно завершено.")

    def test_model(self, image_paths, labels):
        """Проверка точности модели на тестовых данных."""
        X_test = []
        y_true = []
        
        for path, label in zip(image_paths, labels):
            features = self.extract_features(path)
            if features is not None:
                X_test.append(features)
                y_true.append(label)
        
        y_pred = self.svm_classifier.predict(X_test)
        
        print("\n-- Точность модели --")
        print(f"Общая точность (Accuracy): {accuracy_score(y_true, y_pred):.2f}")
        print("\nДетализация по классам:")
        print(classification_report(y_true, y_pred, zero_division=0))

    def estimate_visibility(self, image_path):
        """Оценка видимости для одиночного нового кадра."""
        features = self.extract_features(image_path)
        if features is None:
            return "Ошибка загрузки изображения"
            
        omega = self.svm_classifier.predict([features])[0]
        v3 = 100 * (omega / 6.0)
        
        if v3 >= 75:
            return f"Оценка V3: {v3:.1f}% - Уровень L4: БЕЗОПАСНАЯ В ИДИМОСТЬ"
        elif 25 <= v3 < 75:
            return f"Оценка V3: {v3:.1f}% - Уровень L2/L3: ОПАСНАЯ (Рекомендуется снизить скорость)"
        else:
            return f"Оценка V3: {v3:.1f}% - Уровень L1: КРИТИЧЕСКАЯ (Опасность!)"


def load_dataset_from_folders(base_dir):
    """Функция для сканирования переименованных папок (1-6)"""
    image_paths = []
    labels = []
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(folder_path) and folder_name.isdigit():
            label = int(folder_name)
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(label)
                    
    return np.array(image_paths), np.array(labels)



if __name__ == "__main__":

    DATASET_DIR = "auto_labeled_dataset" 
    
    if not os.path.exists(DATASET_DIR):
        print(f"Ошибка: Папка {DATASET_DIR} не найдена! Сначала запусти скрипт кластеризации.")
    else:
        paths, labels = load_dataset_from_folders(DATASET_DIR)
        print(f"Загружено файлов: {len(paths)} из {len(np.unique(labels))} классов.")
        
        if len(paths) == 0:
            print("Ошибка: Не найдено папок с числовыми именами (1-6) или они пустые. Проверь имена папок!")
        else:
            X_train_p, X_test_p, y_train, y_test = train_test_split(
                paths, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            estimator = VisibilityEstimatorSVM()
            estimator.train_model(X_train_p, y_train)
            
            estimator.test_model(X_test_p, y_test)
            
