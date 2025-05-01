import os
import cv2
import numpy as np
import random
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Пути к предобученным моделям
chromatic_model_path = '../models/chromatic_model.h5'
hog_model_path = '../models/hog_model.h5'
depth_map_model_path = '../models/depth_model.h5'

# Загрузка моделей
chromatic_model = load_model(chromatic_model_path)
hog_model = load_model(hog_model_path)
depth_map_model = load_model(depth_map_model_path)

# Путь к обучающему набору данных
train_dir = '../../meat_freshness_dataset/Meat Freshness.v1-new-dataset.multiclass/train'

def preprocess_chromatic(image):
    return image / 255.0

def preprocess_hog(image):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(image, (128, 128))
    features = hog(resized_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

def preprocess_depth_map(image):
    return image / 255.0

def load_data_for_meta(train_dir, target_size_chromatic=(256,256), target_size_hog=(128,128), target_size_depth=(256,256)):
    class_names = os.listdir(train_dir)
    data = []
    labels = []

    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        # Собираем все изображения данного класса
        images_list = os.listdir(class_path)
        for img_name in images_list:
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Предобработка
            chromatic_image = cv2.resize(image, target_size_chromatic)
            chromatic_image = preprocess_chromatic(chromatic_image)

            hog_image = preprocess_hog(image)
            hog_image = np.expand_dims(hog_image, axis=0)

            depth_image = cv2.resize(image, target_size_depth)
            depth_image = preprocess_depth_map(depth_image)

            # Предсказания от трёх базовых моделей
            chromatic_pred_probs = chromatic_model.predict(np.expand_dims(chromatic_image, axis=0))[0]
            hog_pred_probs = hog_model.predict(hog_image)[0]
            depth_pred_probs = depth_map_model.predict(np.expand_dims(depth_image, axis=0))[0]

            # Выберем самый вероятный класс для каждой модели (или можно использовать сами вероятности)
            chromatic_class = np.argmax(chromatic_pred_probs)
            hog_class = np.argmax(hog_pred_probs)
            depth_class = np.argmax(depth_pred_probs)

            # Сохраняем результаты
            data.append([chromatic_class, hog_class, depth_class])
            labels.append(class_name)

    return np.array(data), labels

# Загружаем данные для мета-классификатора
X_meta, raw_labels = load_data_for_meta(train_dir)
label_encoder = LabelEncoder()
y_meta = label_encoder.fit_transform(raw_labels)

# Обучение мета-классификатора
meta_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
meta_clf.fit(X_meta, y_meta)

# Проверка на тех же данных (только для отладки)
y_pred = meta_clf.predict(X_meta)
acc = accuracy_score(y_meta, y_pred)
print(f"Точность на обучающих данных мета-классификатора: {acc*100:.2f}%")
print(classification_report(y_meta, y_pred, target_names=label_encoder.classes_))

# Сохраняем мета-классификатор и энкодер классов
out_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'meta')
os.makedirs(out_dir, exist_ok=True)
joblib.dump(meta_clf, os.path.join(out_dir, 'meta_clf.joblib'))
joblib.dump(label_encoder, os.path.join(out_dir, 'meta_label_encoder.joblib'))
print("Мета-классификатор и энкодер успешно сохранены.")
