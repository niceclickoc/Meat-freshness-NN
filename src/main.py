import os
import cv2
import numpy as np
import pandas as pd
import random

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from xgboost import XGBClassifier

from src.utils.consensus_committee import ConsensusCommittee

# Пути к моделям
chromatic_model_path = './models/chromatic_model.h5'
hog_model_path = './models/hog_model.h5'
depth_map_model_path = './models/depth_model.h5'

# Загрузка моделей
chromatic_model = load_model(chromatic_model_path)
hog_model = load_model(hog_model_path)
depth_map_model = load_model(depth_map_model_path)

# Пути к тестовым данным
test_dir = '../meat_freshness_dataset/Meat Freshness.v1-new-dataset.multiclass/valid'

# Функция для загрузки данных и получения предсказаний от модели
def load_and_predict(model, preprocess_func, data_dir, num_samples=100, target_size=(224, 224)):
    data = []
    labels = []
    file_paths = []
    class_names = os.listdir(data_dir)
    all_images = []

    # Собираем все изображения из всех классов
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                all_images.append((img_path, class_name))

    # Выбираем случайные изображения
    random_samples = random.sample(all_images, num_samples)

    for img_path, class_name in random_samples:
        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, target_size)
            image = preprocess_func(image)
            data.append(image)
            labels.append(class_name)
            file_paths.append(img_path)
        except Exception as e:
            print(f"Ошибка при обработке файла {img_path}: {e}")

    data = np.array(data)
    predictions = model.predict(data)
    return predictions, labels, file_paths

# Функции предобработки для каждой модели
def preprocess_chromatic(image):
    return image / 255.0

def preprocess_hog(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(image, (128, 128))
    features = hog(resized_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

def preprocess_depth_map(image):
    return image / 255.0


# Инициализация комитета с весами агентов и коэффициентами значимости
committee = ConsensusCommittee(
    weights=[0.4, 0.3, 0.3],  # Веса для моделей хроматического анализа, HOG и карт глубины
    agent_coeffs=[1.0, 1.0, 1.0]  # Коэффициенты значимости для каждого агента
)

# Получение предсказаний от каждой модели
chromatic_preds, labels, file_paths = load_and_predict(chromatic_model, preprocess_chromatic, test_dir, target_size=(256, 256))
hog_preds, _, _ = load_and_predict(hog_model, preprocess_hog, test_dir, target_size=(128, 128))
depth_map_preds, _, _ = load_and_predict(depth_map_model, preprocess_depth_map, test_dir, target_size=(256, 256))

# Преобразование предсказаний в метки классов
chromatic_preds_classes = np.argmax(chromatic_preds, axis=1)
hog_preds_classes = np.argmax(hog_preds, axis=1)
depth_map_preds_classes = np.argmax(depth_map_preds, axis=1)

# Кодирование меток
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# # Проверка вероятностей для каждой модели
# for i in range(len(labels)):
#     print(f"Файл: {file_paths[i]}")
#     print(f"Chromatic Model Prediction: {chromatic_preds[i]}")
#     print(f"HOG Model Prediction: {hog_preds[i]}")
#     print(f"Depth Map Model Prediction: {depth_map_preds[i]}")
#     print("="*50)

# Агент консенсуса: использование для принятия решений
for i in range(len(labels)):
    result, final_prob = committee.evaluate(
        chromatic_preds[i],
        hog_preds[i],
        depth_map_preds[i],
        pred_prob=None  # Можно добавить предсказание от Ppred, если оно имеется
    )
    print(f"Файл: {file_paths[i]}, Итог: {result}, Вероятность: {final_prob}")

# Преобразование предсказаний в формат для мета-классификатора
X_meta = np.stack([chromatic_preds_classes, hog_preds_classes, depth_map_preds_classes], axis=1)

# Обучение мета-классификатора с использованием градиентного бустинга
meta_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
meta_clf.fit(X_meta, labels_encoded)

# Ансамблевое предсказание мета-классификатором
final_preds = meta_clf.predict(X_meta)

# Оценка точности ансамблевой модели
accuracy = accuracy_score(labels_encoded, final_preds)
print(f'Точность на тестовых данных: {accuracy * 100:.2f}%')
print(classification_report(labels_encoded, final_preds, target_names=label_encoder.classes_))

# Укороченные пути для удобства отображения
short_file_paths = [os.path.basename(path) for path in file_paths]

# Вывод результатов в таблице
results_df = pd.DataFrame({
    'Файл': short_file_paths,
    'Метка': label_encoder.inverse_transform(labels_encoded),
    'Chromatic': label_encoder.inverse_transform(chromatic_preds_classes),
    'HOG': label_encoder.inverse_transform(hog_preds_classes),
    'Depth Map': label_encoder.inverse_transform(depth_map_preds_classes),
    'Итоговый вердикт': label_encoder.inverse_transform(final_preds)
})

print(results_df.to_string(index=False))
