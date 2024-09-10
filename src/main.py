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
test_distorted_image = './test/distorted_test.jpg' # ДЛЯ ТЕСТА


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

    # Тестовое изображение (ДЛЯ ТЕСТА)
    random_samples.append((test_distorted_image, 'Spoiled'))

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

# Добавление случайного шума к изображению distorted_test.jpg ЕСЛИ НЕОБХОДИМО
# def add_noise_to_image(image):
#     if image.dtype != np.uint8:
#         image = image.astype(np.uint8)
#
#     noise = np.random.normal(loc=0, scale=10, size=image.shape).astype(np.uint8)
#     noisy_image = cv2.add(image, noise)
#     return noisy_image


# Функции предобработки для каждой модели
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


# Функция для динамического взвешивания на основе расхождения агентов
def dynamic_weight_adjustment(chromatic_pred, hog_pred, depth_map_pred):
    disagreement = max(chromatic_pred, hog_pred, depth_map_pred) - min(chromatic_pred, hog_pred, depth_map_pred)
    if disagreement > 0.2:
        return 1.5  # Увеличиваем вес при значительном расхождении
    else:
        return 1.0  # Стандартный вес


# Рассчет финальных предсказаний с учётом расхождения
def calculate_final_prediction_with_disagreement(chromatic_preds, hog_preds, depth_map_preds):
    chromatic_preds_normalized = chromatic_preds.flatten()
    hog_preds_normalized = hog_preds.flatten()
    depth_map_preds_normalized = depth_map_preds.flatten()

    weights_chromatic = [dynamic_weight_adjustment(c, h, d) for c, h, d in zip(chromatic_preds_normalized, hog_preds_normalized, depth_map_preds_normalized)]
    weights_hog = [dynamic_weight_adjustment(h, c, d) for c, h, d in zip(chromatic_preds_normalized, hog_preds_normalized, depth_map_preds_normalized)]
    weights_depth = [dynamic_weight_adjustment(d, c, h) for c, h, d in zip(chromatic_preds_normalized, hog_preds_normalized, depth_map_preds_normalized)]

    final_preds = (np.array(weights_chromatic) * chromatic_preds_normalized +
                   np.array(weights_hog) * hog_preds_normalized +
                   np.array(weights_depth) * depth_map_preds_normalized) / 3

    return final_preds


# Инициализация комитета с весами агентов и коэффициентами значимости
committee = ConsensusCommittee(
    weights=[0.4, 0.3, 0.3],  # Веса для моделей хроматического анализа, HOG и карт глубины
    agent_coeffs=[1.0, 1.0, 1.0]  # Коэффициенты значимости для каждого агента
)


# Загрузка изображения с шумом УБРАТЬ ПОСЛЕ ТЕСТОВ
# test_distorted_image_with_noise = cv2.imread(test_distorted_image)
# test_distorted_image_with_noise = add_noise_to_image(test_distorted_image_with_noise)


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


# Проверка вероятностей для каждой модели (ДЛЯ ТЕСТОВ)
# for i in range(len(labels)):
#     print(f"Файл: {file_paths[i]}")
#     print(f"Chromatic Model Prediction: {chromatic_preds[i]}")
#     print(f"HOG Model Prediction: {hog_preds[i]}")
#     print(f"Depth Map Model Prediction: {depth_map_preds[i]}")
#     print("="*50)


# Агент консенсуса: использование для принятия решений
for i in range(len(labels)):
    # print(f"Хроматическое предсказание: {chromatic_preds[i]}, HOG предсказание: {hog_preds[i]}, Depth Map предсказание: {depth_map_preds[i]}")
    final_preds = calculate_final_prediction_with_disagreement(
        chromatic_preds[i].reshape(1, -1),
        hog_preds[i].reshape(1, -1),
        depth_map_preds[i].reshape(1, -1)
    )
    # print(f"Файл: {file_paths[i]}, Предсказания: {final_preds}")

    if np.all((final_preds >= 0.5) & (final_preds <= 0.8)) or np.std(final_preds) > 0.2:
        result, final_prob = committee.evaluate(
            chromatic_preds[i],
            hog_preds[i],
            depth_map_preds[i],
            pred_prob=None  # Сюда Ppred, когда будет
        )
        print(f"Файл: {file_paths[i]}, Итог: {result}, Вероятность: {final_prob}")
    else:
        # print(f"Файл: {file_paths[i]} - результат ясен, комитет не нужен.")
        pass


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
