import os
import cv2
import numpy as np
import pandas as pd
import random

from openpyxl.styles.builtins import output
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from xgboost import XGBClassifier

from src.utils.consensus_committee import ConsensusCommittee
from src.utils.expert_interface import expert_interface
from src.utils.report import generate_report

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

# Преобразование предсказаний в формат для мета-классификатора
X_meta = np.stack([chromatic_preds_classes, hog_preds_classes, depth_map_preds_classes], axis=1)

# Обучение мета-классификатора с использованием градиентного бустинга
meta_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
meta_clf.fit(X_meta, labels_encoded)

# Ансамблевое предсказание мета-классификатором
final_preds = meta_clf.predict(X_meta)

# Укороченные пути для удобства отображения
short_file_paths = [os.path.basename(path) for path in file_paths]


# Колбэк для обновления предсказаний от эксперта
def update_prediction(new_prediction):
    final_preds[i] = new_prediction


# Инициализация комитета с весами агентов и коэффициентами значимости
committee = ConsensusCommittee(
    weights=[0.4, 0.3, 0.3],  # Веса для моделей хроматического анализа, HOG и карт глубины
    agent_coeffs=[1.0, 1.0, 1.0]  # Коэффициенты значимости для каждого агента
)

# Агент консенсуса: использование для принятия решений
for i in range(len(labels)):
    # print(f"Хроматическое предсказание: {chromatic_preds[i]}, HOG предсказание: {hog_preds[i]}, Depth Map предсказание: {depth_map_preds[i]}")
    decoded_label = label_encoder.inverse_transform([final_preds[i]])[0]

    if decoded_label != "Spoiled":
        chromatic_half_fresh_prob = chromatic_preds[i][1]  # Вероятность для Half-Fresh
        chromatic_spoiled_prob = chromatic_preds[i][2]  # Вероятность для Spoiled

        hog_half_fresh_prob = hog_preds[i][1]  # Вероятность для Half-Fresh
        hog_spoiled_prob = hog_preds[i][2]  # Вероятность для Spoiled

        depth_map_half_fresh_prob = depth_map_preds[i][1]  # Вероятность для Half-Fresh
        depth_map_spoiled_prob = depth_map_preds[i][2]  # Вероятность для Spoiled

        chromatic_sum = chromatic_half_fresh_prob + chromatic_spoiled_prob
        hog_sum = hog_half_fresh_prob + hog_spoiled_prob
        depth_map_sum = depth_map_half_fresh_prob + depth_map_spoiled_prob

        result, final_prob = committee.evaluate(
            chromatic_sum.reshape(1, -1),
            hog_sum.reshape(1, -1),
            depth_map_sum.reshape(1, -1),
            pred_prob=None  # Сюда Ppred, когда будет
        )

        if result != "No Defect":
            # print(f"Файл: {file_paths[i]}, Итог: {result}, Вероятность: {final_prob}")
            if result == "Ok":
                pass
            if result == "Human needed":
                expert_interface(file_paths[i], short_file_paths[i], decoded_label, update_prediction)


# Оценка точности ансамблевой модели
accuracy = accuracy_score(labels_encoded, final_preds)
print(f'Точность на тестовых данных: {accuracy * 100:.2f}%')
print(classification_report(labels_encoded, final_preds, target_names=label_encoder.classes_))


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

# Подсчет количества свежего, полу-свежего и испорченного мяса
fresh_label = label_encoder.transform(['Fresh'])[0]
half_fresh_label = label_encoder.transform(['Half-Fresh'])[0]
spoiled_label = label_encoder.transform(['Spoiled'])[0]

fresh_count = np.sum(final_preds == fresh_label)
half_fresh_count = np.sum(final_preds == half_fresh_label)
spoiled_count = np.sum(final_preds == spoiled_label)
spoiled_meat_list = [file_paths[i] for i, pred in enumerate(final_preds) if pred == 2]

# Переменные с данными для отчета
supplier_number = 1 if not os.path.exists('./results/report.xlsx') else pd.read_excel('./results/report.xlsx').shape[0] + 1
total_meat = len(labels)
fresh_meat = fresh_count
half_fresh_meat = half_fresh_count
spoiled_meat = spoiled_count
spoiled_meat_images = spoiled_meat_list
output_excel = './results/report.xlsx'

# Вызов функции для генерации отчета
try:
    generate_report(supplier_number,
                    total_meat,
                    fresh_meat,
                    half_fresh_meat,
                    spoiled_meat,
                    spoiled_meat_images,
                    output_excel)
except PermissionError:
    print("="*121)
    print("Ошибка записи в отчет! Закройте файл")
