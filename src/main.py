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
from src.utils.expert_interface import expert_interface
from src.utils.user_interface import main as ui_main
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

# Тестовое изображение
test_image_path, selected_class = ui_main()

# Параметры множественных проходов
MAX_PASSES = 5  # Максимальное количество попыток
SIMILARITY_THRESHOLD = 1  # Требуется полное совпадение предсказаний


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


# Функция для загрузки данных и получения предсказаний от модели с множественными проходами
def load_and_predict_multiple_passes(chromatic_model, hog_model, depth_map_model, data_dir, num_samples=100, target_size_chromatic=(256, 256), target_size_hog=(128, 128), target_size_depth=(256, 256)):
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
    try:
        random_samples = random.sample(all_images, num_samples)
    except ValueError as e:
        print(f"Ошибка при выборке случайных изображений: {e}")
        random_samples = all_images  # Если меньше, чем num_samples, взять все

    # Тестовое изображение
    if test_image_path and selected_class:
        random_samples.append((test_image_path, selected_class))
        print(f"Добавлено тестовое изображение: {test_image_path} с меткой {selected_class}")

    for img_path, class_name in random_samples:
        print(f"Обрабатывается изображение: {img_path} с меткой {class_name}")
        pass_count = 0
        agreed = False
        final_chromatic_pred = None
        final_hog_pred = None
        final_depth_pred = None
        final_chromatic_probs = None
        final_hog_probs = None
        final_depth_probs = None

        while pass_count < MAX_PASSES and not agreed:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError("Не удалось загрузить изображение.")

                # Предобработка для каждой модели
                chromatic_image = cv2.resize(image, target_size_chromatic)
                chromatic_image = preprocess_chromatic(chromatic_image)

                hog_image = preprocess_hog(image)
                # HOG возвращает одномерный массив, добавляем размерность для модели
                hog_image = np.expand_dims(hog_image, axis=0)

                depth_image = cv2.resize(image, target_size_depth)
                depth_image = preprocess_depth_map(depth_image)

                # Получение предсказаний от моделей
                chromatic_pred_probs = chromatic_model.predict(np.expand_dims(chromatic_image, axis=0))[0]
                hog_pred_probs = hog_model.predict(hog_image)[0]
                depth_pred_probs = depth_map_model.predict(np.expand_dims(depth_image, axis=0))[0]

                # Преобразование предсказаний в классы
                chromatic_class = np.argmax(chromatic_pred_probs)
                hog_class = np.argmax(hog_pred_probs)
                depth_class = np.argmax(depth_pred_probs)

                # Сохраняем последние предсказания
                final_chromatic_pred = chromatic_class
                final_hog_pred = hog_class
                final_depth_pred = depth_class
                final_chromatic_probs = chromatic_pred_probs
                final_hog_probs = hog_pred_probs
                final_depth_probs = depth_pred_probs

                # Проверка согласованности предсказаний
                if chromatic_class == hog_class == depth_class:
                    agreed = True
                else:
                    pass_count += 1

            except Exception as e:
                print(f"Ошибка при обработке файла {img_path}: {e}")
                break

        # Добавляем изображение в результаты, даже если согласованность не достигнута
        if final_chromatic_pred is not None and final_hog_pred is not None and final_depth_pred is not None:
            data.append({
                'chromatic_pred': final_chromatic_pred,
                'chromatic_probs': final_chromatic_probs,
                'hog_pred': final_hog_pred,
                'hog_probs': final_hog_probs,
                'depth_pred': final_depth_pred,
                'depth_probs': final_depth_probs
            })
            labels.append(class_name)
            file_paths.append(img_path)
        else:
            print(f"Изображение {img_path} не было добавлено из-за отсутствия предсказаний.")

    return data, labels, file_paths


# Получение предсказаний с множественными проходами
data_predictions, labels, file_paths = load_and_predict_multiple_passes(
    chromatic_model, hog_model, depth_map_model, test_dir, num_samples=100,
    target_size_chromatic=(256, 256), target_size_hog=(128, 128), target_size_depth=(256, 256)
)

# Преобразование предсказаний в метки классов
chromatic_preds_classes = np.array([d['chromatic_pred'] for d in data_predictions])
hog_preds_classes = np.array([d['hog_pred'] for d in data_predictions])
depth_map_preds_classes = np.array([d['depth_pred'] for d in data_predictions])

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


# Список для отслеживания изменений предсказаний
expert_corrections = []

# Сохранение оригинальных предсказаний перед вмешательством эксперта
original_preds = final_preds.copy()

# Колбэк для обновления предсказаний от эксперта
def update_prediction(i, new_prediction):
    if final_preds[i] != new_prediction:
        expert_corrections.append(i)
        final_preds[i] = new_prediction


# Инициализация комитета с весами агентов и коэффициентами значимости
committee = ConsensusCommittee(
    weights=[0.3, 0.4, 0.3],  # Веса для моделей хроматического анализа, HOG и карт глубины
    agent_coeffs=[1.0, 1.0, 1.0]  # Коэффициенты значимости для каждого агента
)

# Агент консенсуса: использование для принятия решений
for i in range(len(labels)):
    print(f"Хроматическое предсказание: {chromatic_preds_classes[i]}, HOG предсказание: {hog_preds_classes[i]}, Depth Map предсказание: {depth_map_preds_classes[i]}")
    decoded_label = label_encoder.inverse_transform([final_preds[i]])[0]

    if decoded_label != "Spoiled":
        chromatic_half_fresh_prob = data_predictions[i]['chromatic_probs'][1]  # Вероятность для Half-Fresh
        chromatic_spoiled_prob = data_predictions[i]['chromatic_probs'][2]  # Вероятность для Spoiled

        hog_half_fresh_prob = data_predictions[i]['hog_probs'][1]  # Вероятность для Half-Fresh
        hog_spoiled_prob = data_predictions[i]['hog_probs'][2]  # Вероятность для Spoiled

        depth_map_half_fresh_prob = data_predictions[i]['depth_probs'][1]  # Вероятность для Half-Fresh
        depth_map_spoiled_prob = data_predictions[i]['depth_probs'][2]  # Вероятность для Spoiled

        chromatic_sum = chromatic_half_fresh_prob + chromatic_spoiled_prob
        hog_sum = hog_half_fresh_prob + hog_spoiled_prob
        depth_map_sum = depth_map_half_fresh_prob + depth_map_spoiled_prob

        chromatic_sum = np.array([[chromatic_sum]])
        hog_sum = np.array([[hog_sum]])
        depth_map_sum = np.array([[depth_map_sum]])

        result, final_prob = committee.evaluate(
            chromatic_sum,
            hog_sum,
            depth_map_sum,
            pred_prob=None  # Сюда Ppred, когда будет
        )

        if result != "No Defect":
            # print(f"Файл: {file_paths[i]}, Итог: {result}, Вероятность: {final_prob}")
            if result == "Ok":
                print(f"Файл: {file_paths[i]}, Итог: {result}, Вероятность: {final_prob}")
            if result == "Human needed":
                expert_interface(file_paths[i], short_file_paths[i], decoded_label, lambda new_pred: update_prediction(i, new_pred))
            if result == "Defect Confirmed":
                print("УПАЛО В SPOILED")
                update_prediction(i, 2)

# Оценка точности до вмешательства эксперта
accuracy_before_expert = accuracy_score(labels_encoded, original_preds)
print(f'\nТочность на тестовых данных без учета эксперта: {accuracy_before_expert * 100:.2f}%\n')
print(classification_report(labels_encoded, original_preds, target_names=label_encoder.classes_))

# Сколько раз эксперт изменил предсказание
num_corrections = len(expert_corrections)
print(f"Количество изменений от эксперта: {num_corrections}")

# Ошибки до вмешательства эксперта (на основе точности до вмешательства)
errors_before_expert = (1 - accuracy_before_expert) * len(labels_encoded)

# Пересчитываем точность, учитывая, что исправления эксперта указывают на ошибки моделей
errors_after_expert = errors_before_expert + num_corrections
correct_predictions_after_expert = len(labels_encoded) - errors_after_expert
accuracy_with_expert = correct_predictions_after_expert / len(labels_encoded)

print(f'\nТочность на тестовых данных после внедрения эксперта: {accuracy_with_expert * 100:.2f}%\n')


# Вывод результатов в таблице
print("="*121)
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
spoiled_meat_list = [file_paths[i] for i, pred in enumerate(final_preds) if pred == spoiled_label]

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
