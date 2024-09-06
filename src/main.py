import os
import zipfile
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from keras import models

from src import config

matplotlib.use('Agg')  # Используйте back-end 'Agg' для совместимости

# Пути к данным и модели
dataset_path = '../meat-freshness-image-dataset.zip'
extract_path = '../meat_freshness_dataset'
val_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'valid')

# Распаковка архива (если необходимо)
if not os.path.exists(extract_path):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Архив успешно распакован.")

# Создание генератора данных для тестирования
val_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Загрузка модели
model = tf.keras.models.load_model('./models/meat_freshness_model.h5')

# Проверка точности на тестовых данных
test_loss, test_acc = model.evaluate(test_generator)
print(f'Точность на тестовых данных: {test_acc * 100:.2f}%')


# Функция для отображения предсказаний
def plot_predictions(images, labels, preds):
    plt.figure(figsize=(15, 10))
    num_images = min(len(images), 16)  # Ограничим количество изображений до 16
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f'Факт: {labels[i]}\nПредсказание: {preds[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./test/predictions.png')  # Сохраните изображение в файл


# Получение всех изображений из тестового генератора
all_images, all_labels, all_preds = [], [], []
class_indices = {v: k for k, v in test_generator.class_indices.items()}

for i in range(len(test_generator)):
    x, y = test_generator[i]
    pred = model.predict(x)
    for j in range(len(x)):
        all_images.append(x[j])
        all_labels.append(class_indices[np.argmax(y[j])])
        all_preds.append(class_indices[np.argmax(pred[j])])

# Случайным образом выбираем подмножество изображений для отображения
indices = random.sample(range(len(all_images)), 16)
images = [all_images[i] for i in indices]
labels = [all_labels[i] for i in indices]
preds = [all_preds[i] for i in indices]

# Отображение предсказаний
plot_predictions(images, labels, preds)

# Создание DataFrame для результатов
results = pd.DataFrame({
    'Файл': [test_generator.filenames[i] for i in indices],
    'Фактический': labels,
    'Предсказанный': preds
})

# Сохранение результатов в CSV файл
results.to_csv('./test/prediction_results.csv', index=False)

# Отображение первых 20 результатов
print(results.head(20))
