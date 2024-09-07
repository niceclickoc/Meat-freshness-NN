import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from src import config

# Пути к данным
dataset_path = config.DATASET_PATH
extract_path = config.EXTRACT_PATH
train_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'train')
val_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'valid')

# Проверка наличия локального файла и распаковка архива
if not os.path.exists(dataset_path):
    print(f"Файл архива {dataset_path} не найден!")
else:
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Архив успешно распакован.")
    else:
        print("Архив уже распакован.")

# Проверка существования директорий train и valid
if not os.path.exists(train_dir):
    print(f"Директория {train_dir} не найдена.")
else:
    print(f"Директория {train_dir} существует.")

if not os.path.exists(val_dir):
    print(f"Директория {val_dir} не найдена.")
else:
    print(f"Директория {val_dir} существует.")

# Создание генераторов данных
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=30, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Загрузка предобученной модели DenseNet121
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Добавление новых слоев
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# Создание финальной модели
model = Model(inputs=base_model.input, outputs=predictions)

# Замораживание начальных слоев DenseNet121
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback для остановки обучения при отсутствии улучшений
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stopping]
)

# Сохранение модели после первого этапа обучения
model.save('chromatic_analysis_model_initial.h5')
print("Initial chromatic analysis model saved.")

# Размораживание всех слоев и повторная компиляция модели для тонкой настройки
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Продолжение обучения
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[early_stopping]
)

# Сохранение модели после тонкой настройки
model.save('../models/chromatic_model.h5')
print("Fine-tuned chromatic analysis model saved.")

if os.path.exists('chromatic_analysis_model_initial.h5'):
    os.remove('chromatic_analysis_model_initial.h5')
    print('Модель init удалена')
else:
    print('Ошибка')
    pass

# Проверка точности на тестовых данных
test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Disable shuffling for consistent results
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Точность на тестовых данных: {test_acc * 100:.2f}%')


# Отображение изображений с результатами
def plot_predictions(images, labels, preds):
    plt.figure(figsize=(15, 10))
    num_images = min(len(images), 16)  # Ограничим количество изображений до 16
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f'Actual: {labels[i]}\nPredicted: {preds[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(config.PREDICTION_RESULTS_PNG)  # Сохраните изображение в файл


# Получение изображений и предсказаний
images, labels, preds = [], [], []
class_indices = {v: k for k, v in test_generator.class_indices.items()}

for x, y in test_generator:
    pred = model.predict(x)
    for j in range(len(x)):
        images.append(x[j])
        labels.append(class_indices[np.argmax(y[j])])
        preds.append(class_indices[np.argmax(pred[j])])
    if len(images) >= 16:
        break

# Plot the predictions
plot_predictions(images, labels, preds)

# Создание DataFrame для результатов
results = pd.DataFrame({
    'Filename': test_generator.filenames[:len(images)],
    'Actual': labels,
    'Predicted': preds
})

# Сохранение результатов в CSV файл
results.to_csv(config.PREDICTION_RESULTS_CSV, index=False)

# Отображение первых 20 результатов
print(results.head(20))
