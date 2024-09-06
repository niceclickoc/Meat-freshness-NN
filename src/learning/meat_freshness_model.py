import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from src import config

# Пути к данным
dataset_path = config.DATASET_PATH
extract_path = config.EXTRACT_PATH
train_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'train')
val_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'valid')

# Распаковка архива
if not os.path.exists(extract_path):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Архив успешно распакован.")

# Создание генераторов данных
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Создание и обучение моделей
from tensorflow.keras.applications import DenseNet121, VGG16

# Depth Model
depth_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
depth_model.trainable = False
depth_inputs = tf.keras.Input(shape=(256, 256, 3))
depth_x = depth_model(depth_inputs, training=False)
depth_x = tf.keras.layers.GlobalAveragePooling2D()(depth_x)
depth_outputs = tf.keras.layers.Dense(3, activation='softmax')(depth_x)
depth_model_final = tf.keras.Model(depth_inputs, depth_outputs)

depth_model_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
depth_model_final.fit(train_generator, epochs=10, validation_data=val_generator)
depth_model_final.save('depth_model.h5')

# Chromatic Model
chromatic_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
chromatic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
chromatic_model.fit(train_generator, epochs=10, validation_data=val_generator)
chromatic_model.save('chromatic_model.h5')

# HOG Model
hog_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
hog_model.trainable = False
hog_inputs = tf.keras.Input(shape=(256, 256, 3))
hog_x = hog_model(hog_inputs, training=False)
hog_x = tf.keras.layers.GlobalAveragePooling2D()(hog_x)
hog_outputs = tf.keras.layers.Dense(3, activation='softmax')(hog_x)
hog_model_final = tf.keras.Model(hog_inputs, hog_outputs)
hog_model_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hog_model_final.fit(train_generator, epochs=10, validation_data=val_generator)
hog_model_final.save('hog_model.h5')

# Ансамблевая модель
depth_model = tf.keras.models.load_model('depth_model.h5')
chromatic_model = tf.keras.models.load_model('chromatic_model.h5')
hog_model = tf.keras.models.load_model('hog_model.h5')


def ensemble_predict(images):
    depth_preds = depth_model.predict(images)
    chromatic_preds = chromatic_model.predict(images)
    hog_preds = hog_model.predict(images)

    final_preds = (depth_preds + chromatic_preds + hog_preds) / 3
    return final_preds


# Тестирование ансамблевой модели
x, y = next(val_generator)
pred = ensemble_predict(x)
accuracy = np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
print(f'Точность ансамблевой модели: {accuracy * 100:.2f}%')
