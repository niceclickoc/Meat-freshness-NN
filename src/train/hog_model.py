import os
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow import lite

from src import config

# Пути к данным
dataset_path = config.DATASET_PATH
extract_path = config.EXTRACT_PATH
train_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'train')
val_dir = os.path.join(extract_path, 'Meat Freshness.v1-new-dataset.multiclass', 'valid')


# Функция для извлечения признаков HOG
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(image, (128, 128))  # Увеличение размера изображения
    features = hog(resized_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features


# Функция для загрузки данных
def load_data(data_dir):
    data = []
    labels = []
    class_names = os.listdir(data_dir)
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    features = extract_hog_features(img_path)
                    data.append(features)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Ошибка при обработке файла {img_path}: {e}")
    return np.array(data), np.array(labels)


# Загрузка данных
train_data, train_labels = load_data(train_dir)
val_data, val_labels = load_data(val_dir)

# Кодирование меток
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels_encoded, test_size=0.2, random_state=42)

# Построение модели
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Использование Early Stopping и ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Оценка модели
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Точность на тестовых данных: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Сохранение модели
model.save('../models/hog_model.h5')
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('../models/hog_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("HOG Deep модель сохранена.")

# Оценка модели на валидационных данных
val_pred = model.predict(val_data)
val_pred_classes = np.argmax(val_pred, axis=1)
val_accuracy = accuracy_score(val_labels_encoded, val_pred_classes)
print(f'Точность на валидационных данных: {val_accuracy * 100:.2f}%')
print(classification_report(val_labels_encoded, val_pred_classes, target_names=label_encoder.classes_))
