from tensorflow.keras.models import load_model

# Загрузка моделей
model1 = load_model('../learning/chromatic_analysis_model_finetuned.h5')
model2 = load_model('../learning/chromatic_analysis_model_initial.h5')
model3 = load_model('../models/chromatic_model.h5')

# # Вывод архитектур моделей
print("Архитектура модели 1:")
model1.summary()

# print("\nАрхитектура модели 2:")
# model2.summary()

print("\nАрхитектура модели 3:")
model3.summary()

# Сравнение конфигураций (архитектур)
config1 = model1.get_config()
config2 = model2.get_config()
config3 = model3.get_config()

if config1 == config3:
    print("Архитектуры всех моделей идентичны.")
else:
    print("Архитектуры моделей различаются.")
