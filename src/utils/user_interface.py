import sys

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMessageBox
)


class UserInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.selected_class = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Загрузите Ваше тестовое изображение!")
        self.resize(500, 300)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Отступ сверху
        self.main_layout.addSpacing(20)

        # Заголовок
        self.header_label = QLabel("Выберите тестовое изображение или продолжите без него")
        self.header_label.setAlignment(QtCore.Qt.AlignCenter)
        self.header_label.setWordWrap(True)  # Перенос слов
        font = QtGui.QFont()
        font.setPointSize(14)  # Размер шрифта заголовка
        self.header_label.setFont(font)
        self.main_layout.addWidget(self.header_label)

        # Отступ после заголовка
        self.main_layout.addSpacing(20)

        # Метка для отображения изображения
        self.img_label = QLabel()
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label.setVisible(False)  # Скрываем метку изображения до загрузки
        self.main_layout.addWidget(self.img_label)

        # Отступ после изображения
        self.main_layout.addSpacing(20)

        # Текст "Выберите класс"
        self.select_class_label = QLabel("Выберите класс")
        self.select_class_label.setAlignment(QtCore.Qt.AlignCenter)
        font_label = QtGui.QFont()
        font_label.setPointSize(10)  # Размер шрифта
        self.select_class_label.setFont(font_label)
        self.select_class_label.setVisible(False)  # Скрываем до загрузки изображения
        self.main_layout.addWidget(self.select_class_label)

        # Отступ после текста "Выберите класс"
        self.main_layout.addSpacing(10)

        # Выпадающий список для выбора класса
        self.class_combo_box = QComboBox()
        self.class_combo_box.addItems(["Fresh", "Half-Fresh", "Spoiled"])
        self.class_combo_box.setVisible(False)

        # Центрирование списка
        class_combo_layout = QHBoxLayout()
        class_combo_layout.addStretch()
        class_combo_layout.addWidget(self.class_combo_box)
        class_combo_layout.addStretch()
        self.main_layout.addLayout(class_combo_layout)

        # Отступ после выпадающего списка
        self.main_layout.addSpacing(10)

        # Кнопка продолжения (изначально скрыта)
        self.continue_button = QPushButton("Продолжить")
        self.continue_button.clicked.connect(self.continue_process)
        self.continue_button.setFixedWidth(200)
        self.continue_button.setVisible(False)  # Скрываем до загрузки изображения

        # Центрирование кнопки
        continue_button_layout = QHBoxLayout()
        continue_button_layout.addStretch()
        continue_button_layout.addWidget(self.continue_button)
        continue_button_layout.addStretch()
        self.main_layout.addLayout(continue_button_layout)

        # Увеличенный отступ после кнопки "Продолжить"
        self.main_layout.addSpacing(30)

        # Кнопки внизу
        self.buttons_layout = QVBoxLayout()
        self.main_layout.addLayout(self.buttons_layout)

        # Кнопка выбора изображения
        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.clicked.connect(self.load_image)
        self.select_button.setFixedWidth(250)  # Фиксированная ширина кнопки

        # Центрирование кнопки
        select_button_layout = QHBoxLayout()
        select_button_layout.addStretch()
        select_button_layout.addWidget(self.select_button)
        select_button_layout.addStretch()
        self.buttons_layout.addLayout(select_button_layout)

        # Отступ между кнопками
        self.buttons_layout.addSpacing(10)

        # Кнопка продолжения без изображения
        self.skip_button = QPushButton("Продолжить без изображения")
        self.skip_button.clicked.connect(self.skip_image)
        self.skip_button.setFixedWidth(250)

        # Центрирование кнопки
        skip_button_layout = QHBoxLayout()
        skip_button_layout.addStretch()
        skip_button_layout.addWidget(self.skip_button)
        skip_button_layout.addStretch()
        self.buttons_layout.addLayout(skip_button_layout)

        # Растяжение, чтобы кнопки были прижаты к низу
        self.buttons_layout.addStretch()

    # Функция загрузки изображения
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            try:
                pixmap = QtGui.QPixmap(file_name)
                if pixmap.isNull():
                    raise ValueError("Файл не является изображением или поврежден.")

                self.image_path = file_name
                pixmap = pixmap.scaled(400, 400, aspectRatioMode=QtCore.Qt.KeepAspectRatio)

                self.img_label.setPixmap(pixmap)
                self.img_label.setVisible(True)  # Показываем метку изображения

                # Установка минимального размера окна после загрузки изображения
                self.setMinimumSize(500, 700)
                self.adjustSize()  # Подгон размера окна под содержимое

                # Показываем текст "Выберите класс" и выпадающий список
                self.select_class_label.setVisible(True)
                self.class_combo_box.setVisible(True)

                # Показываем кнопку "Продолжить"
                self.continue_button.setVisible(True)

                # Перемещаем кнопки в самый низ окна
                self.main_layout.removeItem(self.buttons_layout)
                self.main_layout.addLayout(self.buttons_layout)

                # Меняем текст кнопки на "Выбрать другое изображение"
                self.select_button.setText("Выбрать другое изображение")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", "Не удалось открыть файл как изображение. Пожалуйста, выберите корректный файл изображения.")

    # Функция на скип изображения
    def skip_image(self):
        self.image_path = None
        self.selected_class = None

        # Очищаем изображение и скрываем его
        self.img_label.clear()
        self.img_label.setVisible(False)

        # Скрываем текст "Выберите класс" и выпадающий список
        self.select_class_label.setVisible(False)
        self.class_combo_box.setVisible(False)

        # Скрываем кнопку "Продолжить"
        self.continue_button.setVisible(False)

        # Возвращаем минимальный размер окна к начальному
        self.setMinimumSize(500, 300)
        self.adjustSize()

        # Закрываем окно
        self.close()

    # Функция на передачу изображения
    def continue_process(self):
        # Получаем выбранный класс
        if self.class_combo_box.isVisible():
            self.selected_class = self.class_combo_box.currentText()
        else:
            self.selected_class = None
        self.close()

    # Получаем данные для передачи в лист
    def get_image_info(self):
        return self.image_path, self.selected_class

# Инициализатор
def main():
    app = QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    app.exec_()
    return ui.get_image_info()
