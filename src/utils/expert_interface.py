import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget
)

def expert_interface(image_path, file_name, model_prediction, update_prediction_callback):
    class ExpertWindow(QWidget):
        def __init__(self):
            super().__init__()
            self._closed_by_button = False
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("Необходима помощь эксперта")
            self.resize(500, 600)  # Устанавливаем размер окна

            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

            # Изображение
            img_label = QLabel()
            pixmap = QtGui.QPixmap(image_path)
            pixmap = pixmap.scaled(300, 300, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            img_label.setPixmap(pixmap)
            img_label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(img_label)

            # Информация о файле и предсказании
            file_label = QLabel(f"Файл: {file_name}")
            file_label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(file_label)

            prediction_label = QLabel(f"Модели считают: {model_prediction}")
            prediction_label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(prediction_label)

            # Текст инструкции
            instruction_label = QLabel("Подтвердите или опровергните предсказание модели")
            instruction_label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(instruction_label)

            # Кнопки выбора класса
            button_layout = QHBoxLayout()
            fresh_button = QPushButton("Fresh")
            half_fresh_button = QPushButton("Half-Fresh")
            spoiled_button = QPushButton("Spoiled")

            fresh_button.clicked.connect(lambda: self.button_clicked(0))
            half_fresh_button.clicked.connect(lambda: self.button_clicked(1))
            spoiled_button.clicked.connect(lambda: self.button_clicked(2))

            button_layout.addWidget(fresh_button)
            button_layout.addWidget(half_fresh_button)
            button_layout.addWidget(spoiled_button)

            self.layout.addLayout(button_layout)

        def button_clicked(self, new_prediction):
            self._closed_by_button = True
            update_prediction_callback(new_prediction)
            self.close()

        def closeEvent(self, event: QtGui.QCloseEvent):
            if not self._closed_by_button:
                QtWidgets.QApplication.quit()
                sys.exit(0)
            event.accept()

    # Проверяем, есть ли существующий экземпляр QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = ExpertWindow()
    window.show()
    app.exec_()
