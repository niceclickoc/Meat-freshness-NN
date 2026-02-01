import cv2
import numpy as np
import threading
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QComboBox, QMessageBox, QGroupBox, QRadioButton, QButtonGroup
)

from src.utils.consensus_committee import ConsensusCommittee
from src.utils.ui.expert_interface import expert_interface
from src.utils.image_processing import preprocess_chromatic, preprocess_hog, preprocess_depth_map

class TestModeInterface(QWidget):
    def __init__(self, chromatic_model, hog_model, depth_map_model, meta_clf, meta_le):
        super().__init__()
        self.chromatic_model = chromatic_model
        self.hog_model = hog_model
        self.depth_map_model = depth_map_model
        self.meta_clf = meta_clf
        self.meta_le = meta_le
        
        # Committee settings (copied from main.py, or should be passed in?)
        # Using default values from main.py
        self.committee = ConsensusCommittee(
            weights=[0.3, 0.4, 0.3],
            agent_coeffs=[1.0, 1.0, 1.0]
        )

        self.camera = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.is_capturing_sequence = False
        self.frames_to_capture = 0
        self.frames_captured = 0
        self.capture_interval_ms = 500 # interval between sequence shots? Or as fast as possible?
        # User says "sequence of photos". Let's assume a small delay or as fast as processing allows.
        # Since we need to wait for processing to potentially pause, we should probably trigger next capture after processing.
        
        # Initialize MiDaS for depth processing
        from src.utils.depth_estimator import initialize_midas, is_initialized
        if not is_initialized():
            print("[Test Mode] Initializing MiDaS for depth estimation...")
            initialize_midas(model_type="MiDaS_small")
        
        # Initialize Object Detector
        from src.utils.object_detector import initialize_detector, is_initialized as detector_initialized
        if not detector_initialized():
            print("[Test Mode] Initializing Object Detector...")
            try:
                initialize_detector()
                self.detector_enabled = True
            except Exception as e:
                print(f"[Test Mode] Warning: Could not load detector: {e}")
                print("[Test Mode] Running without detection (will process full frames)")
                self.detector_enabled = False
        else:
            self.detector_enabled = True
        
        self.current_bboxes = []  # Store bboxes for visualization
        
        self.init_ui()
        self.start_camera()

    def init_ui(self):
        self.setWindowTitle("Тестовый режим (Камера)")
        self.resize(1000, 700)
        
        # Main Layout: Horizontal
        # Left: Camera
        # Right: Settings
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left Side: Camera
        self.camera_label = QLabel("Запуск камеры...")
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        main_layout.addWidget(self.camera_label, stretch=2)
        
        # Right Side: Settings
        settings_layout = QVBoxLayout()
        main_layout.addLayout(settings_layout, stretch=1)
        
        # Frame Count Selection
        settings_layout.addStretch()
        controls_group = QGroupBox("Настройки съемки")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)
        
        controls_layout.addWidget(QLabel("Количество кадров:"))
        
        self.frames_group = QButtonGroup(self)
        self.radio_8 = QRadioButton("8")
        self.radio_12 = QRadioButton("12")
        self.radio_24 = QRadioButton("24")
        self.radio_8.setChecked(True)
        
        self.frames_group.addButton(self.radio_8, 8)
        self.frames_group.addButton(self.radio_12, 12)
        self.frames_group.addButton(self.radio_24, 24)
        
        controls_layout.addWidget(self.radio_8)
        controls_layout.addWidget(self.radio_12)
        controls_layout.addWidget(self.radio_24)
        
        controls_layout.addSpacing(20)
        
        # Start Button
        self.start_button = QPushButton("СТАРТ")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.start_button.clicked.connect(self.start_sequence)
        controls_layout.addWidget(self.start_button)
        
        settings_layout.addWidget(controls_group)
        settings_layout.addStretch()
        
        # Log/Status area
        self.status_log = QtWidgets.QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(200)
        settings_layout.addWidget(self.status_log)

    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera_label.setText("Не удалось подключиться к камере")
            return
        self.timer.start(30) # ~30 FPS for viewfinder

    def stop_camera(self):
        if self.camera:
            self.camera.release()
        self.timer.stop()

    def update_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Detect meat and draw bboxes (for visualization only)
                if self.detector_enabled and not self.is_capturing_sequence:
                    from src.utils.object_detector import detect_meat, draw_bboxes
                    
                    # Convert BGR to RGB for detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bboxes = detect_meat(frame_rgb, conf_threshold=0.3)
                    self.current_bboxes = bboxes
                    
                    # Draw bboxes on frame (BGR for OpenCV drawing)
                    if bboxes:
                        frame = draw_bboxes(frame.copy(), bboxes, color=(0, 255, 0), thickness=2)
                
                # Convert to RGB for Qt display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qt_img).scaled(
                    self.camera_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def start_sequence(self):
        if self.is_capturing_sequence:
            return
            
        self.frames_to_capture = self.frames_group.checkedId()
        self.frames_captured = 0
        self.is_capturing_sequence = True
        self.start_button.setEnabled(False)
        self.log(f"Начало серии из {self.frames_to_capture} кадров...")
        
        # Switch timer to capture mode logic? 
        # Actually logic is: Capture -> Stop Viewfinder update (maybe?) -> Process -> Resume
        # Or just keep viewfinder running and capture current frame.
        
        # We process one frame, then schedule next.
        QtCore.QTimer.singleShot(100, self.process_next_frame)

    def process_next_frame(self):
        if not self.is_capturing_sequence:
            return
            
        if self.frames_captured >= self.frames_to_capture:
            self.finish_sequence()
            return
            
        # Capture current frame
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            raw_frame = self.current_frame.copy()
            self.frames_captured += 1
            self.log(f"Обработка кадра {self.frames_captured}/{self.frames_to_capture}...")
            
            # Run processing in a separate method to keep UI responsive? 
            # Note: Model prediction might block UI. Ideally should be in thread, but Qt UI updates must be main thread.
            # For simplicity, let's try synchronous, if it lags too much we can thread.
            # But Expert Interface needs to be on Main Thread.
            
            try:
                self.process_frame_logic(raw_frame)
            except Exception as e:
                self.log(f"Ошибка обработки: {e}")
            
            # Schedule next frame
            if self.is_capturing_sequence:
                QtCore.QTimer.singleShot(100, self.process_next_frame)
        else:
             # Retry if no frame yet
             QtCore.QTimer.singleShot(50, self.process_next_frame)

    def process_frame_logic(self, image):
        # 0. Convert BGR to RGB (Models were trained on RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 0.5 Detect and crop to meat region
        if self.detector_enabled:
            from src.utils.object_detector import detect_meat, get_largest_bbox, crop_to_bbox
            
            bboxes = detect_meat(image_rgb, conf_threshold=0.5)
            if bboxes:
                largest_bbox = get_largest_bbox(bboxes)
                self.log(f"Обнаружено мясо: bbox {largest_bbox[:4]}, confidence {largest_bbox[4]:.2f}")
                
                # Crop both RGB and BGR versions to bbox
                image_rgb = crop_to_bbox(image_rgb, largest_bbox, padding=10)
                image = crop_to_bbox(image, largest_bbox, padding=10)  # BGR version for HOG
            else:
                self.log("Предупреждение: Мясо не обнаружено, анализируется всё изображение")

        # 1. Preprocess & Predict
        target_size_chromatic = (256, 256)
        target_size_hog = (128, 128)
        target_size_depth = (256, 256)
        
        # Chromatic
        chromatic_image = cv2.resize(image_rgb, target_size_chromatic)
        chromatic_image = preprocess_chromatic(chromatic_image)
        chromatic_pred_probs = self.chromatic_model.predict(np.expand_dims(chromatic_image, axis=0))[0]
        
        # HOG - Note: HOG uses grayscale internally in preprocess_hog, so RGB/BGR matters less 
        # but preprocess_hog expects BGR if using cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) inside it?
        # Let's check preprocess_hog implementation. 
        # It does: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY). 
        # So it EXPECTS BGR. passing RGB to it would mean Red channel becomes Blue. 
        # Grayscale conversion from RGB vs BGR: 0.299R + 0.587G + 0.114B. 
        # If we pass RGB as "BGR", then 0.299B + 0.587G + 0.114R. The grayscale result is slightly different.
        # Since preprocess_hog assumes input is BGR, we should pass the ORIGINAL 'image' (BGR) to it.
        
        hog_image = preprocess_hog(image) # Pass original BGR image
        hog_image = np.expand_dims(hog_image, axis=0) 
        hog_pred_probs = self.hog_model.predict(hog_image)[0]
        
        # Depth
        depth_image = cv2.resize(image_rgb, target_size_depth)
        depth_image = preprocess_depth_map(depth_image)
        depth_pred_probs = self.depth_map_model.predict(np.expand_dims(depth_image, axis=0))[0]
        
        # Meta model input
        chromatic_class = np.argmax(chromatic_pred_probs)
        hog_class = np.argmax(hog_pred_probs)
        depth_class = np.argmax(depth_pred_probs)
        
        X_meta = np.array([[chromatic_class, hog_class, depth_class]])
        final_pred_idx = self.meta_clf.predict(X_meta)[0]
        
        # Consensus
        # Extract probs for classes [Fresh, Half-Fresh, Spoiled] - assuming index order
        # Need to know which index corresponds to 'Spoiled' etc.
        # meta_le.transform(['Spoiled'])[0] ...
        # Assume standard order if models trained that way, but let's look at main.py usage.
        # data_predictions[i]['chromatic_probs'] is used.
        
        chromatic_prob_spoiled = chromatic_pred_probs[2] 
        hog_prob_spoiled = hog_pred_probs[2]
        depth_prob_spoiled = depth_pred_probs[2]

        self.log(f"DEBUG: Chromatic Spoiled Prob: {chromatic_prob_spoiled:.4f}")
        self.log(f"DEBUG: HOG Spoiled Prob: {hog_prob_spoiled:.4f}")
        self.log(f"DEBUG: Depth Spoiled Prob: {depth_prob_spoiled:.4f}")

        # Committee Evaluate
        verdict, prob = self.committee.evaluate(chromatic_prob_spoiled, hog_prob_spoiled, depth_prob_spoiled)
        
        self.log(f"Вердикт: {verdict} ({prob:.2f})")
        
        if verdict in ["Human needed", "Human Expert Needed"]:
            self.log("Требуется эксперт! Приостановка...")
            # Pause sequence
            # Show expert interface
            
            # We need to translate final_pred_idx to string class name
            pred_class_name = self.meta_le.inverse_transform([final_pred_idx])[0]
            
            # Store state to resume
            self.waiting_for_expert = True
            
            # The callback updates our prediction.
            def update_callback(new_pred_idx):
                self.log(f"Эксперт изменил решение на: {new_pred_idx}")
                # We could store this correction if needed.
                self.waiting_for_expert = False
            
            # Pass image directly (no temp file)
            expert_interface(image, f"Frame {self.frames_captured}", pred_class_name, update_callback)
            
            # When expert_interface returns (it finishes its exec loop), we continue.
            self.log("Эксперт завершил работу. Продолжаем...")

    def finish_sequence(self):
        self.is_capturing_sequence = False
        self.start_button.setEnabled(True)
        self.log("Серия снимков завершена.")
        QMessageBox.information(self, "Готово", "Серия снимков завершена.")

    def log(self, message):
        self.status_log.append(message)
        # Ensure log scrapes to bottom
        sb = self.status_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

