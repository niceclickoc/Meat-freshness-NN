import cv2
import numpy as np
import threading
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QComboBox, QMessageBox, QGroupBox, QRadioButton, QButtonGroup, QCheckBox
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
        
        # Detection Toggle
        self.detection_checkbox = QCheckBox("Использовать детекцию")
        self.detection_checkbox.setChecked(self.detector_enabled)
        self.detection_checkbox.toggled.connect(self.toggle_detection)
        controls_layout.addWidget(self.detection_checkbox)
        
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

    def toggle_detection(self, checked):
        self.detector_enabled = checked
        if checked:
            # Re-initialize if needed (though we keep it loaded in memory usually)
            from src.utils.object_detector import is_initialized, initialize_detector
            if not is_initialized():
                 try:
                     initialize_detector()
                 except Exception as e:
                     self.detection_checkbox.setChecked(False) # Revert
                     QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить детектор: {e}")
        self.current_bboxes = [] # Clear visualization if disabled

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
                else:
                    self.current_bboxes = [] # Clear if disabled
                
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
        self.sequence_results = []  # Store results for aggregation
        self.is_capturing_sequence = True
        self.waiting_for_rotation = False
        
        self.start_button.setEnabled(False)
        self.log(f"Начало серии из {self.frames_to_capture} кадров.")
        self.log("Пожалуйста, вращайте мясо перед камерой...")
        
        # Start the control loop
        QtCore.QTimer.singleShot(500, self.control_loop)

    def control_loop(self):
        """Main control loop for sequence capture."""
        if not self.is_capturing_sequence:
            return

        if self.frames_captured >= self.frames_to_capture:
            self.finish_sequence()
            return
            
        if self.waiting_for_rotation:
             # Just wait, timer will call back
             return

        # Check for frame availability
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QtCore.QTimer.singleShot(50, self.control_loop)
            return

        # Check for meat detection
        raw_frame = self.current_frame.copy()
        meat_detected = False
        
        if self.detector_enabled:
             # Fast check using cached bboxes from update_frame
             # Or better: run detection here on the snapshotted frame to be sure
             # Let's run detection fresh on the raw_frame to be accurate
             from src.utils.object_detector import detect_meat
             rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
             bboxes = detect_meat(rgb_frame, conf_threshold=0.6) # Higher threshold for capture
             if bboxes:
                 meat_detected = True
        else:
             meat_detected = True # If no detector, assume meat is there
             
        if meat_detected:
            # CAPTURE!
            self.frames_captured += 1
            self.log(f"📸 Снимок {self.frames_captured}/{self.frames_to_capture} сделан!")
            
            # Process Frame
            try:
                result = self.predict_frame(raw_frame)
                self.sequence_results.append(result)
            except Exception as e:
                self.log(f"Ошибка обработки: {e}")
            
            # Trigger Rotation Pause
            if self.frames_captured < self.frames_to_capture:
                self.log("⏳ Следующий кадр...")
                # No long pause requested by user. Short pause to allow UI update?
                # Using 500ms so it's not INSTANT machine gun fire, giving slightly chance to rotate if needed
                # But user said "remove 2 sec pause".
                QtCore.QTimer.singleShot(200, self.control_loop)
            else:
                # Last frame, finish immediately
                self.control_loop()
        else:
            # Meat not found, keep looking
            # Optional: Log warning periodically?
            QtCore.QTimer.singleShot(100, self.control_loop)

    def resume_after_rotation(self):
        self.waiting_for_rotation = False
        self.control_loop()

    def predict_frame(self, image):
        """Run prediction on a single frame and return probs."""
        
        # 0. Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 0.5 Detect and crop
        if self.detector_enabled:
            from src.utils.object_detector import detect_meat, get_largest_bbox, crop_to_bbox
            bboxes = detect_meat(image_rgb, conf_threshold=0.5)
            if bboxes:
                largest_bbox = get_largest_bbox(bboxes)
                image_rgb = crop_to_bbox(image_rgb, largest_bbox, padding=10)
                image = crop_to_bbox(image, largest_bbox, padding=10)
            else:
                 # Should have been caught by control_loop, but just in case
                 pass

        # 1. Preprocess & Predict
        target_size_chromatic = (256, 256)
        target_size_hog = (128, 128)
        target_size_depth = (256, 256)
        
        # Chromatic
        chromatic_image = cv2.resize(image_rgb, target_size_chromatic)
        chromatic_image = preprocess_chromatic(chromatic_image)
        chromatic_pred_probs = self.chromatic_model.predict(np.expand_dims(chromatic_image, axis=0))[0]
        
        # HOG
        hog_image = preprocess_hog(image) # BGR
        hog_image = np.expand_dims(hog_image, axis=0) 
        hog_pred_probs = self.hog_model.predict(hog_image)[0]
        
        # Depth
        depth_image = cv2.resize(image_rgb, target_size_depth)
        depth_image = preprocess_depth_map(depth_image)
        depth_pred_probs = self.depth_map_model.predict(np.expand_dims(depth_image, axis=0))[0]
        
        return {
            'chromatic_probs': chromatic_pred_probs,
            'hog_probs': hog_pred_probs,
            'depth_probs': depth_pred_probs
        }

    def finish_sequence(self):
        self.is_capturing_sequence = False
        self.start_button.setEnabled(True)
        self.log("✅ Серия снимков завершена. Анализ...")
        
        # Aggregate Results
        if not self.sequence_results:
             self.log("Нет результатов для анализа.")
             return

        # Aggregation Strategy: Average Probabilities
        avg_chromatic = np.mean([r['chromatic_probs'] for r in self.sequence_results], axis=0)
        avg_hog = np.mean([r['hog_probs'] for r in self.sequence_results], axis=0)
        avg_depth = np.mean([r['depth_probs'] for r in self.sequence_results], axis=0)
        
        # Log aggregated probs
        # Assuming index 2 is spoiled
        self.log(f"Agg Chromatic Spoiled: {avg_chromatic[2]:.2f}")
        self.log(f"Agg HOG Spoiled: {avg_hog[2]:.2f}")
        self.log(f"Agg Depth Spoiled: {avg_depth[2]:.2f}")
        
        # Evaluate with Committee
        verdict, prob = self.committee.evaluate(avg_chromatic[2], avg_hog[2], avg_depth[2])
        
        self.log(f"🏁 ИТОГОВЫЙ ВЕРДИКТ: {verdict.upper()} (p={prob:.2f})")
        
        color = "green" if verdict == "Ok" else "red"
        if verdict == "Ok":
             msg = "Мясо СВЕЖЕЕ! ✅"
        elif verdict == "Defect Confirmed":
             msg = "Мясо ИСПОРЧЕНО! ❌"
        else:
             msg = "ТРЕБУЕТСЯ ЭКСПЕРТ ⚠️"
             color = "orange"
        
        QMessageBox.information(self, "Результат", msg)
        
        if verdict in ["Human needed", "Human Expert Needed"]:
             # For expert review, we should probably show the frame with the highest spoiled probability
             # Find worst frame
             spoiled_probs = [r['chromatic_probs'][2] + r['hog_probs'][2] + r['depth_probs'][2] for r in self.sequence_results]
             worst_idx = np.argmax(spoiled_probs)
             
             # Re-capture isn't possible, we didn't save images. Use last frame? 
             # Or we should have saved them. For now, use current frame or skip image.
             # Ideally we should store the worst image.
             pass

    def log(self, message):
        self.status_log.append(message)
        # Ensure log scrapes to bottom
        sb = self.status_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

