import queue
import threading
import time
import winsound
import cv2 
import numpy as np
import mediapipe as mp
import sys
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from collections import deque


# ==================== DEFINE CNN STRUCTURE (EYE + MOUTH) ====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13 * 6 * 128, 256)  # 9984
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.yawn_state = 'Not Yawn'
        self.left_eye_state = 'Eye open'
        self.right_eye_state = 'Eye open'
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0.0
        self.yawns = 0
        self.yawn_duration = 0.0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        self.yawn_buffer = deque(maxlen=15)  # ~0.5s tại 30fps

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        self.setWindowTitle("Somnolence Detection")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 10px;")
        self.layout.addWidget(self.info_label)

        self.update_info()
        
        # ==================== LOAD MODELS ====================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Eye model
        self.detecteye = CNN().to(self.device)
        self.detecteye.load_state_dict(torch.load(r"runs\custom\eye_state_cnn_ver2.pth", map_location=self.device))
        self.detecteye.eval()

        # Yawn model (CNN, không dùng YOLO)
        self.detectyawn = CNN().to(self.device)
        self.detectyawn.load_state_dict(torch.load(r"runs\custom\mouth_cnn_ver2.pth", map_location=self.device))
        self.detectyawn.eval()

        self.cap = cv2.VideoCapture(0)
        time.sleep(1.0)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()

    def update_info(self):
        if self.yawn_duration > 4.0:
            self.alert_text = "<p style='color: orange; font-weight: bold;'> Alert: Prolonged Yawn Detected!</p>"
        elif self.microsleeps > 3.0:
            self.alert_text = "<p style='color: red; font-weight: bold;'> Alert: Prolonged Microsleep Detected!</p>"
        else:
            self.alert_text = ""

        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333; font-size:18px'>"
            f"<h2 style='text-align: center; color: #4CAF50; font-size: 24px;'>Drowsiness Detector</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"
            f"<p><b>Blinks:</b> {self.blinks}</p>"
            f"<p><b>Microsleeps:</b> {self.microsleeps:.2f} s</p>"
            f"<p><b>Yawns:</b> {self.yawns}</p>"
            f"<p><b>Yawn Duration:</b> {self.yawn_duration:.2f} s</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)


    # ==================== EYE PREDICTION (GIỮ NGUYÊN) ====================
    def predict_eye(self, eye_frame, eye_state):
        try:
            tensor = torch.tensor(eye_frame, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
            tensor = (tensor - 0.5) / 0.5  # Normalize [-1, 1]

            with torch.no_grad():
                output = self.detecteye(tensor)
                prob = output.item()
                label = "Eye open" if prob > 0.5 else "Eye close"
            return label
        except Exception as e:
            print(f"Error predicting eye: {e}")
            return eye_state


    # ==================== YAWN PREDICTION (DÙNG CNN) ====================
    def predict_yawn(self, mouth_roi):
        try:
            if mouth_roi.size == 0:
                return self.yawn_state

            # Resize về 30x60 (width x height)
            mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            mouth_resized = cv2.resize(mouth_gray, (30, 60))  # (w, h)
            mouth_input = mouth_resized.astype(np.float32) / 255.0
            mouth_input = (mouth_input - 0.5) / 0.5  # [-1, 1]
            mouth_tensor = torch.from_numpy(mouth_input).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,60,30)

            with torch.no_grad():
                output = self.detectyawn(mouth_tensor)
                prob = output.item()
                label = "Yawn" if prob > 0.5 else "Not Yawn"

            # Buffer để lọc nhiễu
            self.yawn_buffer.append(label)
            yawn_count = sum(1 for x in self.yawn_buffer if x == "Yawn")
            if yawn_count > len(self.yawn_buffer) * 0.7:  # 70% frame gần nhất là yawn
                self.yawn_state = "Yawn"
            else:
                self.yawn_state = "Not Yawn"

            return self.yawn_state

        except Exception as e:
            print(f"Error predicting yawn: {e}")
            return self.yawn_state

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and self.frame_queue.qsize() < 2:
                self.frame_queue.put(frame)

    def process_frames(self):
        fps = 60
        frame_time = 1.0 / fps

        while not self.stop_event.is_set():
            start_time = time.time()
            try:
                frame = self.frame_queue.get(timeout=1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        points = [ (int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih))
                                  for i in self.points_ids ]

                        if len(points) >= 7:
                            x1, y1 = points[0]  
                            x2, _ = points[1]  
                            _, y3 = points[2]  
                            x4, y4 = points[3]  
                            x5, y5 = points[4]  
                            x6, y6 = points[5]  
                            x7, y7 = points[6]  

                            x6, x7 = min(x6, x7), max(x6, x7)
                            y6, y7 = min(y6, y7), max(y6, y7)

                            # Crop vùng
                            mouth_roi = frame[y1:y3, x1:x2]
                            margin = 8
                            right_eye_roi = frame[max(0, y4-margin):y5+margin, max(0, x4-margin):x5+margin]
                            left_eye_roi = frame[max(0, y6-margin):y7+margin, x6-margin:x7+margin]

                            # cv2.imshow("roi", mouth_roi)
                            # cv2.imshow("eyes", right_eye_roi)
                            # cv2.imshow("ccc", left_eye_roi)
                            # Eye
                            if right_eye_roi.size > 0 and left_eye_roi.size > 0:
                                right_eye_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
                                left_eye_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
                                right_resized = cv2.resize(right_eye_gray, (30, 60))
                                left_resized = cv2.resize(left_eye_gray, (30, 60))

                                right_input = np.expand_dims(right_resized, axis=(0, -1)) / 255.0
                                left_input = np.expand_dims(left_resized, axis=(0, -1)) / 255.0

                                self.right_eye_state = self.predict_eye(right_input, self.right_eye_state)
                                self.left_eye_state = self.predict_eye(left_input, self.left_eye_state)

                            # Yawn
                            if mouth_roi.size > 0:
                                self.yawn_state = self.predict_yawn(mouth_roi)

                            # Logic đếm
                            if self.left_eye_state == "Eye close" and self.right_eye_state == "Eye close":
                                if not (self.left_eye_still_closed and self.right_eye_still_closed):
                                    self.blinks += 1
                                    self.left_eye_still_closed = self.right_eye_still_closed = True
                                self.microsleeps += frame_time
                            else:
                                self.left_eye_still_closed = self.right_eye_still_closed = False
                                if self.microsleeps > 0.1:
                                    self.microsleeps = max(0, self.microsleeps - frame_time)

                            if self.yawn_state == "Yawn":
                                if not self.yawn_in_progress:
                                    self.yawns += 1
                                    self.yawn_in_progress = True
                                self.yawn_duration += frame_time
                            else:
                                if self.yawn_in_progress:
                                    self.yawn_in_progress = False
                                    self.yawn_duration = 0

                            self.update_info()
                            self.display_frame(frame)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process: {e}")

            # Giữ FPS ổn định
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_event.set()
        self.capture_thread.join()
        self.process_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())