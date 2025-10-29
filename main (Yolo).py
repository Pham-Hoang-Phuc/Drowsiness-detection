import queue
import threading
import time
import winsound
import cv2 
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from collections import deque

class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.yawn_state = ''
        self.left_eye_state =''
        self.right_eye_state= ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        self.yawn_buffer = deque(maxlen=100)  # lưu 15 frame gần nhất

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
        
        
        ############################# THAY MODEL Ở ĐÂY ##############################
        self.detecteye = YOLO(
            r"D:\Fod\desktop\Drowsiness-detection\runs\custom\eyes-ver4 (MRL).pt"
            )
        self.detectyawn = YOLO(
            r"D:\Fod\desktop\Drowsiness-detection\runs\custom\yawn-ver4 (YawDD).pt"
            )
        #-----------------------------------------------------------------------------
        
        self.cap = cv2.VideoCapture(0)
        time.sleep(1.000)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()
        
        
    def update_info(self):
        if round(self.yawn_duration, 2) > 4.0:
            # self.play_sound_in_thread()
            self.alert_text = "<p style='color: orange; font-weight: bold;'> Alert: Prolonged Yawn Detected!</p>"

        if round(self.microsleeps, 2) > 3.0:
            # self.play_sound_in_thread()
            self.alert_text = "<p style='color: red; font-weight: bold;'> Alert: Prolonged Microsleep Detected!</p>"


        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333; font-size:18px'>"
            f"<h2 style='text-align: center; color: #4CAF50;font-size: 24px';>Drowsiness Detector</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"  # Display alert if it exists
            f"<p><b>👁️ Blinks:</b> {self.blinks}</p>"
            f"<p><b>💤 Microsleeps:</b> {round(self.microsleeps,2)} seconds</p>"
            f"<p><b>😮 Yawns:</b> {self.yawns}</p>"
            f"<p><b>⏳ Yawn Duration:</b> {round(self.yawn_duration,2)} seconds</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)



    def predict_eye(self, eye_frame, eye_state):
        results_eye = self.detecteye.predict(eye_frame, verbose=False)

        if results_eye is None or len(results_eye) == 0:
            print("No prediction result returned.")
            return eye_state

        if results_eye[0].probs is None:
            print("No probabilities returned (model may be detection, not classification).")
            return eye_state

        probs = results_eye[0].probs
        top_class_id = probs.top1
        label = self.detecteye.names[top_class_id]
        
        if label.lower() in ["close", "closed", "close-eyes"]:
            label = "Eye close"
            
        eye_state = label

        return eye_state



    def predict_yawn(self, yawn_frame):
        results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
        
        # Lấy kết quả classification
        probs = results_yawn[0].probs  
        if probs is None:
            print("Warning: results_yawn[0].probs = None, có thể model không phải classification model.")
            return self.yawn_state

        # Lấy class có xác suất cao nhất
        top_class_id = probs.top1
        label = self.detectyawn.names[top_class_id]

        # Thêm vào buffer để làm mượt kết quả
        self.yawn_buffer.append(label)

        # Nếu trong 15 frame gần nhất có hơn 10 frame là "Yawn" → coi là đang ngáp
        if self.yawn_buffer.count("yawn") + self.yawn_buffer.count("Yawn") > 90:
            self.yawn_state = "Yawn"
        else:
            self.yawn_state = "Not Yawn"

        return self.yawn_state                           



    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break



    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)        
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        points = []

                        for point_id in self.points_ids:
                            lm = face_landmarks.landmark[point_id]
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            points.append((x, y))
                            # cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                        if len(points) != 0:
                            x1, y1 = points[0]  
                            x2, _ = points[1]  
                            _, y3 = points[2]  

                            x4, y4 = points[3]  
                            x5, y5 = points[4]  

                            x6, y6 = points[5]  
                            x7, y7 = points[6]  

                            x6, x7 = min(x6, x7), max(x6, x7)
                            y6, y7 = min(y6, y7), max(y6, y7)

                            mouth_roi = frame[y1:y3, x1:x2]
                            # cv2.imshow("mouth roi",mouth_roi)
                            
                            right_eye_roi = frame[y4:y5, x4:x5]
                            # cv2.imshow("right eye roi",right_eye_roi)
                            
                            left_eye_roi = frame[y6:y7, x6:x7]
                            # cv2.imshow("left eye roi",left_eye_roi)
                            
                            # chuyển sang grayscale
                            right_eye_roi = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
                            left_eye_roi = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
                            
                            # cv2.imshow("right", right_eye_roi)
                            # cv2.imshow("left", left_eye_roi)
                            try:
                                self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
                                self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
                                self.predict_yawn(frame)
                                
                            except Exception as e:
                                print(f"Error al realizar la predicción: {e}")

                            if self.left_eye_state == "Eye close" and self.right_eye_state == "Eye close":
                                if not self.left_eye_still_closed and not self.right_eye_still_closed:
                                    self.left_eye_still_closed, self.right_eye_still_closed = True , True
                                    self.blinks += 1 
                                self.microsleeps += 45 / 1000
                            else:
                                if self.left_eye_still_closed and self.right_eye_still_closed :
                                    self.left_eye_still_closed, self.right_eye_still_closed = False , False
                                self.microsleeps = 0

                            if self.yawn_state == "Yawn":
                                if not self.yawn_in_progress:
                                    self.yawn_in_progress = True
                                    self.yawns += 1  
                                self.yawn_duration += 45 / 1000
                            else:
                                if self.yawn_in_progress:
                                    self.yawn_in_progress = False
                                    self.yawn_duration = 0

                            self.update_info()
                            self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
            frequency = 1000 
            duration = 500  
            winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()
        
    def show_alert_on_frame(self, frame, text="Alerta!"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (0, 0, 255) 
        line_type = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())