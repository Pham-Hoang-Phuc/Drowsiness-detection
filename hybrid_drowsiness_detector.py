# hybrid_drowsiness_detector.py
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import time
from collections import deque
import pygame

# ==================== DEFINE CNN STRUCTURE ====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13 * 6 * 128, 256)
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

# ==================== MAIN CLASS ====================
class HybridDrowsinessDetector:
    def __init__(self):
        self.blinks = 0
        self.microsleeps = 0.0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.eyes_closed_duration = 0.0
        self.last_process_time = time.time()
        self.MAR_THRESHOLD = 0.7  
        self.MAR_DURATION_THRESHOLD = 4.0  
        self.mar_start_time = None

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.yawn_buffer = deque(maxlen=20)

        # For 1-minute window analysis (assuming 30 FPS)
        self.frame_data = deque(maxlen=1800)  # 30 FPS * 60 seconds
        self.frame_timestamps = deque(maxlen=1800)
        self.drowsy = False  # Store drowsiness status as boolean
        self.was_drowsy = False
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound('alarm\Radar.wav')
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        self.detecteye = CNN().to(self.device)
        self.detecteye.load_state_dict(torch.load(r"runs\custom\eye_state_cnn_ver2.pth", map_location=self.device))
        self.detecteye.eval()

        self.detectyawn = CNN().to(self.device)
        self.detectyawn.load_state_dict(torch.load(r"runs\custom\mouth_cnn_ver2.pth", map_location=self.device))
        self.detectyawn.eval()

    # ==================== Predict Eye ====================
    def predict_eye(self, eye_frame):
        try:
            tensor = torch.tensor(eye_frame, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
            tensor = (tensor - 0.5) / 0.5
            with torch.no_grad():
                prob = self.detecteye(tensor).item()
            return "Eye open" if prob > 0.5 else "Eye close"
        except:
            return "Eye open"

    # ==================== Predict Yawn ====================
    def predict_yawn(self, mouth_roi):
        try:
            if mouth_roi.size == 0:
                return "Not Yawn"

            mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            mouth_resized = cv2.resize(mouth_gray, (30, 60))
            mouth_input = (mouth_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            mouth_tensor = torch.from_numpy(mouth_input).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prob = self.detectyawn(mouth_tensor).item()
            label = "Yawn" if prob > 0.5 else "Not Yawn"

            self.yawn_buffer.append(label)
            if sum(x == "Yawn" for x in self.yawn_buffer) > len(self.yawn_buffer) * 0.7:
                return "Yawn"
            return "Not Yawn"
        except:
            return "Not Yawn"

    # ==================== MAR ====================
    def compute_mar(self, landmarks, image_w, image_h):
        mouth_indices = [61, 62, 63, 64, 65, 66, 67, 68]
        pts = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in mouth_indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[7]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[6]))
        C = np.linalg.norm(np.array(pts[3]) - np.array(pts[5]))
        D = np.linalg.norm(np.array(pts[0]) - np.array(pts[4]))
        if D == 0:
            return 0.0  # Avoid division by zero
        mar = (A + B + C) / (2.0 * D)
        return mar

    # ==================== Compute Drowsiness Score ====================
    def compute_drowsiness_score(self):
        # Initialize weights
        # W_r: Weight for Eye Closure Ratio
        # W_t: Weight for Continuous Eye Closure Time
        # W_b: Weight for Blink Frequency
        # W_y: Weight for Yawn Frequency
        
        W_r = W_t = W_b = W_y = 0

        # 1. Eye Closure Ratio (r): Proportion of frames with eyes closed in 1 minute
        if len(self.frame_data) > 0:
            closed_eye_frames = sum(1 for data in self.frame_data if data['eyes_closed'])
            r = closed_eye_frames / len(self.frame_data)
            if r > 0.3:  # If eye closure ratio > 30%
                W_r = 1

        # 2. Continuous Eye Closure Time (t): Check if eyes closed > 2 seconds
        if self.eyes_closed_duration > 2.0 and not self.yawn_in_progress:  # Not during yawn
            W_t = 1

        # 3. Blink Frequency (b): Blinks per minute
        if len(self.frame_timestamps) > 1:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0:
                blinks_per_minute = (self.blinks / time_span) * 60
                if time_span < 10:  # observation time too short
                    blinks_per_minute = 0  # skip
                elif blinks_per_minute > 25 or blinks_per_minute < 5:
                    W_b = 1

        else:
            blinks_per_minute = 0  # Default if not enough data

        # 4. Yawn Frequency (y): number of yawns within 1 minute
        if len(self.frame_timestamps) > 1:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0:
                yawns_per_minute = (self.yawns / time_span) * 60
                if yawns_per_minute > 2:
                    W_y = 1
                    
        # Calculate total score T
        T = W_r + W_t + W_b + W_y
        
        # Decision
        self.drowsy = T > 2
        # self.drowsy = True
        
        # alarm sound
        if self.drowsy and not self.was_drowsy:
            print("Phát hiện buồn ngủ!")
            self.alarm_sound.play(1) 
            self.was_drowsy = True
        elif not self.drowsy:
            self.was_drowsy = False
            
        # print(f"Drowsiness Score: {T}, Drowsy: {self.drowsy}, W_r: {W_r}, W_t: {W_t}, W_b: {W_b}, W_y: {W_y}")
        return T, self.drowsy

    # ==================== Process Frame ====================
    def process_frames(self, frame):
        self.last_process_time = time.time()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                pts = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in self.points_ids]

                if len(pts) >= 7:
                    x1, y1 = pts[0]; x2, _ = pts[1]; _, y3 = pts[2]
                    x4, y4 = pts[3]; x5, y5 = pts[4]
                    x6, y6 = pts[5]; x7, y7 = pts[6]

                    x6, x7 = min(x6, x7), max(x6, x7)
                    y6, y7 = min(y6, y7), max(y6, y7)

                    mouth_roi = frame[y1:y3, x1:x2]
                    margin = 8
                    right_eye_roi = frame[max(0, y4 - margin):y5 + margin, max(0, x4 - margin):x5 + margin]
                    left_eye_roi = frame[max(0, y6 - margin):y7 + margin, x6 - margin:x7 + margin]

                    right_eye_state = left_eye_state = "Eye open"

                    if right_eye_roi.size > 0 and left_eye_roi.size > 0:
                        re_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
                        le_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
                        re_resized = cv2.resize(re_gray, (30, 60))
                        le_resized = cv2.resize(le_gray, (30, 60))
                        re_input = np.expand_dims(re_resized, axis=(0, -1)) / 255.0
                        le_input = np.expand_dims(le_resized, axis=(0, -1)) / 255.0
                        right_eye_state = self.predict_eye(re_input)
                        left_eye_state = self.predict_eye(le_input)
                    
                    mar = self.compute_mar(face_landmarks.landmark, iw, ih)
                    cnn_yawn = self.predict_yawn(mouth_roi)

                    # Count logic for eyes
                    eyes_closed = False
                    if right_eye_state == "Eye close" and left_eye_state == "Eye close":
                        if not (self.left_eye_still_closed and self.right_eye_still_closed):
                            self.blinks += 1
                        self.left_eye_still_closed = self.right_eye_still_closed = True
                        self.microsleeps += 1/30
                        self.eyes_closed_duration += 1/30
                        eyes_closed = True
                    else:
                        self.left_eye_still_closed = self.right_eye_still_closed = False
                        self.eyes_closed_duration = 0
                        self.microsleeps = 0

                    # Yawn detection combining MAR and CNN
                    current_time = time.time()
                    if mar > self.MAR_THRESHOLD:
                        if self.mar_start_time is None:
                            self.mar_start_time = current_time
                        if (current_time - self.mar_start_time) >= self.MAR_DURATION_THRESHOLD:
                            mar_state = "Yawn"
                        else:
                            mar_state = "Maybe Yawn"
                    else:
                        self.mar_start_time = None
                        mar_state = "Not Yawn"

                    yawn_state = "Yawn" if mar_state == "Yawn" and cnn_yawn == "Yawn" else "Not Yawn"

                    if yawn_state == "Yawn":
                        if not self.yawn_in_progress:
                            self.yawns += 1
                            self.yawn_in_progress = True
                        self.yawn_duration += 1/30
                    else:
                        if self.yawn_in_progress:
                            self.yawn_in_progress = False
                            self.yawn_duration = 0

                    # Store frame data for 1-minute analysis
                    self.frame_data.append({
                        'eyes_closed': eyes_closed,
                        'yawn': yawn_state == "Yawn"
                    })
                    self.frame_timestamps.append(current_time)

                    # Compute drowsiness score
                    T, self.drowsy = self.compute_drowsiness_score()

        return frame