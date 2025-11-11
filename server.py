# server/server.py
import socket
import struct
import json
import cv2
import time
import sys
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

try:
    from hybrid_drowsiness_detector import HybridDrowsinessDetector
except Exception as e:
    print(f"[FATAL] Import lỗi: {e}")
    sys.exit(1)

print("[INFO] Khởi tạo detector...")
detector = HybridDrowsinessDetector()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Không mở camera!")
    sys.exit(1)
print("[OK] Camera đã mở")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server.bind(("127.0.0.1", 9001))
    server.listen(1)
    print("Server đang chạy tại 127.0.0.1:9001")
except Exception as e:
    print(f"[ERROR] Bind lỗi: {e}")
    sys.exit(1)

conn, addr = server.accept()
print(f"Kết nối từ: {addr}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Xử lý (không vẽ box)
        processed_frame = detector.process_frames(frame.copy())

        # Nén ảnh
        _, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        img_bytes = jpeg.tobytes()

        # Dữ liệu gửi C# (không gửi box)
        data = {
            "blinks": detector.blinks,
            "microsleeps": round(detector.microsleeps, 2),
            "yawns": detector.yawns,
            "yawn_duration": round(detector.yawn_duration, 2),
            "processing_time": round((time.time() - detector.last_process_time) * 1000, 2),
            "eyes_closed_duration": round(detector.eyes_closed_duration, 2),
            "drowsy": detector.drowsy
        }
        json_bytes = json.dumps(data).encode('utf-8')

        # print(f"Sending data: {data}")
        
        packet = (
            struct.pack(">I", len(img_bytes)) + img_bytes +
            struct.pack(">I", len(json_bytes)) + json_bytes
        )
        conn.send(packet)

        time.sleep(0.033)

except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cap.release()
    conn.close()
    server.close()
    print("Server đã đóng.")