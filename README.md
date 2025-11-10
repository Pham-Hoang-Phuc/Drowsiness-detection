# Drowsiness Detection

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Kiến trúc & Cách hoạt động](#kiến-trúc--cách-hoạt-động)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Bước 1: Cài đặt và Chạy Server](#bước-1-cài-đặt-và-chạy-server)
  - [Bước 2: Chạy Client (Giao diện)](#bước-2-chạy-client-giao-diện)
- [Thông tin Mô hình](#thông-tin-mô-hình)
- [Cấu trúc Thư mục](#cấu-trúc-thư-mục)

## Giới thiệu
Dự án này giúp **phát hiện tình trạng buồn ngủ của người lái xe** theo thời gian thực.

Ứng dụng sử dụng một mô hình deep learning để phân tích video từ webcam, phân loại trạng thái của người lái xe dựa trên các đặc điểm của mắt và miệng. Mô hình có thể nhận diện 3 trạng thái chính: **Mắt mở**, **Mắt nhắm**, và **Ngáp (Yawn)**.

## Kiến trúc & Cách hoạt động
Dự án được xây dựng theo mô hình **Client-Server**:

* **Server (`server.py`):**
    * Khởi chạy máy chủ tại địa chỉ `127.0.0.1:9001`.
    * Chạy mô hình AI (CNN) để xử lý từng khung hình, phát hiện khuôn mặt và phân loại trạng thái (mở, nhắm, ngáp).
    * Gửi video đã xử lý và dữ liệu JSON (số lần nháy mắt, ngáp, thời gian nhắm mắt, trạng thái buồn ngủ, v.v.) đến Client.
      
* **Client (`DrowsinessClient.exe`):**
    * Giao diện Windows Forms (C#) hiển thị video trực tiếp và các chỉ số từ Server.
    * Khi người dùng bấm "Connect", Client sẽ kết nối đến Server (`127.0.0.1:9001`).
    * Cập nhật liên tục các chỉ số: số lần nháy mắt, thời gian ngáp, microsleep, thời gian xử lý.

## Hướng dẫn sử dụng

Để chạy ứng dụng, bạn cần thực hiện theo 2 bước: **(1)** Cài đặt môi trường và chạy Server, sau đó **(2)** Khởi động Client.

### Bước 1: Cài đặt và Chạy Server

Dự án này được triển khai trong môi trường **Anaconda** (anaconda prompt), sử dụng **PyTorch GPU** để tăng tốc quá trình huấn luyện và chạy mô hình.

**1. Tạo môi trường Anaconda (Khuyến nghị):**
```bash
# Tạo một môi trường mới tên là 'drowsy' với Python 3.10
conda create -n drowsy python=3.10

# Kích hoạt môi trường
conda activate drowsy
```

**2. Cài đặt các thư viện:**

Cài bản **PyTorch GPU** tương thích với **CUDA 12.1**, và các thư viện cần thiết:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
> **Lưu ý:** Nếu máy chưa có driver NVIDIA hoặc CUDA Toolkit, hãy cài đặt trước khi chạy PyTorch GPU.

```bash
# YOLOv11 (classification)
pip install ultralytics

# Mediapipe - phát hiện khuôn mặt, mắt, miệng
pip install mediapipe

# PyQt5 - giao diện hiển thị video
pip install PyQt5

# Các thư viện phụ trợ
pip install opencv-python
```

**3. Khởi chạy Server:**
Sau khi cài đặt xong, chạy file `server.py`:
```bash
python server.py
```
Nếu thành công, bạn sẽ thấy thông báo trong terminal:
```
[OK] Camera đã mở
Server đang chạy tại 127.0.0.1:9001
```
*Hãy giữ cửa sổ terminal này chạy.*

### Bước 2: Chạy Client (Giao diện)

1.  Từ thư mục dự án, bấm theo đường dẫn sau DrowsinessClient\bin\Debug\net8.0-windows.
2.  Nhấp đúp chuột để chạy file **`DrowsinessClient.exe`**
3.  Một giao diện (UI) sẽ xuất hiện.
4.  Nhấp vào nút **"Connect"**.

Màn hình sẽ hiển thị video từ webcam của bạn, kèm theo các phân tích và cảnh báo về tình trạng buồn ngủ (nếu có).

## Thông tin Mô hình
* **Model:** CNN-cls.
* **Nhiệm vụ:** Phân loại Mắt & Ngáp → Phát hiện Buồn ngủ.
* **Dữ liệu huấn luyện:**
    * [YawDD Dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) (Cho phát hiện ngáp).
    * [MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) (Cho phát hiện mắt đóng/mở).

## Cấu trúc Thư mục
```
.
├── DrowsinessClient/       # (Có thể chứa mã nguồn của Client)
├── runs/                   # (Kết quả huấn luyện/log của CNN)
├── .gitignore
├── hybrid_drowsiness_detector.py # (Logic AI/model cốt lõi)
├── README.md               # (Bạn đang đọc file này)
├── requirements.txt        # (Các thư viện Python cho Server)
├── sound.wav               # (Âm thanh cảnh báo)
└── server.py               # (File chạy Server backend)
```
