# Drowsiness Detection

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Kiến trúc & Cách hoạt động](#kiến-trúc--cách-hoạt-động)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Cài đặt và Chạy Server](#cài-đặt-và-chạy-server)
- [Thông tin Mô hình](#thông-tin-mô-hình)
- [Cấu trúc Thư mục](#cấu-trúc-thư-mục)

## Giới thiệu
Dự án này giúp **phát hiện tình trạng buồn ngủ của người lái xe** theo thời gian thực.

Ứng dụng sử dụng một mô hình deep learning để phân tích video từ webcam, phân loại trạng thái của người lái xe dựa trên các đặc điểm của mắt và miệng. Mô hình có thể nhận diện 3 trạng thái chính: **Mắt mở**, **Mắt nhắm**, và **Ngáp**.

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

Để chạy ứng dụng, bạn cần thực hiện theo 2 bước: Cài đặt môi trường và chạy Server.

### Cài đặt và chạy server

Dự án này được triển khai trong môi trường **Anaconda** (anaconda prompt) với python 3.10, sử dụng **PyTorch GPU** để tăng tốc quá trình huấn luyện và chạy mô hình.

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
# Mediapipe 
pip install mediapipe

# OpenCv
pip install opencv-python

# Pygame
pip install pygame
```

Nếu không sử dụng môi trường **Anaconda** (anaconda prompt) thì hãy tạo môi trường ảo và chạy lệnh sau trong terminal
```bash
pip install -r requirements.txt
```

**3. Khởi chạy Server:**

Sau khi cài đặt xong, chạy file `run.py`. Nếu thành công, bạn sẽ thấy thông báo trong terminal:
```
[OK] Camera đã mở
Server đang chạy tại 127.0.0.1:9001
```
Trong ứng dụng Client, hãy bấm nút "CONNECT TO". Sau khi kết nối thành công, Client sẽ nhận và hiển thị video đã được Server xử lý, cùng với các chỉ số và cảnh báo buồn ngủ chi tiết.

<div align="center"> <img src="img/image.png" alt="Ảnh demo mắt nhắm" width="48%">

<img src="img/image_1.png" alt="Ảnh demo đang ngáp" width="48%"> </div>

## Thông tin Mô hình
* **Model:** CNN-cls.
* **Nhiệm vụ:** Phân loại Mắt & Ngáp → Phát hiện Buồn ngủ.
* **Dữ liệu huấn luyện:**
    * [YawDD Dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) (Cho phát hiện ngáp).
    * [MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) (Cho phát hiện mắt đóng/mở).

## Cấu trúc Thư mục
```
.
├── alarm/                  # (thư mục chứa các âm thanh cảnh báo)
├── DrowsinessClient/       # (Có thể chứa mã nguồn của Client)
├── runs/                   # (Kết quả huấn luyện/log của CNN)
├── .gitignore
├── hybrid_drowsiness_detector.py # (Logic AI/model cốt lõi)
├── README.md               # (Bạn đang đọc file này)
├── requirements.txt        # (Các thư viện Python cho Server)
├── run.py                  # (File thực thi)
└── server.py               # (File chạy Server backend)
```
