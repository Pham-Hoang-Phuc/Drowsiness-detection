# Drowsiness Detection

## Giới thiệu  
Dự án này giúp **phát hiện tình trạng buồn ngủ của người lái xe** dựa trên hình ảnh khuôn mặt, mắt và miệng.  
Mô hình được huấn luyện để phân loại các trạng thái **mắt mở**, **mắt nhắm**, và **ngáp (yawn)**.  

Mình sử dụng mô hình **YOLOv11 (classification)** được **custom-train** trên hai tập dữ liệu:  
- [YawDD Dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset)  
- [MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)  

## Mô hình
- **Model:** YOLOv11n-cls and CNN cls  
- **Task:** Eye & Yawn → Drowsiness Classification  
- **Training Data:** YawDD + MRL Eyes  
## Môi trường & Cài đặt  

Dự án này được triển khai trong môi trường **Anaconda** (anaconda prompt), sử dụng **PyTorch GPU** để tăng tốc quá trình huấn luyện và chạy mô hình.

### Bước 1: Tạo môi trường mới  
```bash
conda create -n drowsy python=3.10
conda activate drowsy
````

### Bước 2: Cài đặt PyTorch GPU (CUDA)

Cài bản **PyTorch GPU** tương thích với **CUDA 12.1**:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

> **Lưu ý:** Nếu máy chưa có driver NVIDIA hoặc CUDA Toolkit, hãy cài đặt trước khi chạy PyTorch GPU.

---

### Bước 3: Cài đặt các thư viện cần thiết khác

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
