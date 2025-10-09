# Drowsiness Detection

## Giới thiệu
Dự án này giúp **phát hiện tình trạng buồn ngủ của người lái xe** dựa trên hình ảnh khuôn mặt và mắt.  
Mô hình được huấn luyện để phân loại các trạng thái **mắt mở**, **mắt nhắm**, và **ngáp**.

Mình sử dụng mô hình **YOLOv11 (classification)** được **custom-train** trên hai tập dữ liệu:
- [YawDD Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)
- [MRL Eye Dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset)

---

## Mô hình
- **Model:** YOLOv11n-cls  
- **Task:** Eye & drowsiness classification  
- **Training data:** YawDD + MRL Eyes  
