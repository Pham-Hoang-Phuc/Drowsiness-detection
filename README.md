# Drowsiness Detection

## Table of Contents
- [Introduction](#introduction)
- [Architecture & How It Works](#architecture--how-it-works)
- [Usage Instructions](#usage-instructions)
  - [Setup and Running the Server](#setup-and-running-the-server)
- [Model Information](#model-information)
- [Directory Structure](#directory-structure)

## Introduction
This project aims to **detect driver drowsiness** in real-time.

The application uses a deep learning model to analyze video from a webcam, classifying the driver's state based on eye and mouth features. The model can identify three main states: **Eyes open**, **Eyes closed**, and **Yawning**.

## Architecture & How It Works
The project is built on a **Client-Server** model:

* **Server (`server.py`):**
    * Runs the server at `127.0.0.1:9001`.
    * Executes the AI model (CNN) to process each frame, detect faces, and classify states (open, closed, yawning).
    * Sends processed video and JSON data (blink count, yawn count, eye closure duration, drowsiness status, etc.) to the Client.

* **Client (`DrowsinessClient.exe`):**
    * A Windows Forms (C#) interface displaying live video and metrics from the Server.
    * When the user clicks "Connect", the Client connects to the Server (`127.0.0.1:9001`).
    * Continuously updates metrics: blink count, yawn duration, microsleep, processing time.

## Usage Instructions

To run the application, follow two steps: Set up the environment and run the Server.

### Setup and Running the Server

The project is deployed in an **Anaconda** environment (Anaconda Prompt) with Python 3.10, using **PyTorch GPU** to accelerate training and model execution.

**1. Create an Anaconda Environment (Recommended):**
```bash
# Create a new environment named 'drowsy' with Python 3.10
conda create -n drowsy python=3.10

# Activate the environment
conda activate drowsy
```

**2. Install Dependencies:**

Install the **PyTorch GPU** version compatible with **CUDA 12.1**, along with required libraries:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
> **Note:** If your machine does not have NVIDIA drivers or CUDA Toolkit installed, install them before running PyTorch GPU.

```bash
# Mediapipe
pip install mediapipe

# OpenCV
pip install opencv-python

# Pygame
pip install pygame
```

If not using an **Anaconda** environment (Anaconda Prompt), create a virtual environment and run the following in the terminal:
```bash
pip install -r requirements.txt
```

**3. Start the Server:**

After setup, run the `run.py` file. If successful, you will see the following message in the terminal:
```
[OK] Camera opened
Server running at 127.0.0.1:9001
```
In the Client application, click the "CONNECT TO" button. Once connected successfully, the Client will receive and display the processed video from the Server, along with detailed metrics and drowsiness alerts.

<div align="center"> <img src="img/image.png" alt="Demo image eyes closed" width="48%">

<img src="img/image_1.png" alt="Demo image yawning" width="48%"> </div>

## Model Information
* **Model:** CNN-cls.
* **Task:** Eye & Yawn Classification → Drowsiness Detection.
* **Training Data:**
    * [YawDD Dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) (For yawn detection).
    * [MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) (For eye open/closed detection).

## Directory Structure
```
.
├── alarm/                  # (Folder containing alert sounds)
├── DrowsinessClient/       # (Client source code)
├── runs/                   # (CNN training results/logs)
├── .gitignore
├── hybrid_drowsiness_detector.py # (Core AI/model logic)
├── README.md               # (You are reading this file)
├── requirements.txt        # (Python libraries)
├── run.py                  # (Executable file)
└── server.py               # (Server backend file)
```
