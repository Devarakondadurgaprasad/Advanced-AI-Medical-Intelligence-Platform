# Advanced AI Medical Intelligence Platform

## Project Report

---

# Submitted By

**Name:** Devarakonda Durga Prasad

**Position:** AI/ML Engineer Assignment

**Company:** SN Matrix Software Pvt. Ltd.

---

# 1. Introduction

The **Advanced AI Medical Intelligence Platform** is an end-to-end Artificial Intelligence system developed for automated chest X-ray analysis. The platform combines Deep Learning, Explainable AI (XAI), Large Language Models (LLMs), RESTful APIs, database management, and a web-based interface into a single application.

The objective of this project is to assist in medical image analysis by providing disease prediction, explainable visualization, AI-generated medical reports, and persistent storage of prediction history.

> **Disclaimer:** This project is developed for educational and technical evaluation purposes only and should not be considered a replacement for professional medical diagnosis.

---

# 2. Objectives

The primary objectives of this project are:

- Predict pneumonia from chest X-ray images
- Generate Grad-CAM visual explanations
- Produce AI-assisted medical reports using Google Gemini
- Build REST APIs using FastAPI
- Store prediction history using SQLite
- Provide an interactive Streamlit web interface
- Support Docker-based deployment

---

# 3. Technologies Used

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.11 |
| Deep Learning | PyTorch |
| CNN Model | DenseNet121 |
| Explainable AI | Grad-CAM |
| LLM | Google Gemini API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite + SQLAlchemy |
| Image Processing | OpenCV, Pillow |
| Deployment | Docker |
| Version Control | Git & GitHub |

---

# 4. Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset.

### Classes

- NORMAL
- PNEUMONIA

### Dataset Structure

```text
dataset/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

Images are resized to **224 × 224 pixels** and normalized before training and inference.

---

# 5. System Architecture

```text
Chest X-ray Image
        │
        ▼
Image Preprocessing
        │
        ▼
DenseNet121 Model
        │
        ▼
Disease Prediction
        │
        ├────────► Grad-CAM
        │
        ▼
Gemini AI Report
        │
        ▼
SQLite Database
        │
        ▼
FastAPI Backend
        │
        ▼
Streamlit Frontend
```

---

# 6. Methodology

The application performs the following steps:

1. Upload a chest X-ray image.
2. Preprocess the image.
3. Predict the disease using DenseNet121.
4. Generate a Grad-CAM heatmap.
5. Generate an AI-assisted report using Google Gemini.
6. Store prediction history in SQLite.
7. Display the results using Streamlit.

---

# 7. Deep Learning Model

The project uses **DenseNet121**, a pretrained convolutional neural network based on Transfer Learning.

### Features

- Transfer Learning
- ImageNet Pretrained Weights
- CrossEntropy Loss
- Adam Optimizer
- Best Model Saving

Output Classes:

- NORMAL
- PNEUMONIA

---

# 8. Explainable AI

Grad-CAM is used to visualize the regions of the chest X-ray responsible for the model prediction.

Advantages:

- Improves transparency
- Supports model interpretation
- Increases confidence in predictions

---

# 9. Large Language Model Integration

Google Gemini API generates AI-assisted medical reports.

The generated report includes:

- Clinical Summary
- Possible Findings
- Recommendations
- Medical Disclaimer

If the Gemini API is unavailable, a fallback report is generated automatically.

---

# 10. REST API

FastAPI provides the following endpoints.

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | /health | Health Check |
| POST | /predict | Disease Prediction |
| POST | /explain | Generate Grad-CAM |
| POST | /generate-report | AI Medical Report |
| GET | /history | Prediction History |
| GET | /history/{id} | Retrieve Specific Record |

---

# 11. Database

SQLite with SQLAlchemy stores the following information:

- Image Name
- Prediction
- Confidence Score
- Grad-CAM Image
- AI-generated Report
- Inference Latency
- Timestamp

---

# 12. Frontend

The application provides an interactive Streamlit interface supporting:

- Chest X-ray upload
- Disease prediction
- Confidence score
- Grad-CAM visualization
- AI-generated report
- Prediction history

---

# 13. Deployment

The project supports Docker-based deployment.

Deployment files include:

- Dockerfile
- requirements.txt
- .env.example

The application can be deployed on Docker, Render, Railway, or other cloud platforms.

---

# 14. Project Structure

```text
Advanced-AI-Medical-Intelligence-Platform/

├── app/
│   └── main.py
│
├── frontend/
│   └── streamlit_app.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── artifacts/
│
├── outputs/
│
├── README.md
├── PROJECT_REPORT.pdf
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

---

# 15. Future Enhancements

Future improvements may include:

- Multi-class disease classification
- User authentication
- Electronic Health Record (EHR) integration
- Cloud deployment
- PACS integration
- Additional Explainable AI techniques

---

# 16. Conclusion

The **Advanced AI Medical Intelligence Platform** successfully integrates Deep Learning, Explainable AI, Large Language Models, REST APIs, database management, and a modern web interface into a unified application for chest X-ray analysis.

The platform provides disease prediction, Grad-CAM visualization, AI-generated medical reports, and persistent storage of prediction history while following modern software engineering practices. The project satisfies the major technical requirements of the AI/ML Engineer assignment and serves as a scalable foundation for future AI-assisted healthcare applications.

---

## References

1. PyTorch Documentation — https://pytorch.org/
2. FastAPI Documentation — https://fastapi.tiangolo.com/
3. Streamlit Documentation — https://streamlit.io/
4. Google Gemini API Documentation — https://ai.google.dev/
5. DenseNet: Densely Connected Convolutional Networks (CVPR 2017)
6. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
