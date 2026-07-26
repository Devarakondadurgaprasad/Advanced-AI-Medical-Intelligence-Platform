# Advanced AI Medical Intelligence Platform

## Overview
This project is an end-to-end AI system for chest X-ray analysis with:

- Deep Learning (PyTorch, DenseNet121 transfer learning)
- Explainable AI (Grad-CAM)
- AI-assisted report generation (Google Gemini API)
- FastAPI REST backend
- SQLite + SQLAlchemy prediction history
- Streamlit frontend
- Docker support

> Disclaimer: This tool is AI-assisted and not a medical diagnosis system.

## Repository
https://github.com/Deekshithpoleboina/Advanced-AI-Medical-Intelligence-Platform

## API Endpoints
- `GET /health`
- `POST /predict`
- `POST /explain`
- `POST /generate-report`
- `GET /history`
- `GET /history/{id}`

Swagger docs: `http://localhost:8000/docs`

## Model
DenseNet121 pretrained on ImageNet and fine-tuned for:
- NORMAL
- PNEUMONIA

## Run Locally
```bash
python -m venv .venv
# activate venv
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py --server.port 8501
```

## Training
```bash
export DATA_DIR=dataset/chest_xray
python training/train.py
python training/evaluate.py
```

## Deployment Status
Not deployed at submission time. Dockerized and deployment-ready.
