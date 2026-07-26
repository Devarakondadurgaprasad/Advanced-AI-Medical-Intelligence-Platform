# Advanced AI Medical Intelligence Platform – Project Report

## 1. Introduction
This project implements an end-to-end AI platform for chest X-ray pneumonia analysis with explainability, LLM report generation, API serving, and persistent storage.

## 2. Objective
- Predict disease class from chest X-ray images
- Explain predictions using Grad-CAM
- Generate AI-assisted medical summary reports
- Expose REST APIs
- Store inference history in a database
- Provide a user-friendly UI

## 3. Dataset
Chest X-Ray Images (Pneumonia), binary classes:
- NORMAL
- PNEUMONIA

## 4. Methodology
- Transfer learning using DenseNet121
- Resize + normalize preprocessing
- Training with CrossEntropyLoss and Adam optimizer
- Save best validation model

## 5. Explainable AI
Grad-CAM generates heatmaps over important regions in the chest X-ray and overlays them for visual interpretation.

## 6. LLM Integration
Google Gemini API generates structured output:
1. Clinical Summary
2. Possible Findings
3. Recommendations
4. Disclaimer

Fallback text is returned if the API key is unavailable.

## 7. Backend
FastAPI endpoints:
- /health
- /predict
- /explain
- /generate-report
- /history
- /history/{id}

## 8. Database
SQLite + SQLAlchemy stores:
- image name
- prediction
- confidence
- Grad-CAM path
- generated report
- latency
- timestamp

## 9. Frontend
Streamlit UI supports:
- image upload
- prediction display
- Grad-CAM visualization
- report display
- history table

## 10. Deployment
Dockerfile is provided for containerized deployment.

## 11. Conclusion
The platform satisfies assignment requirements with a complete pipeline from model inference to explainability, LLM reporting, API serving, and database persistence.
