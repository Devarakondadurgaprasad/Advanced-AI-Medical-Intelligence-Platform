import os
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Advanced AI Medical Intelligence Platform"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_history.db")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/best_model.pth")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "artifacts/class_names.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

UPLOAD_DIR = Path("uploads")
GRADCAM_DIR = Path("outputs/gradcam")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    gradcam_path = Column(String(500), nullable=True)
    report = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam: str
    report: str
    record_id: int
    latency_ms: float

class ExplainResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam: str

class ReportRequest(BaseModel):
    prediction: str
    confidence: float
    gradcam_observation: str = "Salient lung region highlighted by Grad-CAM."

class ReportResponse(BaseModel):
    report: str

class HistoryResponse(BaseModel):
    id: int
    image_name: str
    prediction: str
    confidence: float
    gradcam_path: Optional[str]
    report: Optional[str]
    latency_ms: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not Path(CLASS_NAMES_PATH).exists() or not Path(MODEL_PATH).exists():
    print("WARNING: Model artifacts missing. Add artifacts/best_model.pth and artifacts/class_names.json")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
if Path(CLASS_NAMES_PATH).exists():
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)

model = models.densenet121(weights=None)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, len(CLASS_NAMES))
if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def predict_pil(image: Image.Image):
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), idx, x

def make_gradcam(image: Image.Image, input_tensor) -> str:
    target_layers = [model.features.denseblock4]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb = np.array(image.resize((224,224))).astype(np.float32) / 255.0
    vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = GRADCAM_DIR / out_name
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return f"/outputs/gradcam/{out_name}"

def generate_report(prediction: str, confidence: float, gradcam_observation: str) -> str:
    fallback = (
        f"Clinical Summary: The model predicts '{prediction}' with confidence {confidence:.2%}. "
        f"Possible Findings: Grad-CAM highlights lung regions ({gradcam_observation}). "
        f"Recommendations: Correlate with clinical findings and radiologist review. "
        f"Disclaimer: This is AI-assisted output and not a medical diagnosis."
    )
    if not GEMINI_API_KEY:
        return fallback
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gmodel = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
You are a medical AI assistant.

Prediction: {prediction}
Confidence: {confidence:.4f}
Grad-CAM observation: {gradcam_observation}

Generate:
1) Clinical Summary
2) Possible Findings
3) Recommendations
4) Disclaimer

Clearly mention this is not a diagnosis.
"""
        resp = gmodel.generate_content(prompt)
        return resp.text.strip() if hasattr(resp, "text") and resp.text else fallback
    except Exception:
        return fallback

app = FastAPI(title=APP_NAME)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
def home():
    return {"message": APP_NAME}

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/generate-report", response_model=ReportResponse)
def report(payload: ReportRequest):
    text = generate_report(payload.prediction, payload.confidence, payload.gradcam_observation)
    return ReportResponse(report=text)

@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image")
    pred, conf, _, x = predict_pil(image)
    gradcam_url = make_gradcam(image, x)
    return ExplainResponse(prediction=pred, confidence=conf, gradcam=gradcam_url)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>10MB)")

    ext = Path(file.filename).suffix if file.filename else ".jpg"
    saved = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(saved, "wb") as f:
        f.write(contents)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image")

    t0 = time.perf_counter()
    pred, conf, _, x = predict_pil(image)
    gradcam_url = make_gradcam(image, x)
    report_text = generate_report(pred, conf, "salient lung region highlighted")
    latency_ms = (time.perf_counter() - t0) * 1000

    rec = PredictionHistory(
        image_name=file.filename or saved.name,
        prediction=pred,
        confidence=conf,
        gradcam_path=gradcam_url,
        report=report_text,
        latency_ms=latency_ms
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return PredictResponse(
        prediction=pred,
        confidence=conf,
        gradcam=gradcam_url,
        report=report_text,
        record_id=rec.id,
        latency_ms=latency_ms
    )

@app.get("/history", response_model=list[HistoryResponse])
def history(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit).all()

@app.get("/history/{record_id}", response_model=HistoryResponse)
def history_by_id(record_id: int, db: Session = Depends(get_db)):
    rec = db.query(PredictionHistory).filter(PredictionHistory.id == record_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")
    return rec
