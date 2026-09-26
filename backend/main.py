import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import UPLOAD_DIR, GEMINI_API_KEY, GEMINI_MODEL, FRONTEND_URL
from backend.modules.data_loader import load_dataset, ALLOWED_EXTENSIONS
from backend.modules.data_cleaner import clean_dataset
from backend.modules.analyzer import dataset_statistics
from backend.modules.visualizer import generate_charts
from backend.modules.ml_model import train_ml_model
from backend.modules.predictor import predict_live
from backend.modules.chat import chat_with_data
from backend.modules.report_generator import generate_report_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Autonomous Data Analyst & Prediction System API",
    version="2.0",
    description="Backend API for automated dataset analysis, cleaning, visualization, ML training, live prediction, and AI chat."
)

# CORS configuration for production (Vercel) and local development
origins = list({
    FRONTEND_URL.rstrip("/"),
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"  # Allow all for flexible public API access
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists securely
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Maximum upload size limit: 50MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# ─── Startup: Validate Gemini API Key ─────────────────────────────────────────
if not GEMINI_API_KEY:
    logger.warning(
        "\n" + "="*60 +
        "\n  WARNING: GEMINI_API_KEY is not set in backend/.env" +
        "\n  The AI Assistant will return error messages until this is fixed." +
        "\n  Add your Gemini API key to: backend/.env" +
        "\n" + "="*60
    )
else:
    logger.info(f"Gemini AI configured — Model: {GEMINI_MODEL}")


class ChatRequest(BaseModel):
    query: str
    file_path: str
    # api_key is intentionally removed — Gemini key is backend-only via .env


class TrainMLRequest(BaseModel):
    file_path: str
    target_column: str


class PredictRequest(BaseModel):
    file_path: str
    target_column: str
    feature_values: Dict[str, Any]


class ClearRequest(BaseModel):
    file_path: str


def sanitize_stats(stats):
    """Sanitize float values like NaN and Inf for clean JSON response."""
    import numpy as np
    if not isinstance(stats, dict):
        return stats
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_stats(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_stats(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, (int, float, np.integer, np.floating)):
            if np.isnan(value) or np.isinf(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


@app.get("/")
def home():
    return {
        "title": "AI Autonomous Data Analyst & Prediction System API",
        "status": "online",
        "version": "2.0",
        "gemini_model": GEMINI_MODEL,
        "ai_configured": bool(GEMINI_API_KEY)
    }


@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend server is reachable and active."}


@app.get("/ai-status")
def ai_status():
    """Quick health check for Gemini AI configuration — no key is exposed."""
    if not GEMINI_API_KEY:
        return {
            "status": "not_configured",
            "model": GEMINI_MODEL,
            "message": "GEMINI_API_KEY is missing from backend/.env"
        }
    return {
        "status": "configured",
        "model": GEMINI_MODEL,
        "message": "Gemini AI is configured and ready."
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded. Please select a valid dataset file.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Generate unique safe filename to avoid filename collisions
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / safe_filename

    # Save uploaded file with file size validation
    file_size = 0
    with file_path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE_BYTES:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="File size exceeds maximum allowed limit of 50MB."
                )
            buffer.write(chunk)

    if file_size == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty (0 bytes).")

    try:
        # 1. Load Dataset
        df_raw = load_dataset(str(file_path))

        # 2. Clean Dataset with Before vs After stats
        df_cleaned, cleaning_report = clean_dataset(df_raw)

        # 3. Analyze raw vs cleaned stats & insights
        raw_stats = dataset_statistics(df_raw, df_cleaned, cleaning_report)

        # 4. Generate Plotly Charts (Priority 1: Primary Visualization System)
        charts = generate_charts(df_cleaned)

        # 5. Automatically detect and train default ML model
        ml_prediction = train_ml_model(df_cleaned, target_col=None)

        # 6. Generate downloadable text report
        report_text = generate_report_content(raw_stats, ml_prediction)

        stats_sanitized = sanitize_stats(raw_stats)

        return {
            "status": "success",
            "file_path": str(file_path.resolve()),
            "filename": file.filename,
            "analysis": stats_sanitized,
            "charts": charts,
            "ml_prediction": ml_prediction,
            "report_text": report_text
        }

    except ValueError as ve:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Data processing failed: {str(e)}"
        )


@app.post("/train-ml")
def train_machine_learning(request: TrainMLRequest):
    req_path = Path(request.file_path)
    if not req_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found. Please re-upload your dataset.")

    try:
        df_raw = load_dataset(str(req_path))
        df_cleaned, _ = clean_dataset(df_raw)

        if request.target_column not in df_cleaned.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.target_column}' is not present in the uploaded dataset."
            )

        ml_result = train_ml_model(df_cleaned, target_col=request.target_column)

        if "error" in ml_result:
            raise HTTPException(status_code=400, detail=ml_result["error"])

        return {
            "status": "success",
            "ml_prediction": ml_result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@app.post("/predict")
def predict_target(request: PredictRequest):
    req_path = Path(request.file_path)
    if not req_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found. Please re-upload your dataset.")

    try:
        df_raw = load_dataset(str(req_path))
        df_cleaned, _ = clean_dataset(df_raw)

        prediction_result = predict_live(
            df_cleaned,
            target_col=request.target_column,
            feature_values=request.feature_values
        )

        if "error" in prediction_result:
            raise HTTPException(status_code=400, detail=prediction_result["error"])

        return prediction_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/chat")
def data_chat(request: ChatRequest):
    if not request.file_path or not request.query:
        raise HTTPException(status_code=400, detail="Missing question query or file reference.")

    req_path = Path(request.file_path)
    if not req_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found. Please re-upload your dataset.")

    # API key is ALWAYS read from backend .env — never from the request body
    response_text = chat_with_data(str(req_path), request.query)
    return {"status": "success", "response": response_text}


@app.post("/clear")
def clear_analysis(request: ClearRequest):
    """
    Secure temporary file deletion with strict path traversal protection.
    Ensures that files being deleted reside strictly within backend/uploads/.
    """
    if not request.file_path or not request.file_path.strip():
        return {"status": "success", "message": "No active file path specified."}

    try:
        target_path = Path(request.file_path).resolve()
        upload_dir_resolved = UPLOAD_DIR.resolve()

        # Priority 10: Path Traversal Protection
        if not target_path.is_relative_to(upload_dir_resolved):
            raise HTTPException(
                status_code=400,
                detail="Security violation: Requested path lies outside the allowed uploads directory."
            )

        if target_path.exists() and target_path.is_file():
            target_path.unlink()
            return {
                "status": "success",
                "message": f"Temporary file '{target_path.name}' deleted successfully."
            }
        else:
            return {
                "status": "success",
                "message": "File does not exist or has already been cleared."
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing temporary dataset file: {str(e)}")
