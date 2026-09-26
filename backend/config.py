# ─── Central Configuration for AI Data Analyst System ─────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

# Load .env from backend directory if present
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    # Fallback to root .env
    load_dotenv()

# ─── Gemini AI Configuration ───────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Ordered fallback chain if primary model is unavailable
GEMINI_FALLBACK_MODELS: list = [
    GEMINI_MODEL,
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

# ─── CORS Configuration ────────────────────────────────────────────────────────
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://ai-data-analyzer.vercel.app")
