import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# ─── Load environment from backend/.env ───────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ─── Central Model Configuration ──────────────────────────────────────────────
# Update GEMINI_MODEL here to change the model across the entire backend.
GEMINI_MODEL = "gemini-3.8-flash"

# Ordered fallback list: primary model first, then stable legacy options
GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

# ─── User-Friendly Error Messages ─────────────────────────────────────────────
def _friendly_error(raw_error: str) -> str:
    """Map technical Gemini API errors to friendly user-facing messages."""
    err = str(raw_error).lower()
    if "401" in err or "403" in err or "api_key_invalid" in err or "unauthorized" in err:
        return (
            "⚠️ AI Assistant is temporarily unavailable.\n"
            "The Gemini API key appears to be invalid or unauthorized.\n"
            "Please check that **GEMINI_API_KEY** is correctly set in **backend/.env** and restart the server."
        )
    if "404" in err or "not found" in err or "no longer available" in err:
        return (
            "⚠️ AI Assistant is temporarily unavailable.\n"
            "The configured Gemini model could not be reached (model may be deprecated).\n"
            "The system has attempted to fall back to an alternative model automatically."
        )
    if "429" in err or "quota" in err or "rate" in err:
        return (
            "⚠️ AI rate limit reached.\n"
            "You've sent too many requests in a short time. Please wait a moment and try again."
        )
    if "500" in err or "503" in err or "unavailable" in err:
        return (
            "⚠️ Gemini service is temporarily unavailable.\n"
            "Google's AI service may be experiencing issues. Please try again in a few minutes."
        )
    return (
        "⚠️ AI Assistant encountered an unexpected error.\n"
        "Please check the API configuration and try again."
    )


def chat_with_data(file_path: str, query: str, api_key: str = None) -> str:
    """
    Sends a dataset-context-aware question to Gemini and returns the AI response.
    - api_key: selected key passed from the endpoint (user key or server .env key).
    - Sends compact summaries, never full raw datasets.
    - Applies user-friendly error messages for common failure modes.
    """
    # Priority: passed api_key → environment variable → error
    key_to_use = (
        api_key.strip() if api_key and api_key.strip()
        else os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    if not key_to_use:
        return (
            "⚠️ AI Assistant is not configured.\n"
            "No API key was provided, and **GEMINI_API_KEY** is missing from **backend/.env**.\n"
            "Please provide a key in Settings or configure the server."
        )

    # Configure Gemini SDK
    try:
        genai.configure(api_key=key_to_use)
    except Exception as e:
        return _friendly_error(str(e))

    # Load dataset safely — only schema + stats, not all rows
    try:
        from backend.modules.data_loader import load_dataset
        df = load_dataset(file_path)
    except Exception as e:
        return f"⚠️ Could not load dataset for context: {str(e)}"

    # Build compact dataset context (no full dump)
    rows, cols = df.shape
    col_names = df.columns.tolist()
    col_dtypes = {col: str(df[col].dtype) for col in col_names}

    context_lines = [
        f"Dataset Overview:",
        f"  - Total Rows: {rows}",
        f"  - Total Columns: {cols}",
        f"  - Column Names: {col_names}",
        f"  - Data Types: {col_dtypes}",
        "",
    ]

    # Statistical summary (safe subset)
    try:
        desc = df.describe(include="all").head(8).to_string()
        context_lines += ["Statistical Summary:", desc, ""]
    except Exception:
        pass

    # Missing value counts
    try:
        missing = df.isnull().sum()
        missing_info = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
        if missing_info:
            context_lines += [f"Missing Values per Column: {missing_info}", ""]
    except Exception:
        pass

    # Sample data (top 5 rows only)
    try:
        sample_csv = df.head(5).to_csv(index=False)
        context_lines += ["Sample Data (First 5 Rows):", sample_csv, ""]
    except Exception:
        pass

    context_summary = "\n".join(context_lines)

    prompt = f"""You are an expert AI Data Analyst assistant helping a user understand their dataset.

{context_summary}

User's Question: "{query}"

Instructions:
- Answer clearly and concisely based strictly on the dataset context provided above.
- Use Markdown formatting: **bold**, bullet points, headers where helpful.
- Be professional, friendly, and precise.
- Do NOT include raw code blocks unless the user explicitly asks for code.
- If you cannot determine an answer from the provided summary, say so honestly.
"""

    # Try models in order, fall back gracefully
    last_error = None
    for model_name in GEMINI_FALLBACK_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and hasattr(response, "text") and response.text:
                return response.text
        except Exception as e:
            last_error = str(e)
            # Don't retry on key auth failures — they will fail for every model
            if any(code in str(e) for code in ["401", "403", "API_KEY_INVALID"]):
                break
            continue

    return _friendly_error(last_error or "Unknown error")
