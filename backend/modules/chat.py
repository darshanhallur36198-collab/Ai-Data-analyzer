import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# ─── Load environment from backend/.env ───────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ─── Centralized Model Configuration ──────────────────────────────────────────
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Ordered model fallback list
RAW_CANDIDATES = [
    PRIMARY_MODEL,
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Deduplicate candidates while maintaining exact priority order
MODEL_CANDIDATES = list(dict.fromkeys(RAW_CANDIDATES))

# Global memory cache for currently active working Gemini model
ACTIVE_GEMINI_MODEL = None


def get_current_ai_model() -> str:
    """Returns the current active working Gemini model name or primary model if configured."""
    global ACTIVE_GEMINI_MODEL
    return ACTIVE_GEMINI_MODEL or PRIMARY_MODEL


def is_ai_configured() -> bool:
    """Returns True if default backend GEMINI_API_KEY is configured."""
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _map_error(raw_error: str, is_custom_user_key: bool = False) -> str:
    """Maps technical Gemini API exceptions to clean, user-friendly messages."""
    err = str(raw_error).lower()
    if any(k in err for k in ["400", "401", "403", "api_key", "invalid", "unauthorized", "api_key_invalid"]):
        if is_custom_user_key:
            return "⚠️ Your Gemini API key is invalid. Remove it to use the default AI service."
        return "⚠️ The Gemini API key is invalid or unauthorized. Please check backend configuration."
    if any(k in err for k in ["429", "quota", "rate limit", "resource_exhausted"]):
        return "⚠️ AI request limit reached. Please try again shortly."
    return "⚠️ AI Assistant is temporarily unavailable. Please try again later."


def generate_gemini_response(prompt: str, api_key: str, is_custom_user_key: bool = False):
    """
    Executes Gemini API call using the official google-genai SDK.
    Handles automatic model fallback and memory caching of the working model.
    Returns: tuple (response_text, used_model_name, error_message)
    """
    global ACTIVE_GEMINI_MODEL

    if not api_key or not api_key.strip():
        err_msg = _map_error("invalid api_key", is_custom_user_key=is_custom_user_key)
        return None, None, err_msg

    clean_key = api_key.strip()

    try:
        client = genai.Client(api_key=clean_key)
    except Exception as e:
        return None, None, _map_error(str(e), is_custom_user_key=is_custom_user_key)

    # Build sequence of models to attempt: try cached ACTIVE_GEMINI_MODEL first if valid
    models_to_try = []
    if ACTIVE_GEMINI_MODEL and ACTIVE_GEMINI_MODEL in MODEL_CANDIDATES:
        models_to_try.append(ACTIVE_GEMINI_MODEL)
    for m in MODEL_CANDIDATES:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and hasattr(response, "text") and response.text:
                ACTIVE_GEMINI_MODEL = model_name
                return response.text, model_name, None
        except Exception as e:
            err_str = str(e)
            last_error = err_str
            err_lower = err_str.lower()

            # DO NOT fallback for API Key invalid, Quota/Rate limits, or Bad Requests!
            is_key_error = any(k in err_lower for k in ["400", "401", "403", "api_key", "invalid", "unauthorized", "api_key_invalid"])
            is_rate_error = any(k in err_lower for k in ["429", "quota", "rate limit", "resource_exhausted"])

            if is_key_error or is_rate_error:
                return None, model_name, _map_error(err_str, is_custom_user_key=is_custom_user_key)

            # Model 404 / Unavailable / Deprecated error -> clear cached active model if it failed & try next candidate
            if ACTIVE_GEMINI_MODEL == model_name:
                ACTIVE_GEMINI_MODEL = None

            continue

    return None, None, _map_error(last_error or "Service unavailable", is_custom_user_key=is_custom_user_key)


def chat_with_data(file_path: str, query: str, api_key: str = None, is_custom_user_key: bool = False) -> str:
    """
    Sends a dataset-context-aware question to Gemini and returns the AI response.
    - Sends compact dataset summaries, never full raw datasets.
    """
    key_to_use = (
        api_key.strip() if api_key and api_key.strip()
        else os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    if not key_to_use:
        return _map_error("invalid api_key", is_custom_user_key=is_custom_user_key)

    # Load dataset safely — only schema + stats, not all rows
    try:
        from backend.modules.data_loader import load_dataset
        df = load_dataset(file_path)
    except Exception as e:
        return f"⚠️ Could not load dataset for context: {str(e)}"

    # Build compact dataset context
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

    try:
        desc = df.describe(include="all").head(8).to_string()
        context_lines += ["Statistical Summary:", desc, ""]
    except Exception:
        pass

    try:
        missing = df.isnull().sum()
        missing_info = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
        if missing_info:
            context_lines += [f"Missing Values per Column: {missing_info}", ""]
    except Exception:
        pass

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

    text, used_model, err_msg = generate_gemini_response(prompt, key_to_use, is_custom_user_key=is_custom_user_key)
    if err_msg:
        return err_msg
    return text
