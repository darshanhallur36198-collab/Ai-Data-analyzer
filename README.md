# 📊 AI Data Analyzer — AI-Assisted Automated Data Analysis, Visualization & Prediction System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=flat-square&logo=plotly)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Google Gemini](https://img.shields.io/badge/AI-Google_Gemini-4285F4?style=flat-square&logo=google)](https://aistudio.google.com/)

An end-to-end web platform for automated dataset analysis, cleaning, interactive visualization, machine learning prediction, and conversational AI insights.

---

## 🏗️ Architecture & Deployment Flow

```text
       User (Browser)
             │
             ▼
 ┌──────────────────────┐
 │   Vercel Frontend    │  (HTML5 / CSS3 / JavaScript / Plotly.js)
 └───────────┬──────────┘
             │ REST API Calls (HTTPS)
             ▼
 ┌──────────────────────┐
 │    Render Backend    │  (FastAPI / Python 3.10+)
 └───────────┬──────────┘
             ├───────────────────────┬───────────────────────┐
             ▼                       ▼                       ▼
   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
   │ Pandas Data Engine│   │ Scikit-Learn ML   │   │ Gemini AI API     │
   └───────────────────┘   └───────────────────┘   └───────────────────┘
```

---

## ✨ Features

- 🚀 **Professional Welcome Landing Page**: Modern branding, responsive logo, and seamless entry flow.
- 📁 **Multi-Format Dataset Upload**: Accepts `.csv`, `.xlsx`, `.xls`, `.json`, `.parquet`, and `.tsv` files up to 50MB.
- 🧹 **Automated Data Cleaning**: Detects missing values, duplicates, and column anomalies with **Before vs After** comparative statistics.
- 📊 **Interactive Plotly Visualization Suite**: Auto-generates vertical bar charts, scatter plots, line charts, and box plots with horizontal/vertical orientation toggling.
- 🤖 **Automated Machine Learning Engine**: Auto-detects Classification vs Regression based on target column characteristics, displays model evaluation metrics (R², RMSE, Accuracy, F1-Score), and computes feature importances.
- 🎯 **Interactive Live Prediction Calculator**: Generates dynamic input forms based on dataset features to run live predictions.
- 💬 **Dataset-Aware AI Assistant (Gemini 3.8 Flash)**: Natural language conversational interface that answers questions strictly based on summary statistics without exposing raw data.
- 📑 **Comprehensive Report Export**: Single-click export of data analysis reports in JSON or formatted TXT.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+), Plotly.js |
| **Backend** | FastAPI, Uvicorn, Pydantic, Python-Multipart |
| **Data & ML** | Pandas, NumPy, Scikit-Learn, OpenPyXL |
| **AI Integration** | Google Generative AI SDK (`google-generativeai`), Gemini 3.8 Flash |
| **Hosting** | Vercel (Frontend), Render (Backend) |

---

## ⚡ Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Node.js (or any static HTTP server)

### 1. Backend Setup
```bash
# Navigate to project root
cd new_project_dataVisual_working

# Create and activate virtual environment (optional)
python -m venv backend/venv
# Windows: backend\venv\Scripts\activate
# Linux/Mac: source backend/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example backend/.env
# Edit backend/.env and add your GEMINI_API_KEY from https://aistudio.google.com/apikey

# Start FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup
In a second terminal window:
```bash
python -m http.server 3000 --directory frontend
```

Open your browser at: **`http://127.0.0.1:3000`**

---

## 🔑 Environment Variables

Set the following variables in `backend/.env` (Local) or **Render Dashboard** (Production):

| Variable | Description | Default / Example |
|---|---|---|
| `GEMINI_API_KEY` | **Required**: Google Gemini API key | `AIzaSy...` |
| `GEMINI_MODEL` | Gemini Model Identifier | `gemini-3.8-flash` |
| `FRONTEND_URL` | Production Vercel Frontend URL for CORS | `https://ai-data-analyzer.vercel.app` |

---

## 🚀 Deployment Instructions

### Phase 1: Deploy Backend on Render

1. Push repository to **GitHub**.
2. Log into [Render Dashboard](https://dashboard.render.com/) and create a **New Web Service**.
3. Connect your GitHub repository.
4. Configure settings:
   - **Name**: `ai-data-analyzer-api`
   - **Environment**: `Python 3`
   - **Region**: Choose closest to target audience
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add **Environment Variables** in Render:
   - `GEMINI_API_KEY` = *your_real_gemini_api_key*
   - `GEMINI_MODEL` = `gemini-3.8-flash`
   - `FRONTEND_URL` = `https://<your-app-name>.vercel.app`
6. Click **Deploy Web Service** and note your backend API URL (e.g., `https://ai-data-analyzer-api.onrender.com`).

---

### Phase 2: Deploy Frontend on Vercel

1. Log into [Vercel Dashboard](https://vercel.com/) and click **Add New Project**.
2. Import your GitHub repository.
3. Configure settings:
   - **Framework Preset**: `Other` / `Static HTML`
   - **Root Directory**: `frontend`
   - **Build Command**: *(Leave empty)*
   - **Output Directory**: `.`
4. Click **Deploy**.
5. Once deployed, copy your Vercel URL (e.g., `https://ai-data-analyzer.vercel.app`).
6. Ensure `vercel.json` contains the reverse proxy rewrite for `/api/(.*)` pointing to your Render backend URL.
7. Commit and push changes to GitHub.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root API status & model details |
| `GET` | `/health` | Server health check endpoint |
| `GET` | `/ai-status` | Gemini AI configuration status check |
| `POST` | `/upload` | Uploads dataset file, cleans data, generates charts & trains default ML model |
| `POST` | `/train-ml` | Re-trains model on a specific user-selected target column |
| `POST` | `/predict` | Computes live machine learning prediction for custom feature inputs |
| `POST` | `/chat` | Conversational dataset querying via Gemini 3.8 Flash |
| `POST` | `/clear` | Safely removes uploaded temporary dataset files from server |

---

## 📌 Limitations & Future Enhancements

- **Temporary Storage**: Render free-tier instances use ephemeral filesystems. Uploaded datasets are treated as temporary processing sessions.
- **Cold Starts**: Render free instances sleep after 15 minutes of inactivity; initial startup might take 30–50 seconds.
- **Future Enhancements**: Integration of persistent database logging (PostgreSQL / MongoDB), automated PDF report generation, and time-series forecasting.

---

## 📄 License

Developed for academic demonstration as a Final-Year Project. All rights reserved.