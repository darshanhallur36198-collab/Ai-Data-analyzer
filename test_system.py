import os
import sys
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_full_pipeline():
    print("=== STARTING AUTOMATED SYSTEM VERIFICATION TESTS ===")

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    print("✅ Health Check Endpoint: PASS")

    # 2. Upload Binary Classification Dataset (with missing values & duplicate rows)
    test_csv_path = "datasets/test_churn_classification.csv"
    with open(test_csv_path, "rb") as f:
        res = client.post("/upload", files={"file": ("test_churn_classification.csv", f, "text/csv")})

    assert res.status_code == 200, f"Upload failed: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    print("✅ File Upload Endpoint: PASS")

    # Verify Priority 2: Data Cleaning Statistics Before vs After
    cleaning = data["analysis"]["cleaning_report"]
    print(f"   - Rows Before: {cleaning['rows_before']}, Rows After: {cleaning['rows_after']}")
    print(f"   - Missing Before: {cleaning['missing_before']}, Fixed: {cleaning['missing_values_fixed']}")
    print(f"   - Duplicates Before: {cleaning['duplicates_before']}, Removed: {cleaning['duplicates_removed']}")
    assert cleaning['missing_values_fixed'] > 0, "Missing values repaired count should be > 0"
    assert cleaning['duplicates_removed'] > 0, "Duplicate rows removed count should be > 0"
    assert cleaning['rows_before'] > cleaning['rows_after'], "Row count should adjust after duplicate drop"
    print("✅ Priority 2 - Before vs After Data Cleaning Report: PASS")

    # Verify Priority 1: Plotly Charts Generation
    charts = data["charts"]
    assert len(charts) > 0, "Charts list should not be empty"
    chart_types = set(c.get("_chart_type") for c in charts)
    print(f"   - Generated {len(charts)} Plotly charts across types: {chart_types}")
    print("✅ Priority 1 - Chart Generation: PASS")

    # Verify Priority 3 & 4 & 5 & 6: ML Classification Target, Metrics & Feature Importance
    ml = data["ml_prediction"]
    assert ml["problem_type"] == "classification"
    assert ml["model_type"] == "RandomForestClassifier"
    assert "accuracy" in ml["metrics"]
    assert "precision" in ml["metrics"]
    assert "recall" in ml["metrics"]
    assert "f1_score" in ml["metrics"]
    assert "confusion_matrix" in ml["metrics"]
    assert len(ml["feature_importance"]) > 0
    print(f"   - Detected ML Problem Type: {ml['problem_type']}")
    print(f"   - Model Used: {ml['model_type']}")
    print(f"   - Classification Accuracy: {ml['metrics']['accuracy']*100}%")
    print(f"   - Feature Importance: {ml['feature_importance'][:3]}")
    print("✅ Priority 3, 4, 5, 6 - ML Classification & Feature Importance: PASS")

    # Verify Priority 7: Live Prediction Calculator (Classification)
    file_path = data["file_path"]
    pred_res = client.post("/predict", json={
        "file_path": file_path,
        "target_column": "churned",
        "feature_values": {
            "age": 30,
            "monthly_spend": 120.0,
            "tenure_months": 6,
            "support_tickets": 4,
            "contract_type": "Month-to-Month"
        }
    })
    assert pred_res.status_code == 200, f"Predict failed: {pred_res.text}"
    pred_data = pred_res.json()
    assert pred_data["status"] == "success"
    assert "prediction" in pred_data
    assert "confidence" in pred_data
    print(f"   - Live Classification Prediction Output: {pred_data['prediction']} (Confidence: {pred_data['confidence']}%)")
    print("✅ Priority 7 - Live Prediction Calculator (Classification): PASS")

    # Verify Target Column Switching to Regression (Priority 3 & 4 Regression Test)
    train_res = client.post("/train-ml", json={
        "file_path": file_path,
        "target_column": "monthly_spend"
    })
    assert train_res.status_code == 200
    reg_ml = train_res.json()["ml_prediction"]
    assert reg_ml["problem_type"] == "regression"
    assert reg_ml["model_type"] == "RandomForestRegressor"
    assert "r2_score" in reg_ml["metrics"]
    assert "mae" in reg_ml["metrics"]
    assert "rmse" in reg_ml["metrics"]
    print(f"   - Switched Target to 'monthly_spend' -> Detected Regression ({reg_ml['model_type']})")
    print(f"   - Regression Metrics: R²={reg_ml['metrics']['r2_score']}, MAE={reg_ml['metrics']['mae']}, RMSE={reg_ml['metrics']['rmse']}")
    print("✅ Priority 3 & 4 - Regression Detection & Metrics: PASS")

    # Verify Priority 7: Live Prediction Calculator (Regression)
    pred_reg_res = client.post("/predict", json={
        "file_path": file_path,
        "target_column": "monthly_spend",
        "feature_values": {
            "age": 35,
            "tenure_months": 24,
            "support_tickets": 1,
            "contract_type": "One-Year",
            "churned": "No"
        }
    })
    assert pred_reg_res.status_code == 200
    pred_reg_data = pred_reg_res.json()
    assert pred_reg_data["status"] == "success"
    print(f"   - Live Regression Prediction Output: {pred_reg_data['prediction_formatted']}")
    print("✅ Priority 7 - Live Prediction Calculator (Regression): PASS")

    # Verify Priority 10: Path Traversal Protection on /clear
    bad_clear_res = client.post("/clear", json={"file_path": "../../etc/passwd"})
    assert bad_clear_res.status_code == 400
    print("✅ Priority 10 - Path Traversal Protection: PASS")

    # Verify Priority 10 & 11: Normal File Clear
    good_clear_res = client.post("/clear", json={"file_path": file_path})
    assert good_clear_res.status_code == 200
    assert not os.path.exists(file_path)
    print("✅ Priority 10 & 11 - Safe File Deletion & Clear Endpoint: PASS")

    print("\n🎉 ALL BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    test_full_pipeline()
