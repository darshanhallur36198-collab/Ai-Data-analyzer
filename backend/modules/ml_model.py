import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_ml_model(df: pd.DataFrame, target_col: str = None):
    """
    Trains an ML model on df given a selected target_col.
    Automatically detects Classification vs Regression, calculates appropriate metrics,
    calculates feature importance, and returns feature schema for Live Prediction.
    """
    all_columns = df.columns.tolist()

    if not target_col or target_col not in all_columns:
        target_col = all_columns[-1]

    # Target series
    y_raw = df[target_col].dropna()

    if y_raw.empty:
        return {"error": f"Target column '{target_col}' contains no valid non-null values."}

    num_unique = y_raw.nunique()
    is_numeric_dtype = pd.api.types.is_numeric_dtype(y_raw)

    # ── Problem Type Detection ──────────────────────────────────────────────
    if not is_numeric_dtype or num_unique <= 10 or y_raw.dtype == 'bool':
        problem_type = "classification"
    else:
        problem_type = "regression"

    # Prepare features X and target y
    df_model = df.copy()

    # Drop target column from features
    feature_cols = [c for c in df_model.columns if c != target_col]

    # Filter out columns with 100% unique values like IDs or timestamps if high cardinality
    usable_feature_cols = []
    for c in feature_cols:
        if df_model[c].nunique() == len(df_model) and (pd.api.types.is_object_dtype(df_model[c]) or "id" in c.lower()):
            continue
        usable_feature_cols.append(c)

    if not usable_feature_cols:
        return {"error": "Not enough valid feature columns to train a model."}

    # Extract target y and encode if classification
    label_encoder = None
    class_names = []
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df_model[target_col].astype(str))
        class_names = [str(cls) for cls in label_encoder.classes_]
        y_data = y_encoded
    else:
        y_data = pd.to_numeric(df_model[target_col], errors="coerce")

    # Build X matrix with categorical dummy encoding
    X_raw = df_model[usable_feature_cols].copy()

    # Feature schema metadata for live prediction inputs
    feature_schema = []
    for col in usable_feature_cols:
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            mean_val = float(X_raw[col].mean()) if not pd.isna(X_raw[col].mean()) else 0.0
            min_val = float(X_raw[col].min()) if not pd.isna(X_raw[col].min()) else 0.0
            max_val = float(X_raw[col].max()) if not pd.isna(X_raw[col].max()) else 100.0
            feature_schema.append({
                "name": col,
                "type": "numeric",
                "default": round(mean_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2)
            })
        else:
            unique_opts = X_raw[col].astype(str).unique().tolist()[:20]
            mode_val = unique_opts[0] if unique_opts else ""
            feature_schema.append({
                "name": col,
                "type": "categorical",
                "default": mode_val,
                "options": unique_opts
            })

    # One-hot encoding for categorical features
    X_encoded = pd.get_dummies(X_raw, drop_first=False)

    # Clean combined dataframe to align X and y
    combined = X_encoded.copy()
    combined["_target_"] = y_data
    combined = combined.dropna(subset=["_target_"])

    if len(combined) < 5:
        return {"error": "Insufficient valid data samples remaining after data preparation."}

    X_final = combined.drop(columns=["_target_"])
    y_final = combined["_target_"]

    # Train / test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42,
            stratify=y_final if problem_type == "classification" and len(np.unique(y_final)) > 1 else None
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42
        )

    # ── Model Training & Evaluation ──────────────────────────────────────────
    if problem_type == "classification":
        model_name = "RandomForestClassifier"
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "class_names": class_names
        }

    else:
        model_name = "RandomForestRegressor"
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        safe_r2 = round(float(r2), 4) if not math.isnan(r2) and not math.isinf(r2) else 0.0

        metrics = {
            "r2_score": safe_r2,
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4)
        }

    # ── Feature Importance Calculation ──────────────────────────────────────
    importances = model.feature_importances_
    feature_names = X_final.columns.tolist()

    # Aggregate importance for one-hot encoded variables back to original feature names
    orig_importance = {}
    for orig_feat in usable_feature_cols:
        orig_importance[orig_feat] = 0.0

    for feat_name, imp in zip(feature_names, importances):
        matched = False
        for orig_feat in usable_feature_cols:
            if feat_name == orig_feat or feat_name.startswith(orig_feat + "_"):
                orig_importance[orig_feat] += float(imp)
                matched = True
                break
        if not matched:
            orig_importance[feat_name] = float(imp)

    # Convert to sorted array of percentages
    total_imp = sum(orig_importance.values()) if sum(orig_importance.values()) > 0 else 1.0
    feature_importance = []
    for feat, imp in sorted(orig_importance.items(), key=lambda x: x[1], reverse=True):
        pct = round((imp / total_imp) * 100.0, 1)
        feature_importance.append({
            "feature": feat.replace("_", " ").title(),
            "feature_raw": feat,
            "importance_pct": pct
        })

    return {
        "status": "success",
        "target_column": target_col,
        "all_columns": all_columns,
        "problem_type": problem_type,
        "model_type": model_name,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_schema": feature_schema,
        "sample_count": len(df_model)
    }
