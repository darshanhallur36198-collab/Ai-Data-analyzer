import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def predict_live(df: pd.DataFrame, target_col: str, feature_values: dict):
    """
    Executes live prediction based on user-entered feature values for a trained dataset.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}

    y_raw = df[target_col].dropna()
    num_unique = y_raw.nunique()
    is_numeric_dtype = pd.api.types.is_numeric_dtype(y_raw)

    problem_type = "classification" if not is_numeric_dtype or num_unique <= 10 or y_raw.dtype == 'bool' else "regression"

    feature_cols = [c for c in df.columns if c != target_col]
    usable_feature_cols = []
    for c in feature_cols:
        if df[c].nunique() == len(df) and (pd.api.types.is_object_dtype(df[c]) or "id" in c.lower()):
            continue
        usable_feature_cols.append(c)

    if not usable_feature_cols:
        return {"error": "No usable feature columns found."}

    df_model = df.dropna(subset=[target_col]).copy()

    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(df_model[target_col].astype(str))
    else:
        y_data = pd.to_numeric(df_model[target_col], errors="coerce")

    X_raw = df_model[usable_feature_cols].copy()
    X_encoded = pd.get_dummies(X_raw, drop_first=False)

    # Train model on full available clean dataset for max prediction accuracy
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_encoded, y_data)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_encoded, y_data)

    # Construct input row matching original features X_raw structure
    input_row_dict = {}
    for col in usable_feature_cols:
        val = feature_values.get(col, None)
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            try:
                input_row_dict[col] = float(val) if val is not None and str(val).strip() != "" else float(X_raw[col].mean())
            except Exception:
                input_row_dict[col] = float(X_raw[col].mean())
        else:
            input_row_dict[col] = str(val) if val is not None else str(X_raw[col].iloc[0])

    input_df = pd.DataFrame([input_row_dict])
    input_encoded = pd.get_dummies(input_df, drop_first=False)

    # Align input encoded columns with training columns
    input_final = pd.DataFrame(0, index=[0], columns=X_encoded.columns)
    for col in input_encoded.columns:
        if col in input_final.columns:
            input_final[col] = input_encoded[col]

    # Perform prediction
    if problem_type == "classification":
        pred_idx = model.predict(input_final)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0] if label_encoder else str(pred_idx)

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_final)[0]
            confidence = round(float(np.max(probs)) * 100.0, 1)

        return {
            "status": "success",
            "problem_type": "classification",
            "prediction": str(pred_label),
            "confidence": confidence,
            "target_column": target_col
        }

    else:
        pred_val = model.predict(input_final)[0]
        formatted_val = f"{round(float(pred_val), 2):,}"

        return {
            "status": "success",
            "problem_type": "regression",
            "prediction_value": round(float(pred_val), 2),
            "prediction_formatted": formatted_val,
            "target_column": target_col
        }
