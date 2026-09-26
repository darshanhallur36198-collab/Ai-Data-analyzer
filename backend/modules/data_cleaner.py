import pandas as pd

def clean_dataset(df: pd.DataFrame):
    """
    Cleans dataset and records statistics BEFORE and AFTER cleaning.
    Returns:
        df_cleaned: Cleaned DataFrame
        cleaning_report: dict containing before/after statistics
    """
    df_raw = df.copy()

    rows_before = int(df_raw.shape[0])
    cols_before = int(df_raw.shape[1])
    missing_before_dict = {col: int(df_raw[col].isnull().sum()) for col in df_raw.columns}
    missing_before_total = int(sum(missing_before_dict.values()))
    duplicates_before = int(df_raw.duplicated().sum())

    # Perform cleaning
    df_cleaned = df_raw.drop_duplicates()
    duplicates_removed = duplicates_before

    column_details = []

    for column in df_cleaned.columns:
        col_missing_before = missing_before_dict[column]
        strategy = "No changes needed"

        if pd.api.types.is_numeric_dtype(df_cleaned[column]):
            if col_missing_before > 0:
                median_val = df_cleaned[column].median()
                fill_val = median_val if not pd.isna(median_val) else 0
                df_cleaned[column] = df_cleaned[column].fillna(fill_val)
                strategy = f"Imputed missing with median ({round(float(fill_val), 2)})"
        else:
            if col_missing_before > 0:
                mode_series = df_cleaned[column].mode()
                mode_val = mode_series[0] if not mode_series.empty else "Unknown"
                df_cleaned[column] = df_cleaned[column].fillna(mode_val)
                strategy = f"Imputed missing with mode ('{mode_val}')"

        col_missing_after = int(df_cleaned[column].isnull().sum())

        column_details.append({
            "column": column,
            "missing_before": col_missing_before,
            "missing_after": col_missing_after,
            "fixed": col_missing_before - col_missing_after,
            "strategy": strategy,
            "dtype": str(df_cleaned[column].dtype)
        })

    rows_after = int(df_cleaned.shape[0])
    missing_after_dict = {col: int(df_cleaned[col].isnull().sum()) for col in df_cleaned.columns}
    missing_after_total = int(sum(missing_after_dict.values()))
    missing_values_fixed = missing_before_total - missing_after_total

    cleaning_report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_count": cols_before,
        "missing_before": missing_before_total,
        "missing_after": missing_after_total,
        "missing_values_fixed": missing_values_fixed,
        "duplicates_before": duplicates_before,
        "duplicates_removed": duplicates_removed,
        "missing_before_per_col": missing_before_dict,
        "column_details": column_details
    }

    return df_cleaned, cleaning_report