import numpy as np
import pandas as pd

def dataset_statistics(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame, cleaning_report: dict):
    """
    Computes summary metrics, dataset overview info, health score, and narrative insights.
    """
    num_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_cleaned.select_dtypes(exclude=['number']).columns.tolist()

    rows_before = cleaning_report["rows_before"]
    cols_count = cleaning_report["cols_count"]
    missing_before = cleaning_report["missing_before"]
    total_cells = rows_before * cols_count if (rows_before * cols_count) > 0 else 1

    # Real data quality / health calculation based on BEFORE cleaning state
    missing_pct = (missing_before / total_cells) * 100.0
    duplicate_penalty = (cleaning_report["duplicates_before"] / max(rows_before, 1)) * 10.0
    health_score = max(0.0, round(100.0 - missing_pct - duplicate_penalty, 1))

    summary_df = df_cleaned.describe(include='all').copy()
    # Convert summary dataframe to dictionary safely
    summary_dict = {}
    for col in summary_df.columns:
        summary_dict[col] = {}
        for idx in summary_df.index:
            val = summary_df.loc[idx, col]
            if pd.isna(val):
                summary_dict[col][idx] = None
            elif isinstance(val, (int, float, np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    summary_dict[col][idx] = None
                else:
                    summary_dict[col][idx] = round(float(val), 4)
            else:
                summary_dict[col][idx] = str(val)

    stats = {
        "rows": cleaning_report["rows_after"],
        "rows_before": rows_before,
        "rows_after": cleaning_report["rows_after"],
        "columns": cols_count,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "health_score": health_score,
        "missing": cleaning_report["missing_before_per_col"],
        "cleaning_report": cleaning_report,
        "summary": summary_dict,
        "insights": generate_narrative_insights(df_raw, df_cleaned, cleaning_report, health_score)
    }

    return stats


def generate_narrative_insights(df_raw, df_cleaned, cleaning_report, health_score):
    insights = []

    rows, cols = df_cleaned.shape
    missing_fixed = cleaning_report["missing_values_fixed"]
    dups_removed = cleaning_report["duplicates_removed"]

    # 1. Dataset Audit & Health
    insights.append(
        f"AI Audit: Dataset contains {rows} clean records across {cols} features. "
        f"Initial Data Quality Score is {health_score}%. "
        f"Repaired {missing_fixed} missing values and removed {dups_removed} duplicate rows."
    )

    # 2. Key Pattern Detection (Correlations)
    num_df = df_cleaned.select_dtypes(include=['number'])
    if len(num_df.columns) >= 2:
        corr = num_df.corr().stack().reset_index()
        corr.columns = ['v1', 'v2', 'val']
        strong = corr[(corr['v1'] != corr['v2']) & (corr['val'].abs() > 0.5)].sort_values(by='val', ascending=False)
        if not strong.empty:
            row = strong.iloc[0]
            sentiment = "positive" if row['val'] > 0 else "inverse"
            insights.append(
                f"Strategic Pattern: Key {sentiment} correlation ({round(row['val'], 2)}) identified between "
                f"'{row['v1']}' and '{row['v2']}'."
            )

    # 3. Categorical Dominance
    cat_df = df_cleaned.select_dtypes(exclude=['number'])
    if not cat_df.empty:
        top_cat_col = cat_df.columns[0]
        mode_val = df_cleaned[top_cat_col].mode()
        if not mode_val.empty:
            val = mode_val[0]
            pct = (df_cleaned[top_cat_col] == val).mean() * 100
            insights.append(
                f"Distribution Dominance: '{val}' is the primary value in '{top_cat_col}', representing "
                f"{round(pct, 1)}% of all entries."
            )

    # 4. Outlier Awareness
    if not num_df.empty:
        for col in num_df.columns[:3]:
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = df_cleaned[(df_cleaned[col] < (q1 - 1.5 * iqr)) | (df_cleaned[col] > (q3 + 1.5 * iqr))]
                if not outliers.empty:
                    insights.append(
                        f"Outlier Alert: Column '{col}' contains {len(outliers)} statistical outliers "
                        f"which may influence model variance."
                    )
                    break

    # 5. ML Recommendation
    insights.append(
        "Machine Learning Recommendation: Target column and problem type are automatically evaluated. "
        "Select your target feature in the Machine Learning tab to view model performance and live prediction options."
    )

    return insights