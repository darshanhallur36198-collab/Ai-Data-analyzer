import json
import os

def generate_report_content(analysis: dict, ml_results: dict = None):
    """
    Generates a structured text and JSON analysis report.
    """
    lines = []
    lines.append("==========================================================================")
    lines.append("           AI AUTOMATED DATA ANALYSIS & PREDICTION REPORT                 ")
    lines.append("==========================================================================")
    lines.append("")

    # 1. Dataset Overview
    lines.append("1. DATASET OVERVIEW & HEALTH")
    lines.append("--------------------------------------------------------------------------")
    lines.append(f"Total Clean Rows: {analysis.get('rows_after', analysis.get('rows', 'N/A'))}")
    lines.append(f"Total Columns   : {analysis.get('columns', 'N/A')}")
    lines.append(f"Data Health Score: {analysis.get('health_score', 'N/A')}%")
    lines.append(f"Numerical Cols  : {', '.join(analysis.get('numerical_cols', [])) or 'None'}")
    lines.append(f"Categorical Cols: {', '.join(analysis.get('categorical_cols', [])) or 'None'}")
    lines.append("")

    # 2. Data Cleaning Summary
    clean_rep = analysis.get('cleaning_report', {})
    lines.append("2. DATA CLEANING & REPAIR SUMMARY")
    lines.append("--------------------------------------------------------------------------")
    lines.append(f"Rows Before Cleaning: {clean_rep.get('rows_before', 'N/A')}")
    lines.append(f"Rows After Cleaning : {clean_rep.get('rows_after', 'N/A')}")
    lines.append(f"Missing Values Fixed: {clean_rep.get('missing_values_fixed', 'N/A')}")
    lines.append(f"Duplicates Removed  : {clean_rep.get('duplicates_removed', 'N/A')}")
    lines.append("")

    # 3. Machine Learning Evaluation
    if ml_results and not ml_results.get("error"):
        lines.append("3. MACHINE LEARNING MODEL PERFORMANCE")
        lines.append("--------------------------------------------------------------------------")
        lines.append(f"Target Column: {ml_results.get('target_column')}")
        lines.append(f"Problem Type : {ml_results.get('problem_type', '').upper()}")
        lines.append(f"Model Used   : {ml_results.get('model_type')}")
        lines.append("")

        metrics = ml_results.get('metrics', {})
        lines.append("Evaluation Metrics:")
        for k, v in metrics.items():
            if k != "confusion_matrix" and k != "class_names":
                lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
        lines.append("")

        lines.append("Top Feature Importances:")
        for item in ml_results.get('feature_importance', [])[:5]:
            lines.append(f"  - {item['feature']}: {item['importance_pct']}%")
        lines.append("")

    # 4. AI Strategic Insights
    insights = analysis.get('insights', [])
    if insights:
        lines.append("4. AI STRATEGIC INSIGHTS")
        lines.append("--------------------------------------------------------------------------")
        for idx, ins in enumerate(insights, 1):
            lines.append(f" [{idx}] {ins}")
        lines.append("")

    lines.append("==========================================================================")
    lines.append("                     END OF AUTOMATED REPORT                              ")
    lines.append("==========================================================================")

    return "\n".join(lines)