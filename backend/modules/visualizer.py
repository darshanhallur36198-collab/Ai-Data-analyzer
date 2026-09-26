import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def fmt_num(n):
    """Convert large numbers to human readable strings: 4970000 -> '4.97M', etc."""
    try:
        n = float(n)
        if abs(n) >= 1_000_000_000:
            return f"{n/1_000_000_000:.1f}B"
        if abs(n) >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if abs(n) >= 1_000:
            return f"{n/1_000:.1f}K"
        return f"{n:,.0f}"
    except Exception:
        return str(n)


def generate_charts(df: pd.DataFrame):
    """
    Primary visualization generator for Plotly charts.
    Generates: Bar charts, Pie charts, Histograms, Box plots, Scatter plots, Heatmaps, and Line charts.
    Assigns _chart_type metadata to each chart: 'distribution', 'categorical', 'relationship', or 'other'.
    """
    charts = []
    COLORS = [
        "#2563eb", "#3b82f6", "#60a5fa", "#0284c7", "#0ea5e9",
        "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    def finalize(fig, title: str, subtitle: str = "", chart_type: str = "other", rotate_x: bool = False):
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{title}</b><br><sup style='color:#94a3b8; font-size:12px'>{subtitle}</sup>" if subtitle else f"<b>{title}</b>",
                x=0.02, xanchor="left",
                font=dict(size=16, color="#ffffff")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=90, r=40, b=90, t=70),
            font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#cbd5e1"),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="#1e293b",
                bordercolor="#334155",
                font_size=12,
                font_color="#ffffff",
                font_family="Inter, sans-serif"
            ),
            legend=dict(bgcolor="rgba(30,41,59,0.7)", borderwidth=0, font=dict(size=11)),
            autosize=True,
            height=460
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            automargin=True,
            tickfont=dict(size=12, color="#ffffff"),
            title_font=dict(size=13, color="#ffffff"),
            tickangle=-35 if rotate_x else 0
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            automargin=True,
            tickfont=dict(size=12, color="#ffffff"),
            title_font=dict(size=13, color="#ffffff"),
            tickformat=",",
            exponentformat="none"
        )
        result = json.loads(fig.to_json())
        result["_chart_type"] = chart_type
        return result

    # 1. CATEGORICAL: BAR CHARTS
    useful_cats = [c for c in cat_cols if 1 < df[c].nunique() <= 30]
    for cat_col in useful_cats[:3]:
        if numeric_cols:
            num_col = numeric_cols[0]
            grouped = (
                df.groupby(cat_col)[num_col]
                .sum()
                .reset_index()
                .sort_values(num_col, ascending=False)
                .head(10)
            )
            grouped.columns = [cat_col, "Value"]
            text_labels = [fmt_num(v) for v in grouped["Value"]]

            fig = go.Figure(go.Bar(
                x=grouped[cat_col].astype(str),
                y=grouped["Value"],
                text=text_labels,
                textposition="outside",
                textfont=dict(size=11, color="#ffffff"),
                marker_color=COLORS[0],
                marker_line_width=0,
                hovertemplate=f"<b>%{{x}}</b><br>{num_col}: %{{text}}<extra></extra>",
            ))
            fig.update_xaxes(title_text=cat_col.replace("_", " ").title())
            fig.update_yaxes(title_text=num_col.replace("_", " ").title())
            charts.append(finalize(
                fig,
                f"Top 10 {cat_col.replace('_', ' ').title()} by {num_col.replace('_', ' ').title()}",
                "Bar chart ranking",
                chart_type="categorical",
                rotate_x=True
            ))

        # Frequency Count Bar Chart
        counts = df[cat_col].value_counts().reset_index().head(10)
        counts.columns = [cat_col, "Count"]
        text_labels = [fmt_num(v) for v in counts["Count"]]

        fig = go.Figure(go.Bar(
            x=counts[cat_col].astype(str),
            y=counts["Count"],
            text=text_labels,
            textposition="outside",
            textfont=dict(size=11, color="#ffffff"),
            marker=dict(
                color=counts["Count"],
                colorscale="Blues",
                showscale=False,
                line_width=0
            ),
            hovertemplate="<b>%{x}</b><br>Count: %{text}<extra></extra>",
        ))
        fig.update_xaxes(title_text=cat_col.replace("_", " ").title())
        fig.update_yaxes(title_text="Count")
        charts.append(finalize(
            fig,
            f"Distribution of {cat_col.replace('_', ' ').title()}",
            "Category frequency count",
            chart_type="categorical",
            rotate_x=True
        ))

    # 2. CATEGORICAL: PIE / DONUT CHART
    for col in cat_cols:
        if 2 <= df[col].nunique() <= 8:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "Count"]
            fig = go.Figure(go.Pie(
                labels=counts[col].astype(str),
                values=counts["Count"],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(size=12),
                marker=dict(colors=COLORS, line=dict(color="rgba(0,0,0,0)", width=1)),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            ))
            charts.append(finalize(
                fig,
                f"Share of {col.replace('_', ' ').title()}",
                "Proportional breakdown",
                chart_type="categorical"
            ))
            break

    # 3. DISTRIBUTION: HISTOGRAMS
    for col in numeric_cols[:3]:
        if df[col].nunique() < 2:
            continue
        mean_v = df[col].mean()
        median_v = df[col].median()
        fig = px.histogram(
            df, x=col, nbins=25,
            color_discrete_sequence=[COLORS[1]]
        )
        fig.add_vline(
            x=mean_v, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Mean: {fmt_num(mean_v)}",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#f59e0b")
        )
        fig.add_vline(
            x=median_v, line_dash="dot", line_color="#10b981",
            annotation_text=f"Median: {fmt_num(median_v)}",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#10b981")
        )
        fig.update_xaxes(title_text=col.replace("_", " ").title())
        fig.update_yaxes(title_text="Frequency")
        charts.append(finalize(
            fig,
            f"Distribution of {col.replace('_', ' ').title()}",
            "Histogram with mean & median indicators",
            chart_type="distribution"
        ))

    # 4. DISTRIBUTION: BOX PLOTS
    if numeric_cols:
        cols_for_box = numeric_cols[:4]
        fig = go.Figure()
        for i, col in enumerate(cols_for_box):
            fig.add_trace(go.Box(
                y=df[col],
                name=col.replace("_", " ").title(),
                marker_color=COLORS[i % len(COLORS)],
                boxmean=True,
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
            ))
        charts.append(finalize(
            fig,
            "Outlier & Spread Overview",
            "Box plot distribution of key numerical variables",
            chart_type="distribution",
            rotate_x=True
        ))

    # 5. RELATIONSHIP: CORRELATION HEATMAP
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr().round(2)
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale="Blues",
            zmin=-1, zmax=1
        )
        charts.append(finalize(
            fig,
            "Correlation Heatmap",
            "Pairwise linear correlation matrix",
            chart_type="relationship",
            rotate_x=True
        ))

    # 6. RELATIONSHIP: SCATTER PLOT
    if len(numeric_cols) >= 2:
        for i in range(min(2, len(numeric_cols) - 1)):
            c1, c2 = numeric_cols[i], numeric_cols[i + 1]
            fig = px.scatter(
                df, x=c1, y=c2,
                opacity=0.7,
                color_discrete_sequence=[COLORS[0]]
            )
            fig.update_xaxes(title_text=c1.replace("_", " ").title())
            fig.update_yaxes(title_text=c2.replace("_", " ").title())
            charts.append(finalize(
                fig,
                f"{c1.replace('_', ' ').title()} vs {c2.replace('_', ' ').title()}",
                "Bivariate scatter plot",
                chart_type="relationship"
            ))

    # 7. RELATIONSHIP: LINE / TREND CHART
    date_col = None
    for col in cat_cols + numeric_cols:
        if any(k in col.lower() for k in ["date", "time", "year", "month", "day"]):
            date_col = col
            break

    if date_col and numeric_cols:
        try:
            tmp = df.copy()
            if tmp[date_col].dtype == "object":
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            trend = tmp.sort_values(date_col).groupby(date_col)[numeric_cols[0]].mean().reset_index()
            fig = px.line(
                trend, x=date_col, y=numeric_cols[0],
                markers=True,
                color_discrete_sequence=[COLORS[2]]
            )
            fig.update_xaxes(title_text=date_col.replace("_", " ").title())
            fig.update_yaxes(title_text=numeric_cols[0].replace("_", " ").title())
            charts.append(finalize(
                fig,
                f"Trend: {numeric_cols[0].replace('_', ' ').title()} over {date_col.replace('_', ' ').title()}",
                "Time series trend analysis",
                chart_type="relationship",
                rotate_x=True
            ))
        except Exception:
            pass

    return charts[:20]