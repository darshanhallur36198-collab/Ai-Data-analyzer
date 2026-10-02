import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Identifier column detector ─────────────────────────────────
ID_KEYWORDS = ["id", "code", "index", "serial", "rollno", "order_id", "studentid",
               "customerid", "userid", "no", "num", "seq", "number"]

def _is_id_column(col: str, series: pd.Series) -> bool:
    """Return True if the column looks like an identifier (skip for statistics)."""
    col_lower = col.lower().replace(" ", "").replace("_", "")
    if any(kw in col_lower for kw in ID_KEYWORDS):
        return True
    if series.dtype in [np.int64, np.int32, np.float64]:
        if series.nunique() / max(len(series), 1) > 0.9:
            return True
    return False


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
        return f"{n:,.2f}".rstrip('0').rstrip('.')
    except Exception:
        return str(n)


def generate_charts(df: pd.DataFrame):
    """
    Primary visualization generator. Generates:
      Categorical:   Bar-Ranking, Bar-Frequency, Pie/Donut
      Distribution:  Histogram, Box Plot
      Relationship:  Correlation Heatmap, Scatter Plot, Line/Trend
      Statistical:   Mean vs Median, Min/Max/Avg, Std-Dev
    Tags each result with _chart_type for frontend filtering.
    """
    charts = []
    COLORS = [
        "#2563eb", "#3b82f6", "#60a5fa", "#0284c7", "#0ea5e9",
        "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    # Filter out ID-like numeric columns from stats
    stat_numeric = [c for c in numeric_cols if not _is_id_column(c, df[c])]

    # ─── Helper: apply consistent readable layout ─────────────────
    def finalize(fig, title: str, subtitle: str = "", chart_type: str = "other", rotate_x: bool = False):
        clean_title = title.strip()
        subtitle_html = f"<sup style='color:#94a3b8; font-size:11px'>{subtitle}</sup>" if subtitle else ""
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"<b>{clean_title}</b><br>{subtitle_html}" if subtitle else f"<b>{clean_title}</b>",
                x=0.02, xanchor="left",
                font=dict(size=15, color="#0f172a")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.8)",
            margin=dict(l=90, r=40, b=90, t=70),
            font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#0f172a"),
            hovermode="closest",
            hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155",
                            font_size=12, font_color="#ffffff",
                            font_family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(248,250,252,0.9)", borderwidth=1,
                        bordercolor="#e2e8f0", font=dict(size=11)),
            autosize=True,
            height=460
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
            automargin=True,
            showticklabels=True,
            tickfont=dict(size=12, color="#0f172a"),
            title_font=dict(size=13, color="#0f172a"),
            tickangle=-35 if rotate_x else 0
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
            automargin=True,
            showticklabels=True,
            tickfont=dict(size=12, color="#0f172a"),
            title_font=dict(size=13, color="#0f172a"),
            tickformat=",",
            exponentformat="none"
        )
        result = json.loads(fig.to_json())
        result["_chart_type"] = chart_type
        return result

    # ═══════════════════════════════════════════════════
    # 1. CATEGORICAL: BAR CHARTS (Ranking + Frequency)
    # ═══════════════════════════════════════════════════
    useful_cats = [c for c in cat_cols if 1 < df[c].nunique() <= 30]
    for cat_col in useful_cats[:2]:
        # A – Ranking bar: category vs first numeric
        if stat_numeric:
            num_col = stat_numeric[0]
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
                textfont=dict(size=11, color="#0f172a"),
                marker_color=COLORS[0],
                marker_line_width=0,
                hovertemplate=f"<b>%{{x}}</b><br>{num_col}: %{{text}}<extra></extra>",
            ))
            cat_label = cat_col.replace("_", " ").title()
            num_label = num_col.replace("_", " ").title()
            fig.update_xaxes(title_text=cat_label)
            fig.update_yaxes(title_text=num_label)
            charts.append(finalize(
                fig,
                f"Top {len(grouped)} {cat_label} by {num_label}",
                "Ranking bar chart",
                chart_type="categorical",
                rotate_x=True
            ))

        # B – Frequency bar: category counts
        counts = df[cat_col].value_counts().reset_index().head(10)
        counts.columns = [cat_col, "Count"]
        text_labels = [fmt_num(v) for v in counts["Count"]]
        cat_label = cat_col.replace("_", " ").title()

        fig = go.Figure(go.Bar(
            x=counts[cat_col].astype(str),
            y=counts["Count"],
            text=text_labels,
            textposition="outside",
            textfont=dict(size=11, color="#0f172a"),
            marker=dict(
                color=counts["Count"],
                colorscale="Blues",
                showscale=False,
                line_width=0
            ),
            hovertemplate="<b>%{x}</b><br>Count: %{text}<extra></extra>",
        ))
        fig.update_xaxes(title_text=cat_label)
        fig.update_yaxes(title_text="Count")
        charts.append(finalize(
            fig,
            f"{cat_label} Frequency",
            "Category occurrence count",
            chart_type="categorical",
            rotate_x=True
        ))

    # ═══════════════════════════════════════════════════
    # 2. CATEGORICAL: PIE / DONUT CHART
    # ═══════════════════════════════════════════════════
    for col in cat_cols:
        if 2 <= df[col].nunique() <= 8:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "Count"]
            col_label = col.replace("_", " ").title()
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
                f"{col_label} Breakdown",
                "Proportional share by category",
                chart_type="categorical"
            ))
            break

    # ═══════════════════════════════════════════════════
    # 3. DISTRIBUTION: HISTOGRAMS
    # ═══════════════════════════════════════════════════
    for col in stat_numeric[:3]:
        if df[col].nunique() < 3:
            continue
        mean_v = df[col].mean()
        median_v = df[col].median()
        col_label = col.replace("_", " ").title()

        fig = px.histogram(df, x=col, nbins=25, color_discrete_sequence=[COLORS[1]])
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
        fig.update_xaxes(title_text=col_label)
        fig.update_yaxes(title_text="Frequency")
        charts.append(finalize(
            fig,
            f"{col_label} Distribution",
            "Histogram with Mean & Median markers",
            chart_type="distribution"
        ))

    # ═══════════════════════════════════════════════════
    # 4. DISTRIBUTION: BOX PLOTS
    # ═══════════════════════════════════════════════════
    box_cols = stat_numeric[:4]
    if box_cols:
        fig = go.Figure()
        for i, col in enumerate(box_cols):
            fig.add_trace(go.Box(
                y=df[col],
                name=col.replace("_", " ").title(),
                marker_color=COLORS[i % len(COLORS)],
                boxmean=True,
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
            ))
        charts.append(finalize(
            fig,
            "Spread & Outlier Overview",
            "Box plot — min, Q1, median, Q3, max, outliers",
            chart_type="distribution",
            rotate_x=True
        ))

    # ═══════════════════════════════════════════════════
    # 5. RELATIONSHIP: CORRELATION HEATMAP
    # ═══════════════════════════════════════════════════
    if len(stat_numeric) >= 2:
        corr_cols = stat_numeric[:10]  # cap at 10 columns for readability
        corr_df = df[corr_cols].corr().round(2)
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig.update_xaxes(tickangle=-35, tickfont=dict(size=11))
        fig.update_yaxes(tickfont=dict(size=11))
        charts.append(finalize(
            fig,
            "Correlation Heatmap",
            "Pearson correlation — range −1 to +1",
            chart_type="relationship",
            rotate_x=False
        ))

    # ═══════════════════════════════════════════════════
    # 6. RELATIONSHIP: SCATTER PLOTS
    # ═══════════════════════════════════════════════════
    if len(stat_numeric) >= 2:
        for i in range(min(2, len(stat_numeric) - 1)):
            c1, c2 = stat_numeric[i], stat_numeric[i + 1]
            fig = px.scatter(
                df, x=c1, y=c2,
                opacity=0.65,
                color_discrete_sequence=[COLORS[0]]
            )
            c1_label = c1.replace("_", " ").title()
            c2_label = c2.replace("_", " ").title()
            fig.update_xaxes(title_text=c1_label)
            fig.update_yaxes(title_text=c2_label)
            charts.append(finalize(
                fig,
                f"{c1_label} vs {c2_label}",
                "Bivariate scatter plot",
                chart_type="relationship"
            ))

    # ═══════════════════════════════════════════════════
    # 7. RELATIONSHIP: LINE / TREND CHART
    # ═══════════════════════════════════════════════════
    DATE_KEYWORDS = ["date", "time", "year", "month", "day", "created_at",
                     "timestamp", "period", "week", "quarter"]
    date_col = None
    for col in cat_cols + numeric_cols:
        if any(k in col.lower() for k in DATE_KEYWORDS):
            date_col = col
            break

    if date_col and stat_numeric:
        try:
            tmp = df.copy()
            if tmp[date_col].dtype == "object":
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
            trend_col = stat_numeric[0]
            trend = tmp.sort_values(date_col).groupby(date_col)[trend_col].mean().reset_index()
            if len(trend) >= 3:
                fig = px.line(trend, x=date_col, y=trend_col,
                              markers=True, color_discrete_sequence=[COLORS[2]])
                date_label = date_col.replace("_", " ").title()
                trend_label = trend_col.replace("_", " ").title()
                fig.update_xaxes(title_text=date_label)
                fig.update_yaxes(title_text=trend_label)
                charts.append(finalize(
                    fig,
                    f"{trend_label} Trend Over {date_label}",
                    "Time-series trend analysis",
                    chart_type="relationship",
                    rotate_x=True
                ))
        except Exception:
            pass

    # ═══════════════════════════════════════════════════
    # 8. STATISTICAL: MEAN vs MEDIAN COMPARISON
    # ═══════════════════════════════════════════════════
    if len(stat_numeric) >= 2:
        cols_stat = stat_numeric[:6]
        col_labels = [c.replace("_", " ").title() for c in cols_stat]
        means = [round(df[c].mean(), 2) for c in cols_stat]
        medians = [round(df[c].median(), 2) for c in cols_stat]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Mean", x=col_labels, y=means,
            marker_color=COLORS[0],
            text=[fmt_num(v) for v in means],
            textposition="outside",
            textfont=dict(size=10, color="#0f172a"),
            hovertemplate="<b>%{x}</b><br>Mean: %{y}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            name="Median", x=col_labels, y=medians,
            marker_color=COLORS[5],
            text=[fmt_num(v) for v in medians],
            textposition="outside",
            textfont=dict(size=10, color="#0f172a"),
            hovertemplate="<b>%{x}</b><br>Median: %{y}<extra></extra>"
        ))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Feature")
        fig.update_yaxes(title_text="Value")
        charts.append(finalize(
            fig,
            "Mean vs Median Comparison",
            "Statistical central tendency per numeric feature",
            chart_type="statistical",
            rotate_x=True
        ))

    # ═══════════════════════════════════════════════════
    # 9. STATISTICAL: MIN / AVERAGE / MAX SUMMARY
    # ═══════════════════════════════════════════════════
    if len(stat_numeric) >= 2:
        cols_stat = stat_numeric[:6]
        col_labels = [c.replace("_", " ").title() for c in cols_stat]
        mins = [round(df[c].min(), 2) for c in cols_stat]
        avgs = [round(df[c].mean(), 2) for c in cols_stat]
        maxs = [round(df[c].max(), 2) for c in cols_stat]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Min", x=col_labels, y=mins,
            marker_color="#10b981",
            hovertemplate="<b>%{x}</b><br>Min: %{y}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            name="Average", x=col_labels, y=avgs,
            marker_color="#2563eb",
            hovertemplate="<b>%{x}</b><br>Avg: %{y}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            name="Max", x=col_labels, y=maxs,
            marker_color="#ef4444",
            hovertemplate="<b>%{x}</b><br>Max: %{y}<extra></extra>"
        ))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Feature")
        fig.update_yaxes(title_text="Value")
        charts.append(finalize(
            fig,
            "Min / Average / Max Summary",
            "Range summary for numeric features",
            chart_type="statistical",
            rotate_x=True
        ))

    # ═══════════════════════════════════════════════════
    # 10. STATISTICAL: STANDARD DEVIATION CHART
    # ═══════════════════════════════════════════════════
    if len(stat_numeric) >= 2:
        cols_stat = stat_numeric[:8]
        col_labels = [c.replace("_", " ").title() for c in cols_stat]
        stds = [round(df[c].std(), 4) for c in cols_stat]

        fig = go.Figure(go.Bar(
            x=col_labels,
            y=stds,
            text=[fmt_num(v) for v in stds],
            textposition="outside",
            textfont=dict(size=11, color="#0f172a"),
            marker=dict(
                color=stds,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Std Dev", thickness=12)
            ),
            hovertemplate="<b>%{x}</b><br>Std Dev: %{y:.4f}<extra></extra>"
        ))
        fig.update_xaxes(title_text="Feature")
        fig.update_yaxes(title_text="Standard Deviation")
        charts.append(finalize(
            fig,
            "Standard Deviation by Feature",
            "Higher values indicate greater variability",
            chart_type="statistical",
            rotate_x=True
        ))

    return charts[:20]