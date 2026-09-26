/* ═══════════════════════════════════════════════════════════════
   AI Autonomous Data Analyst & Prediction System — app.js
   ═══════════════════════════════════════════════════════════════ */

let analysisData = null;

// ─── Welcome Page Navigation ───────────────────────────────────
function startApp() {
    const welcomePage = document.getElementById('welcome-page');
    const appContainer = document.getElementById('app-container');

    if (welcomePage) {
        welcomePage.classList.add('hide');
        setTimeout(() => {
            welcomePage.style.display = 'none';
            if (appContainer) {
                appContainer.classList.remove('app-hidden');
            }
        }, 300);
    }
}

// ─── Central API URL Configuration ──────────────────────────────
// Replace this with your deployed Render backend URL after creating the Render Web Service:
const PRODUCTION_API_URL = 'https://ai-data-analyzer-api.onrender.com';

function getSettings() {
    let url = localStorage.getItem('api_url');
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

    // Purge stale local API URLs from browser localStorage when running in production
    if (!isLocalhost && url && (url.includes('localhost') || url.includes('127.0.0.1'))) {
        localStorage.removeItem('api_url');
        url = null;
    }

    if (!url) {
        url = isLocalhost ? 'http://127.0.0.1:8000' : PRODUCTION_API_URL;
    }
    return {
        apiUrl: url.replace(/\/+$/, ''),
        maxCharts: parseInt(localStorage.getItem('max_charts') || '12'),
        theme: localStorage.getItem('chart_theme') || 'plotly_dark'
    };
}

function saveSettings() {
    localStorage.setItem('max_charts', document.getElementById('setting-max-charts').value);
    localStorage.setItem('chart_theme', document.getElementById('setting-theme').value);
    localStorage.setItem('gemini_key', document.getElementById('setting-gemini-key').value.trim());

    if (analysisData && analysisData.charts) {
        populateDashboardCharts(analysisData.charts);
        populateGraphsSection(analysisData.charts);
    }

    const msg = document.getElementById('settings-msg');
    msg.style.display = 'block';
    setTimeout(() => msg.style.display = 'none', 2500);
}

// ─── Section Meta & Navigation ─────────────────────────────────
const SECTION_META = {
    'section-dashboard': { title: 'AI Dataset Analytics', sub: 'Upload a dataset to begin automated data processing and machine learning' },
    'section-overview': { title: 'Data Overview', sub: 'Inspect dataset shape, columns, and quality statistics' },
    'section-cleaning': { title: 'Data Cleaning Report', sub: 'Compare statistics before vs after automatic data cleaning' },
    'section-visualizations': { title: 'Visualization Suite', sub: 'Interactive Plotly charts generated automatically from your dataset' },
    'section-ml': { title: 'Machine Learning Model', sub: 'Select target column, view classification or regression metrics and feature importance' },
    'section-predict': { title: 'Live Prediction Calculator', sub: 'Input custom feature values for live real-time model predictions' },
    'section-chat': { title: 'AI Assistant', sub: 'Ask natural language questions about your dataset powered by Gemini' },
    'section-reports': { title: 'Report & Export', sub: 'Download comprehensive statistical analysis report' },
    'section-settings': { title: 'Settings', sub: 'Configure API key, max charts, and chart themes' },
};

function initNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active-section'));
            const targetId = item.dataset.section;
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add('active-section');
            }

            const meta = SECTION_META[targetId];
            if (meta) {
                document.getElementById('page-title').textContent = meta.title;
                document.getElementById('page-subtitle').textContent = meta.sub;
            }

            // Trigger Plotly resize for visible charts
            setTimeout(() => {
                if (targetSection) {
                    targetSection.querySelectorAll('.plotly-box').forEach(el => {
                        if (el && el.data) {
                            Plotly.Plots.resize(el);
                        }
                    });
                }
            }, 60);
        });
    });
}

// ─── File Upload Binding ───────────────────────────────────────
function initFileInputs() {
    const input = document.getElementById('dataset-dash');
    const nameEl = document.getElementById('file-name-dash');
    const zone = document.getElementById('drop-zone-dash');

    if (!input) return;

    input.addEventListener('change', e => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            if (nameEl) nameEl.textContent = `📄 Selected File: ${file.name}`;
            runAnalysis(file);
            input.value = '';
        }
    });

    if (zone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
            zone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); })
        );
        ['dragenter', 'dragover'].forEach(e => zone.addEventListener(e, () => zone.classList.add('dragover')));
        ['dragleave', 'drop'].forEach(e => zone.addEventListener(e, () => zone.classList.remove('dragover')));

        zone.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                if (nameEl) nameEl.textContent = `📄 Selected File: ${files[0].name}`;
                runAnalysis(files[0]);
            }
        });
    }
}

// ─── Progress Animation Helper ─────────────────────────────────
const STEPS = [
    'Uploading dataset...',
    'Analyzing columns & types...',
    'Cleaning dataset...',
    'Generating visualizations...',
    'Training machine-learning model...',
    'Preparing results...'
];

function setProgress(pct, label) {
    const bar = document.getElementById('progress-bar-dash');
    const lbl = document.getElementById('progress-label-dash');
    if (bar) bar.style.width = pct + '%';
    if (lbl) lbl.textContent = label;
}

function showProgress() {
    const wrap = document.getElementById('progress-wrap-dash');
    if (wrap) wrap.style.display = 'block';
    document.getElementById('kpi-box').style.display = 'none';
    document.getElementById('ai-assistant').style.display = 'none';
    document.getElementById('chart-container').innerHTML = '';
    setProgress(5, STEPS[0]);
}

function hideProgress() {
    const wrap = document.getElementById('progress-wrap-dash');
    if (wrap) wrap.style.display = 'none';
}

function animateProgress(startPct, endPct, duration, stepIdx) {
    return new Promise(resolve => {
        const fps = 30;
        const steps = (duration / 1000) * fps;
        const stepSize = (endPct - startPct) / steps;
        let current = startPct;

        const interval = setInterval(() => {
            current += stepSize;
            setProgress(Math.min(current, endPct), STEPS[Math.min(stepIdx, STEPS.length - 1)]);
            if (current >= endPct) { clearInterval(interval); resolve(); }
        }, 1000 / fps);
    });
}

// ─── Main Analysis Pipeline ────────────────────────────────────
async function runAnalysis(file) {
    const settings = getSettings();
    const formData = new FormData();
    formData.append('file', file);

    showProgress();

    const progressPromise = (async () => {
        await animateProgress(5, 25, 600, 0);
        await animateProgress(25, 45, 800, 1);
        await animateProgress(45, 65, 800, 2);
        await animateProgress(65, 85, 700, 3);
        await animateProgress(85, 95, 500, 4);
    })();

    let result;
    try {
        const response = await fetch(`${settings.apiUrl}/upload`, { method: 'POST', body: formData });
        result = await response.json();

        await progressPromise;

        if (!response.ok || result.status === 'error') {
            throw new Error(result.detail || result.message || 'Data processing failed');
        }
    } catch (err) {
        hideProgress();
        console.error("Analysis Error:", err);
        showError(err.message === 'Failed to fetch'
            ? "Could not connect to the FastAPI backend server. Ensure backend is running at " + settings.apiUrl
            : err.message);
        return;
    }

    setProgress(100, STEPS[5]);
    await new Promise(r => setTimeout(r, 400));
    hideProgress();

    analysisData = result;

    // Populate UI Sections
    populateKPIs(result.analysis);
    populateInsights(result.analysis.insights);
    populateDashboardCharts(result.charts || []);
    populateDataOverview(result.analysis, result.filename);
    populateCleaningSection(result.analysis.cleaning_report);
    populateGraphsSection(result.charts || []);
    populateMLSection(result.ml_prediction);
    populateReportsSection(result.analysis, result.report_text);

    // Show AI Panel
    document.getElementById('ai-assistant').style.display = 'block';
}

// ─── Display Errors ─────────────────────────────────────────────
function showError(msg) {
    const el = document.getElementById('chart-container');
    if (el) {
        el.innerHTML = `<div class="empty-state" style="color:#ef4444; grid-column:1/-1">❌ Error: ${msg}</div>`;
    } else {
        alert(`Error: ${msg}`);
    }
}

// ─── KPI Bar ───────────────────────────────────────────────────
function populateKPIs(analysis) {
    document.getElementById('kpi-rows').textContent = (analysis.rows || 0).toLocaleString();
    document.getElementById('kpi-cols').textContent = analysis.columns || 0;
    document.getElementById('kpi-health').textContent = `${analysis.health_score || 100}%`;
    document.getElementById('kpi-insights').textContent = (analysis.insights || []).length;
    document.getElementById('kpi-box').style.display = 'grid';
}

// ─── AI Narrative Insights ─────────────────────────────────────
function populateInsights(insights) {
    const bubbles = document.getElementById('ai-bubbles');
    bubbles.innerHTML = '';
    (insights || []).forEach((text, i) => setTimeout(() => {
        const div = document.createElement('div');
        div.className = 'bubble';
        div.textContent = text;
        bubbles.appendChild(div);
    }, i * 150));
}

// ─── Section 2: Data Overview ──────────────────────────────────
function populateDataOverview(analysis, filename) {
    const el = document.getElementById('overview-content');
    if (!analysis) return;

    const numCols = analysis.numerical_cols || [];
    const catCols = analysis.categorical_cols || [];

    let colRows = Object.entries(analysis.summary || {}).map(([col, stats]) => `
        <tr>
            <td style="font-weight:600">${col}</td>
            <td><span class="badge">${numCols.includes(col) ? 'Numeric' : 'Categorical'}</span></td>
            <td>${stats['count'] !== undefined ? stats['count'] : '-'}</td>
            <td>${stats['mean'] !== undefined ? stats['mean'] : '-'}</td>
            <td>${stats['min'] !== undefined ? stats['min'] : '-'}</td>
            <td>${stats['max'] !== undefined ? stats['max'] : '-'}</td>
        </tr>
    `).join('');

    el.innerHTML = `
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:24px">
            <div class="kpi-card">
                <span class="kpi-label">Dataset Name</span>
                <span class="kpi-val" style="font-size:1.1rem; color:var(--accent-color); word-break:break-all">${filename || 'Uploaded File'}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Total Rows</span>
                <span class="kpi-val">${analysis.rows}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Total Columns</span>
                <span class="kpi-val">${analysis.columns}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Numerical Features</span>
                <span class="kpi-val" style="color:var(--accent-secondary)">${numCols.length}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Categorical Features</span>
                <span class="kpi-val" style="color:#8b5cf6">${catCols.length}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Quality Score</span>
                <span class="kpi-val" style="color:#10b981">${analysis.health_score}%</span>
            </div>
        </div>

        <h4 style="margin-bottom:12px; color:var(--text-primary)">Column Statistics & Data Types</h4>
        <div style="overflow-x:auto">
            <table class="clean-table">
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Mean</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>${colRows || '<tr><td colspan="6" style="text-align:center">No numeric summary available</td></tr>'}</tbody>
            </table>
        </div>`;
}

// ─── Section 3: Data Cleaning ──────────────────────────────────
function populateCleaningSection(rep) {
    const el = document.getElementById('cleaning-info');
    if (!rep) {
        el.innerHTML = '<div class="empty-state">No cleaning report available.</div>';
        return;
    }

    const colDetails = rep.column_details || [];
    const rowsHtml = colDetails.map(item => `
        <tr>
            <td style="font-weight:600">${item.column}</td>
            <td style="color:${item.missing_before > 0 ? '#ef4444' : 'var(--text-secondary)'}">${item.missing_before}</td>
            <td style="color:#10b981">${item.missing_after}</td>
            <td>${item.strategy}</td>
            <td>
                <span class="badge" style="background:${item.missing_before > 0 ? '#eff6ff' : '#f1f5f9'}; color:${item.missing_before > 0 ? 'var(--accent-color)' : 'var(--text-secondary)'}">
                    ${item.missing_before > 0 ? '⚡ Repaired' : '✅ Clean'}
                </span>
            </td>
        </tr>
    `).join('');

    el.innerHTML = `
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:24px">
            <div class="kpi-card" style="border-top-color:#10b981">
                <span class="kpi-label">Missing Values Fixed</span>
                <span class="kpi-val" style="color:#10b981">${rep.missing_values_fixed}</span>
            </div>
            <div class="kpi-card" style="border-top-color:#f59e0b">
                <span class="kpi-label">Duplicate Rows Removed</span>
                <span class="kpi-val" style="color:#f59e0b">${rep.duplicates_removed}</span>
            </div>
            <div class="kpi-card" style="border-top-color:var(--accent-color)">
                <span class="kpi-label">Row Count Change</span>
                <span class="kpi-val" style="font-size:1.4rem">${rep.rows_before} → ${rep.rows_after}</span>
            </div>
        </div>

        <h4 style="margin-bottom:12px; color:var(--text-primary)">Per-Column Data Cleaning Breakdown</h4>
        <div style="overflow-x:auto">
            <table class="clean-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Missing (Before)</th>
                        <th>Missing (After)</th>
                        <th>Cleaning Action Applied</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>`;
}

// ─── Plotly Chart Rendering with Rotation & Label Fixes ───────
function renderChart(chartData, index, container, prefix = 'c') {
    const settings = getSettings();
    const chartType = chartData._chart_type || 'other';

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.dataset.chartType = chartType;

    const header = document.createElement('div');
    header.className = 'chart-header';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'chart-card-title';
    titleSpan.textContent = (chartData.layout?.title?.text || `Chart ${index + 1}`).replace(/<[^>]+>/g, '');
    header.appendChild(titleSpan);

    const btnGroup = document.createElement('div');
    btnGroup.style.display = 'flex';
    btnGroup.style.gap = '8px';
    btnGroup.style.alignItems = 'center';

    const plotId = `${prefix}_${index}`;

    // Check if chart has bar trace for rotation toggle
    const hasBarTrace = (chartData.data || []).some(t => t.type === 'bar');

    if (hasBarTrace) {
        const rotBtn = document.createElement('button');
        rotBtn.innerHTML = '🔄 Rotate';
        rotBtn.className = 'filter-btn';
        rotBtn.style.padding = '5px 12px';
        rotBtn.style.fontSize = '0.82rem';
        rotBtn.title = 'Switch bar chart orientation (Horizontal vs Vertical)';
        rotBtn.onclick = () => toggleChartRotation(plotId);
        btnGroup.appendChild(rotBtn);
    }

    const dlBtn = document.createElement('button');
    dlBtn.innerHTML = '📥 Download';
    dlBtn.className = 'filter-btn';
    dlBtn.style.padding = '5px 12px';
    dlBtn.style.fontSize = '0.82rem';
    dlBtn.onclick = () => downloadChart(plotId, titleSpan.textContent);
    btnGroup.appendChild(dlBtn);

    header.appendChild(btnGroup);

    const plotDiv = document.createElement('div');
    plotDiv.id = plotId;
    plotDiv.className = 'plotly-box';
    plotDiv.dataset.originalData = JSON.stringify(chartData.data);
    plotDiv.dataset.originalLayout = JSON.stringify(chartData.layout);

    card.appendChild(header);
    card.appendChild(plotDiv);
    container.appendChild(card);

    // Enhanced Layout with high label visibility
    const rawX = chartData.data && chartData.data[0] ? chartData.data[0].x : [];
    const isLongCategory = Array.isArray(rawX) && rawX.length > 4;

    const layout = {
        ...chartData.layout,
        autosize: true,
        height: 480,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#f8fafc',
        margin: { l: 100, r: 40, t: 60, b: isLongCategory ? 110 : 80, pad: 6 },
        font: { family: 'Inter, sans-serif', size: 12, color: '#0f172a' },
        title: {
            text: (chartData.layout?.title?.text || '').replace(/<[^>]+>/g, ''),
            font: { size: 16, color: '#0f172a', family: 'Inter, sans-serif' },
            x: 0.02,
            xanchor: 'left'
        },
        xaxis: {
            ...(chartData.layout?.xaxis || {}),
            automargin: true,
            tickangle: isLongCategory ? -35 : 0,
            tickfont: { size: 12, color: '#0f172a', family: 'Inter, sans-serif' },
            title: {
                ...(chartData.layout?.xaxis?.title || {}),
                font: { size: 13, color: '#0f172a', family: 'Inter, sans-serif' }
            },
            gridcolor: '#e2e8f0'
        },
        yaxis: {
            ...(chartData.layout?.yaxis || {}),
            automargin: true,
            tickfont: { size: 12, color: '#0f172a', family: 'Inter, sans-serif' },
            title: {
                ...(chartData.layout?.yaxis?.title || {}),
                font: { size: 13, color: '#0f172a', family: 'Inter, sans-serif' }
            },
            gridcolor: '#e2e8f0'
        }
    };
    delete layout.width;

    try {
        Plotly.newPlot(plotId, JSON.parse(JSON.stringify(chartData.data || [])), layout, {
            responsive: true,
            displayModeBar: false,
            scrollZoom: false,
            staticPlot: false
        });
    } catch (err) {
        console.error(`Failed to render Plotly chart #${index}:`, err);
        plotDiv.innerHTML = `<div class="empty-state" style="padding:20px; color:#ef4444">⚠️ Could not render chart ${index + 1}: ${err.message}</div>`;
    }
}

// ─── Rotate / Orientation Toggle Function ──────────────────────
function toggleChartRotation(plotId) {
    const el = document.getElementById(plotId);
    if (!el || !el.data) return;

    const isRotated = el.dataset.isRotated === 'true';

    if (!isRotated) {
        // Rotate to Horizontal (swapped x and y)
        const newTraces = el.data.map(t => {
            if (t.type === 'bar') {
                return {
                    ...t,
                    x: t.y,
                    y: t.x,
                    orientation: 'h',
                    textposition: 'outside',
                    textfont: { size: 11, color: '#0f172a' }
                };
            }
            return t;
        });

        const currentXTitle = el.layout.xaxis?.title?.text || '';
        const currentYTitle = el.layout.yaxis?.title?.text || '';

        const newLayout = {
            ...el.layout,
            xaxis: {
                ...el.layout.xaxis,
                title: { text: currentYTitle, font: { size: 13, color: '#0f172a', family: 'Inter, sans-serif' } },
                tickangle: 0,
                automargin: true,
                tickfont: { size: 12, color: '#0f172a', family: 'Inter, sans-serif' }
            },
            yaxis: {
                ...el.layout.yaxis,
                title: { text: currentXTitle, font: { size: 13, color: '#0f172a', family: 'Inter, sans-serif' } },
                automargin: true,
                tickfont: { size: 12, color: '#0f172a', family: 'Inter, sans-serif' }
            },
            margin: { l: 150, r: 50, t: 60, b: 70, pad: 6 }
        };

        Plotly.react(plotId, newTraces, newLayout);
        el.dataset.isRotated = 'true';
    } else {
        // Restore original vertical orientation
        const origData = JSON.parse(el.dataset.originalData);
        const origLayout = JSON.parse(el.dataset.originalLayout);

        const rawX = origData && origData[0] ? origData[0].x : [];
        const isLongCategory = Array.isArray(rawX) && rawX.length > 4;

        const restoredLayout = {
            ...el.layout,
            margin: { l: 100, r: 40, t: 60, b: isLongCategory ? 110 : 80, pad: 6 },
            xaxis: {
                ...el.layout.xaxis,
                title: origLayout.xaxis?.title || el.layout.xaxis?.title,
                tickangle: isLongCategory ? -35 : 0,
                automargin: true,
                tickfont: { size: 12, color: '#0f172a', family: 'Inter, sans-serif' }
            },
            yaxis: {
                ...el.layout.yaxis,
                title: origLayout.yaxis?.title || el.layout.yaxis?.title,
                automargin: true,
                tickfont: { size: 12, color: '#0f172a', family: 'Inter, sans-serif' }
            }
        };

        Plotly.react(plotId, origData, restoredLayout);
        el.dataset.isRotated = 'false';
    }
}

function downloadChart(id, title) {
    const el = document.getElementById(id);
    if (el) {
        Plotly.downloadImage(el, {
            format: 'png',
            width: 1200,
            height: 800,
            filename: title.trim().replace(/\s+/g, '_') || 'chart'
        });
    }
}

function populateDashboardCharts(charts) {
    const container = document.getElementById('chart-container');
    container.innerHTML = '';
    if (!charts || !charts.length) return;
    charts.slice(0, 6).forEach((c, i) => renderChart(c, i, container, 'dash'));
}

function populateGraphsSection(charts) {
    const container = document.getElementById('all-charts-container');
    container.innerHTML = '';
    if (!charts || !charts.length) {
        container.innerHTML = '<div class="empty-state" style="grid-column:1/-1">No charts generated yet.</div>';
        return;
    }

    const settings = getSettings();
    charts.slice(0, settings.maxCharts).forEach((c, i) => renderChart(c, i, container, 'graph'));

    // Wire Chart Filters
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn[data-filter]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const filter = btn.dataset.filter;

            document.querySelectorAll('#all-charts-container .chart-card').forEach(card => {
                if (filter === 'all' || card.dataset.chartType === filter) {
                    card.style.display = 'block';
                    const plotDiv = card.querySelector('.plotly-box');
                    if (plotDiv && plotDiv.data) {
                        Plotly.Plots.resize(plotDiv);
                    }
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
}

// ─── Machine Learning Section ──────────────────────────────────
function populateMLSection(ml) {
    const el = document.getElementById('ml-result');
    const targetSelect = document.getElementById('target-col-select');
    const trainBtn = document.getElementById('btn-train-ml');

    if (!ml || ml.error) {
        el.innerHTML = `<div class="empty-state" style="color:#ef4444">⚠️ ${ml ? ml.error : 'ML Model not available'}</div>`;
        return;
    }

    // Populate Target Column Select Dropdown
    if (ml.all_columns && ml.all_columns.length > 0) {
        targetSelect.innerHTML = ml.all_columns.map(col =>
            `<option value="${col}" ${col === ml.target_column ? 'selected' : ''}>${col}</option>`
        ).join('');
        targetSelect.disabled = false;
        trainBtn.disabled = false;
    }

    const isClassif = ml.problem_type === 'classification';
    const metrics = ml.metrics || {};

    let metricsHtml = '';
    if (isClassif) {
        metricsHtml = `
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px; margin-bottom:20px">
                <div class="kpi-card" style="border-top-color:#10b981">
                    <span class="kpi-label">Accuracy Score</span>
                    <span class="kpi-val" style="color:#10b981">${(metrics.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div class="kpi-card" style="border-top-color:var(--accent-color)">
                    <span class="kpi-label">Precision</span>
                    <span class="kpi-val" style="color:var(--accent-color)">${(metrics.precision * 100).toFixed(1)}%</span>
                </div>
                <div class="kpi-card" style="border-top-color:#8b5cf6">
                    <span class="kpi-label">Recall</span>
                    <span class="kpi-val" style="color:#8b5cf6">${(metrics.recall * 100).toFixed(1)}%</span>
                </div>
                <div class="kpi-card" style="border-top-color:#f59e0b">
                    <span class="kpi-label">F1 Score</span>
                    <span class="kpi-val" style="color:#f59e0b">${(metrics.f1_score * 100).toFixed(1)}%</span>
                </div>
            </div>`;
    } else {
        metricsHtml = `
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:20px">
                <div class="kpi-card" style="border-top-color:#10b981">
                    <span class="kpi-label">R² Score (Variance)</span>
                    <span class="kpi-val" style="color:#10b981">${(metrics.r2_score * 100).toFixed(1)}%</span>
                </div>
                <div class="kpi-card" style="border-top-color:var(--accent-color)">
                    <span class="kpi-label">Mean Absolute Error (MAE)</span>
                    <span class="kpi-val" style="color:var(--accent-color)">${metrics.mae}</span>
                </div>
                <div class="kpi-card" style="border-top-color:#f59e0b">
                    <span class="kpi-label">Root Mean Sq Error (RMSE)</span>
                    <span class="kpi-val" style="color:#f59e0b">${metrics.rmse}</span>
                </div>
            </div>`;
    }

    // Feature Importance Bar Chart
    const featImpList = ml.feature_importance || [];
    const featImpBars = featImpList.map(item => `
        <div class="feat-imp-row">
            <span class="feat-imp-name" title="${item.feature}">${item.feature}</span>
            <div class="feat-imp-bar-bg">
                <div class="feat-imp-bar-fill" style="width:${Math.max(item.importance_pct, 4)}%"></div>
            </div>
            <span class="feat-imp-val">${item.importance_pct}%</span>
        </div>
    `).join('');

    el.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:20px">
            <div>
                <h3 style="margin:0; color:var(--text-primary)">Target Feature: <span style="color:var(--accent-color)">${ml.target_column}</span></h3>
                <p style="margin-top:4px; color:var(--text-secondary); font-size:0.9rem">
                    Model: <strong>${ml.model_type}</strong> &bull; Problem Type: <span class="badge" style="background:#eff6ff; color:var(--accent-color); font-size:0.85rem">${ml.problem_type.toUpperCase()}</span>
                </p>
            </div>
        </div>

        ${metricsHtml}

        <h4 style="margin-top:24px; margin-bottom:12px; color:var(--text-primary)">Ranked Feature Importance</h4>
        <div class="feat-imp-container">
            ${featImpBars || '<div class="empty-state">No feature importances available.</div>'}
        </div>`;

    // Render Live Prediction Form
    populateLivePredictionForm(ml);
}

// Target column change event listener
async function onTargetColumnChange() {
    if (!analysisData || !analysisData.file_path) return;
    const targetSelect = document.getElementById('target-col-select');
    const selectedTarget = targetSelect.value;
    if (!selectedTarget) return;

    const settings = getSettings();
    const trainBtn = document.getElementById('btn-train-ml');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';

    try {
        const response = await fetch(`${settings.apiUrl}/train-ml`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: analysisData.file_path,
                target_column: selectedTarget
            })
        });

        const result = await response.json();
        if (!response.ok || result.status === 'error') {
            throw new Error(result.detail || 'Training failed');
        }

        analysisData.ml_prediction = result.ml_prediction;
        populateMLSection(result.ml_prediction);
    } catch (err) {
        alert(`Error training model: ${err.message}`);
    } finally {
        trainBtn.disabled = false;
        trainBtn.textContent = '⚡ Train Model';
    }
}

// ─── Live Prediction Form & Calculator ─────────────────────────
function populateLivePredictionForm(ml) {
    const container = document.getElementById('predict-form-container');
    const outputDiv = document.getElementById('predict-output');

    if (!ml || !ml.feature_schema || ml.feature_schema.length === 0) {
        container.innerHTML = '<div class="empty-state">Live prediction inputs not available for this dataset.</div>';
        return;
    }

    const inputsHtml = ml.feature_schema.map(field => {
        if (field.type === 'numeric') {
            return `
                <div class="setting-group" style="margin-bottom:14px">
                    <label class="form-label">${field.name.replace(/_/g, ' ').toUpperCase()}</label>
                    <input type="number" class="form-input predict-input" data-feature="${field.name}" value="${field.default}" step="any">
                </div>`;
        } else {
            const opts = (field.options || []).map(opt => `<option value="${opt}" ${opt === field.default ? 'selected' : ''}>${opt}</option>`).join('');
            return `
                <div class="setting-group" style="margin-bottom:14px">
                    <label class="form-label">${field.name.replace(/_/g, ' ').toUpperCase()}</label>
                    <select class="form-select predict-input" data-feature="${field.name}">
                        ${opts}
                    </select>
                </div>`;
        }
    }).join('');

    container.innerHTML = `
        <form id="live-predict-form" onsubmit="executeLivePrediction(event)">
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px">
                ${inputsHtml}
            </div>
            <button type="submit" class="btn-primary" style="margin-top:20px; width:100%; padding:12px; font-size:1rem">
                🎯 Calculate Live Prediction
            </button>
        </form>`;

    outputDiv.innerHTML = '';
}

async function executeLivePrediction(event) {
    event.preventDefault();
    if (!analysisData || !analysisData.file_path || !analysisData.ml_prediction) {
        alert("Upload dataset and train model first!");
        return;
    }

    const settings = getSettings();
    const ml = analysisData.ml_prediction;
    const inputs = document.querySelectorAll('.predict-input');

    const featureValues = {};
    inputs.forEach(input => {
        const featName = input.dataset.feature;
        featureValues[featName] = input.value;
    });

    const outputDiv = document.getElementById('predict-output');
    outputDiv.innerHTML = `<div class="empty-state"><div class="spinner"></div> Computing prediction...</div>`;

    try {
        const response = await fetch(`${settings.apiUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: analysisData.file_path,
                target_column: ml.target_column,
                feature_values: featureValues
            })
        });

        const result = await response.json();
        if (!response.ok || result.status === 'error') {
            throw new Error(result.detail || 'Prediction failed');
        }

        if (result.problem_type === 'classification') {
            outputDiv.innerHTML = `
                <div class="kpi-card" style="border-top-color:#10b981; background:#f0fdf4">
                    <span class="kpi-label">Predicted ${ml.target_column} Output</span>
                    <span class="kpi-val" style="color:#16a34a; font-size:2.2rem; margin-top:8px">${result.prediction.toUpperCase()}</span>
                    <div style="margin-top:8px; font-weight:600; color:var(--text-secondary); font-size:0.9rem">
                        Model Confidence: <span style="color:#10b981">${result.confidence}%</span>
                    </div>
                </div>`;
        } else {
            outputDiv.innerHTML = `
                <div class="kpi-card" style="border-top-color:var(--accent-color); background:#eff6ff">
                    <span class="kpi-label">Predicted ${ml.target_column} Value</span>
                    <span class="kpi-val" style="color:var(--accent-color); font-size:2.2rem; margin-top:8px">${result.prediction_formatted}</span>
                </div>`;
        }
    } catch (err) {
        outputDiv.innerHTML = `<div class="empty-state" style="color:#ef4444">❌ Prediction error: ${err.message}</div>`;
    }
}

// ─── Download Analysis Report ──────────────────────────────────
function populateReportsSection(analysis, reportText) {
    const el = document.getElementById('result');
    if (reportText) {
        el.textContent = reportText;
    } else {
        el.textContent = JSON.stringify(analysis, null, 2);
    }
}

function downloadTextReport() {
    if (!analysisData) return alert('No report available to download. Please upload a dataset first.');
    const text = analysisData.report_text || JSON.stringify(analysisData.analysis, null, 2);
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'automated_data_analysis_report.txt';
    link.click();
}

function downloadReport() {
    if (!analysisData) return alert('No report available to download. Please upload a dataset first.');
    const dataStr = JSON.stringify(analysisData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    const link = document.createElement('a');
    link.setAttribute('href', dataUri);
    link.setAttribute('download', 'ai_data_report.json');
    link.click();
}

// ─── AI Chat with Data ─────────────────────────────────────────
async function sendChatQuery() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const query = input.value.trim();
    if (!query) return;

    if (!analysisData || !analysisData.file_path) {
        alert("Please upload a dataset first on the Dashboard!");
        return;
    }

    const settings = getSettings();
    const emptyState = document.getElementById('chat-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    const messagesBox = document.getElementById('chat-messages');

    // ── User message bubble ──────────────────────────────────────
    const userDiv = document.createElement('div');
    userDiv.style.cssText = 'background:#eff6ff; padding:12px 16px; border-radius:12px 12px 0 12px; align-self:flex-end; max-width:80%; border:1px solid #bfdbfe; font-size:0.92rem; margin-bottom:6px;';
    userDiv.innerHTML = `<strong style="color:var(--accent-color)">You:</strong> <span>${query.replace(/</g, "&lt;")}</span>`;
    messagesBox.appendChild(userDiv);

    input.value = '';

    // ── AI loading bubble ────────────────────────────────────────
    const aiDiv = document.createElement('div');
    aiDiv.style.cssText = 'background:#ffffff; padding:12px 16px; border-radius:12px 12px 12px 0; align-self:flex-start; max-width:80%; border:1px solid var(--border-color); font-size:0.92rem; margin-bottom:6px;';
    aiDiv.innerHTML = `<strong style="color:var(--text-primary)">AI Assistant:</strong> <span><span class="spinner"></span> AI is analyzing your dataset...</span>`;
    messagesBox.appendChild(aiDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;

    // Disable send button while processing
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.textContent = '⏳ Thinking...';
    }
    if (input) input.disabled = true;

    try {
        // ── API key is NEVER sent from frontend — handled by backend ──
        const response = await fetch(`${settings.apiUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                file_path: analysisData.file_path
                // api_key intentionally omitted — backend reads from .env
            })
        });

        const result = await response.json();
        let rawText = result.response || result.detail || "No response received.";

        // Render Markdown-lite formatting
        let mdText = rawText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^### (.+)$/gm, '<h4 style="margin:8px 0 4px;color:var(--text-primary)">$1</h4>')
            .replace(/^## (.+)$/gm, '<h3 style="margin:10px 0 5px;color:var(--text-primary)">$1</h3>')
            .replace(/^- (.+)$/gm, '<li style="margin-left:14px;margin-bottom:3px">$1</li>')
            .replace(/\n/g, '<br>');

        // Check if response is an error/warning from backend
        const isError = rawText.startsWith('⚠️') || rawText.startsWith('Error:');
        const labelColor = isError ? '#ef4444' : 'var(--text-primary)';

        aiDiv.innerHTML = `<strong style="color:${labelColor}">AI Assistant:</strong> <div style="margin-top:6px; line-height:1.6">${mdText}</div>`;

    } catch (err) {
        // Network/connection error
        aiDiv.innerHTML = `
            <strong style="color:#ef4444">AI Assistant:</strong>
            <div style="margin-top:6px; color:#ef4444; line-height:1.5">
                ⚠️ AI Assistant is temporarily unavailable.<br>
                Could not reach the backend server. Please ensure the FastAPI server is running and try again.
            </div>`;
    } finally {
        // Re-enable button and input regardless of success or failure
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.textContent = '🚀 Ask AI';
        }
        if (input) input.disabled = false;
        messagesBox.scrollTop = messagesBox.scrollHeight;
    }
}


// Support Enter key in chat input
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', e => {
            if (e.key === 'Enter') sendChatQuery();
        });
    }
});

// ─── Clear Button & Safe Cleanup ───────────────────────────────
async function clearAll() {
    const settings = getSettings();

    if (analysisData && analysisData.file_path) {
        try {
            await fetch(`${settings.apiUrl}/clear`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: analysisData.file_path })
            });
        } catch (err) {
            console.error("Error deleting backend file:", err);
        }
    }

    analysisData = null;

    ['file-name-dash'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '';
    });

    document.getElementById('chart-container').innerHTML = '';
    document.getElementById('all-charts-container').innerHTML = '';
    document.getElementById('cleaning-info').innerHTML = 'Upload a dataset to generate the Before vs After cleaning report.';
    document.getElementById('overview-content').innerHTML = 'Upload and analyze a dataset to view shape, column types, and data quality details.';
    document.getElementById('ml-result').innerHTML = 'Upload and analyze a dataset first to configure and train machine learning models.';
    document.getElementById('predict-form-container').innerHTML = '<div class="empty-state">Train a machine learning model first to generate the live prediction calculator.</div>';
    document.getElementById('predict-output').innerHTML = '';
    document.getElementById('result').textContent = 'No report generated yet. Upload a dataset to view full report.';
    document.getElementById('ai-bubbles').innerHTML = '';
    document.getElementById('chat-messages').innerHTML = `
        <div class="empty-state" id="chat-empty-state">
            Ask questions about your uploaded dataset, such as:<br>
            <em style="color:var(--accent-color)">"Summarize the key correlations in this dataset."</em>
        </div>`;

    document.getElementById('target-col-select').innerHTML = '<option value="">Upload a dataset first</option>';
    document.getElementById('target-col-select').disabled = true;
    document.getElementById('btn-train-ml').disabled = true;

    document.getElementById('kpi-box').style.display = 'none';
    document.getElementById('ai-assistant').style.display = 'none';

    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    const dashNav = document.querySelector('.nav-item[data-section="section-dashboard"]');
    if (dashNav) dashNav.classList.add('active');

    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active-section'));
    document.getElementById('section-dashboard').classList.add('active-section');

    const meta = SECTION_META['section-dashboard'];
    if (meta) {
        document.getElementById('page-title').textContent = meta.title;
        document.getElementById('page-subtitle').textContent = meta.sub;
    }
}

// ─── Application Initialization ────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNav();
    initFileInputs();

    const { maxCharts, theme, geminiKey } = getSettings();
    const maxInput = document.getElementById('setting-max-charts');
    const themeInput = document.getElementById('setting-theme');
    const keyInput = document.getElementById('setting-gemini-key');

    if (maxInput) maxInput.value = maxCharts;
    if (themeInput) themeInput.value = theme;
    if (keyInput) keyInput.value = geminiKey;
});