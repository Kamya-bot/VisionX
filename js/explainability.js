/**
 * VisionX - AI Explainability
 * Auto-loads the last prediction's SHAP values on page load
 */

let shapChart = null;

document.addEventListener('DOMContentLoaded', async () => {
    if (typeof isAuthenticated === 'function' && !isAuthenticated()) {
        window.location.replace('login.html');
        return;
    }

    // Wire up the demo button
    const btn = document.getElementById('loadDemoBtn');
    if (btn) btn.addEventListener('click', loadDemoExplanation);

    // Auto-load the latest prediction on page load
    await autoLoadLatest();
});

async function autoLoadLatest() {
    showLoading(true);
    try {
        const data = await apiCall('/predictions/history');
        const preds = data.predictions || data.history || data;

        if (!Array.isArray(preds) || preds.length === 0) {
            showLoading(false);
            showError('No predictions yet. Make a comparison first, then come back here.');
            return;
        }

        // Find the most recent prediction that has SHAP values
        const withShap = preds.find(p => p.shap_values && Object.keys(p.shap_values).length > 0);
        const latest = withShap || preds[0];
        const predId = latest.prediction_id || latest.id;

        try {
            const explanation = await apiCall(`/predictions/${predId}/explanation`);
            renderExplanation(explanation);
        } catch (e) {
            // Explanation endpoint failed — render directly from history record
            renderFromHistoryRecord(latest);
        }
    } catch (e) {
        showLoading(false);
        showError('Could not load predictions: ' + e.message);
    }
}

async function loadDemoExplanation() {
    showLoading(true);
    hideError();
    await autoLoadLatest();
}

function renderExplanation(data) {
    // Normalize fields across different response shapes
    const shap = data.shap_values || data.feature_importance || data.features || {};
    const confidence = data.confidence || data.probability || (data.prediction && data.prediction.confidence) || 0;
    const recommended = data.recommended_option || data.winner || data.recommended
        || (data.prediction && (data.prediction.recommended || data.prediction.winner))
        || 'Option A';
    const domain = data.domain_detected || data.domain
        || (data.prediction && data.prediction.domain_detected)
        || 'products';

    if (!shap || Object.keys(shap).length === 0) {
        showLoading(false);
        showError('No feature data available for this prediction.');
        return;
    }

    showLoading(false);
    document.getElementById('explanationCard').style.display = 'block';
    hideError();

    renderChart(shap);
    renderFeatureList(shap);
    renderExplanationText(shap, recommended, confidence, domain);
}

function renderFromHistoryRecord(pred) {
    const shap = pred.shap_values || {};
    const confidence = pred.confidence || pred.probability || 0;
    const recommended = pred.recommended || pred.winner || pred.option_a_name || pred.option_a || 'Option A';
    const domain = pred.domain_detected || pred.domain || 'products';

    if (Object.keys(shap).length === 0) {
        showLoading(false);
        showError('This prediction has no SHAP values. Try a newer prediction or make a new comparison.');
        return;
    }

    renderExplanation({ shap_values: shap, confidence, recommended_option: recommended, domain });
}

function renderChart(shap) {
    const entries = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const labels = entries.map(([k]) => k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
    const values = entries.map(([, v]) => parseFloat(v.toFixed(4)));
    const colors = values.map(v => v >= 0
        ? 'rgba(16, 185, 129, 0.8)'
        : 'rgba(239, 68, 68, 0.8)');

    const ctx = document.getElementById('shapChart').getContext('2d');
    if (shapChart) shapChart.destroy();

    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'SHAP Value (Feature Impact)',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` Impact: ${ctx.parsed.x >= 0 ? '+' : ''}${ctx.parsed.x.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: { color: '#9AA3C7', font: { size: 12 } },
                    title: { display: true, text: 'SHAP Value', color: '#9AA3C7' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#E6E8F2', font: { size: 13 } }
                }
            }
        }
    });
}

function renderFeatureList(shap) {
    const entries = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.0001);

    const levels = (v) => {
        const pct = Math.abs(v) / maxAbs;
        if (pct > 0.6) return 'high';
        if (pct > 0.3) return 'medium';
        return 'low';
    };

    document.getElementById('featureList').innerHTML = entries.map(([feat, val]) => {
        const pct = (Math.abs(val) / maxAbs * 100).toFixed(1);
        const isPos = val >= 0;
        const label = feat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return `
        <div class="feature-item">
            <span class="feature-name">${label}</span>
            <div class="impact-bar-container">
                <div class="impact-bar ${isPos ? 'positive' : 'negative'}" style="width:${pct}%">
                    ${pct > 15 ? (isPos ? '+' : '') + val.toFixed(4) : ''}
                </div>
            </div>
            <span class="contribution-value ${isPos ? 'positive' : 'negative'}">${isPos ? '+' : ''}${val.toFixed(4)}</span>
            <div class="impact-badge">
                <span class="badge ${levels(val)}">${levels(val)}</span>
            </div>
        </div>`;
    }).join('');
}

function renderExplanationText(shap, recommended, confidence, domain) {
    const entries = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const top3 = entries.slice(0, 3);
    const positives = top3.filter(([, v]) => v > 0).map(([k]) => k.replace(/_/g, ' '));
    const negatives = top3.filter(([, v]) => v < 0).map(([k]) => k.replace(/_/g, ' '));

    let text = `The model recommends <strong>${recommended}</strong> with <strong>${Math.round(confidence * 100)}% confidence</strong> for this ${domain} decision. `;

    if (positives.length > 0)
        text += `The strongest supporting factors are <strong>${positives.join(', ')}</strong>`;
    if (negatives.length > 0)
        text += `, while <strong>${negatives.join(', ')}</strong> work against this recommendation`;
    text += '.';

    const topFeat = top3[0];
    if (topFeat)
        text += ` <strong>${topFeat[0].replace(/_/g, ' ')}</strong> is the single most influential feature (SHAP: ${topFeat[1] >= 0 ? '+' : ''}${topFeat[1].toFixed(4)}).`;

    document.getElementById('explanationText').innerHTML = text;
}

function showLoading(show) {
    document.getElementById('loadingSpinner').classList.toggle('active', show);
}
function showError(msg) {
    const el = document.getElementById('errorMessage');
    document.getElementById('errorText').textContent = msg;
    el.classList.add('active');
}
function hideError() {
    document.getElementById('errorMessage').classList.remove('active');
}