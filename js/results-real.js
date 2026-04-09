/**
 * VisionX — Results Page (Phase 3)
 * Reads prediction data from real API: GET /predictions/{id}
 * Falls back to localStorage only if prediction_id is a legacy "comp_" ID.
 */

async function loadResults() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    const params = new URLSearchParams(window.location.search);
    const id     = params.get('id');

    if (!id) {
        _showError('No prediction ID in URL. <a href="comparison.html">Make a new comparison</a>');
        return;
    }

    // Legacy localStorage comparison (comp_TIMESTAMP format)
    if (id.startsWith('comp_')) {
        const stored = JSON.parse(localStorage.getItem('comparisons') || '[]');
        const comp   = stored.find(c => c.id === id);
        if (comp) { _renderFromLegacy(comp); return; }
    }

    // ── Real API ──────────────────────────────────────────────────────────────
    try {
        showLoading('recommendationCard');
        const res  = await apiCall(`/predictions/${id}`);
        const data = res.data || res;
        _renderPrediction(data);
    } catch (err) {
        _showError('Could not load prediction: ' + err.message);
    }
}

// ── Render real API prediction ────────────────────────────────────────────────
function _renderPrediction(data) {
    const confidence = Math.round((data.confidence || 0) * 100);

    // Recommendation card
    document.getElementById('recommendationCard').innerHTML = `
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
            <div style="width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#4F8CFF,#7B61FF);display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🤖</div>
            <div>
                <h2 style="margin:0;">AI Recommendation</h2>
                <p style="color:var(--text-secondary);margin:0;">${data.domain_detected || 'Decision Analysis'}</p>
            </div>
        </div>
        <div style="background:rgba(79,140,255,0.1);border:1px solid rgba(79,140,255,0.3);border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:0.25rem;">RECOMMENDED CHOICE</div>
            <div style="font-size:1.8rem;font-weight:700;color:#4F8CFF;">${data.recommended_option_name || 'N/A'}</div>
            <div style="margin-top:0.75rem;display:flex;align-items:center;gap:0.5rem;">
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;">
                    <div style="width:${confidence}%;height:100%;background:linear-gradient(90deg,#4F8CFF,#7B61FF);border-radius:4px;"></div>
                </div>
                <span style="font-weight:600;color:#4F8CFF;">${confidence}% confidence</span>
            </div>
        </div>
        <p style="color:var(--text-secondary);line-height:1.6;">${data.reasoning || ''}</p>
        ${data.cluster_id != null ? `
        <div style="margin-top:1rem;font-size:0.85rem;color:var(--text-secondary);">
            Predicted time: <strong style="color:#9AA3C7;">${data.prediction_time_ms ? data.prediction_time_ms.toFixed(0) + 'ms' : '—'}</strong>
            &nbsp;·&nbsp; Prediction ID: <code style="color:#6B7298;font-size:0.8rem;">${data.prediction_id}</code>
        </div>` : ''}
        ${data.feedback ? `
        <div style="margin-top:1rem;padding:0.75rem 1rem;background:rgba(16,185,129,0.1);border-radius:8px;font-size:0.85rem;color:#10b981;">
            ✅ You submitted feedback for this prediction ${data.feedback.accepted ? '(accepted)' : '(rejected)'}
            ${data.feedback.satisfaction ? ' — Satisfaction: ' + data.feedback.satisfaction + '/5' : ''}
        </div>` : _renderFeedbackForm(data.prediction_id)}
    `;

    // SHAP / feature importance
    const featEl = document.getElementById('featureImportance');
    const shap   = data.shap_values;
    const uf     = data.universal_features;

    if (shap && Object.keys(shap).length) {
        // SHAP waterfall
        const sorted = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.001);
        featEl.innerHTML = `
            <div style="margin-bottom:0.75rem;font-size:0.8rem;color:#9AA3C7;">
                SHAP values — how each feature pushed the recommendation
            </div>
            ${sorted.map(([feat, val]) => {
                const label = feat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                const pct   = Math.abs(val) / maxAbs * 100;
                const pos   = val >= 0;
                return `
                <div style="margin-bottom:0.85rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                        <span style="font-weight:500;">${label}</span>
                        <span style="color:${pos ? '#10b981' : '#ef4444'};font-family:monospace;">${val >= 0 ? '+' : ''}${val.toFixed(3)}</span>
                    </div>
                    <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden;">
                        <div style="width:${pct}%;height:100%;background:${pos ? 'linear-gradient(90deg,#10b981,#34d399)' : 'linear-gradient(90deg,#ef4444,#f87171)'};border-radius:4px;"></div>
                    </div>
                </div>`;
            }).join('')}`;
    } else {
        featEl.innerHTML = '<p style="color:var(--text-secondary);">SHAP values not available for this prediction.</p>';
    }

    // Score chart — universal features radar if available
    const ctx = document.getElementById('scoreChart')?.getContext('2d');
    if (ctx && uf && Object.keys(uf).length) {
        const labels = Object.keys(uf).map(k => k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
        const values = Object.values(uf);
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels,
                datasets: [{
                    label: data.recommended_option_name || 'Recommended',
                    data: values.map(v => v * 10),
                    borderColor: '#4F8CFF',
                    backgroundColor: 'rgba(79,140,255,0.15)',
                    pointBackgroundColor: '#4F8CFF',
                    pointRadius: 4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { r: {
                    min: 0, max: 10,
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    ticks: { color: '#9AA3C7', backdropColor: 'transparent', stepSize: 2 },
                    pointLabels: { color: '#9AA3C7', font: { size: 11 }}
                }},
                plugins: {
                    legend: { labels: { color: '#9AA3C7' }},
                    tooltip: { backgroundColor: 'rgba(11,15,43,0.9)', titleColor: '#E6E8F2', bodyColor: '#9AA3C7' }
                }
            }
        });
    }
}

// ── Feedback form ─────────────────────────────────────────────────────────────
function _renderFeedbackForm(predictionId) {
    return `
        <div id="feedbackForm" style="margin-top:1.5rem;padding:1rem;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;">
            <div style="font-size:0.9rem;font-weight:500;margin-bottom:0.75rem;color:#E6E8F2;">Did you follow this recommendation?</div>
            <div style="display:flex;gap:0.75rem;flex-wrap:wrap;">
                <button onclick="submitFeedback('${predictionId}', true, 5)"
                    style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.4);color:#10b981;padding:0.5rem 1.25rem;border-radius:8px;cursor:pointer;">
                    ✅ Yes, I followed it
                </button>
                <button onclick="submitFeedback('${predictionId}', false, 2)"
                    style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#ef4444;padding:0.5rem 1.25rem;border-radius:8px;cursor:pointer;">
                    ❌ No, I chose differently
                </button>
            </div>
        </div>`;
}

async function submitFeedback(predictionId, accepted, satisfaction) {
    try {
        await apiCall('/feedback/prediction', {
            method: 'POST',
            body: { prediction_id: predictionId, accepted, satisfaction },
        });
        document.getElementById('feedbackForm').outerHTML = `
            <div style="margin-top:1.5rem;padding:0.75rem 1rem;background:rgba(16,185,129,0.1);border-radius:8px;font-size:0.85rem;color:#10b981;">
                ✅ Feedback recorded — thank you! This improves future recommendations.
            </div>`;
    } catch (err) {
        showNotification('Feedback failed: ' + err.message, 'error');
    }
}

// ── Legacy localStorage renderer ──────────────────────────────────────────────
function _renderFromLegacy(comparison) {
    const result     = comparison.result;
    const confidence = Math.round((result.confidence || 0) * 100);

    if (result.is_fallback) {
        document.getElementById('fallbackWarning').style.display = 'block';
    }

    document.getElementById('recommendationCard').innerHTML = `
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
            <div style="width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#4F8CFF,#7B61FF);display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">🤖</div>
            <div><h2 style="margin:0;">AI Recommendation</h2><p style="color:var(--text-secondary);margin:0;">${comparison.title || 'Comparison'}</p></div>
        </div>
        <div style="background:rgba(79,140,255,0.1);border:1px solid rgba(79,140,255,0.3);border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:0.25rem;">RECOMMENDED CHOICE</div>
            <div style="font-size:1.8rem;font-weight:700;color:#4F8CFF;">${result.recommended_option_name || 'N/A'}</div>
            <div style="margin-top:0.75rem;display:flex;align-items:center;gap:0.5rem;">
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;">
                    <div style="width:${confidence}%;height:100%;background:linear-gradient(90deg,#4F8CFF,#7B61FF);border-radius:4px;"></div>
                </div>
                <span style="font-weight:600;color:#4F8CFF;">${confidence}% confidence</span>
            </div>
        </div>
        <p style="color:var(--text-secondary);line-height:1.6;">${result.reasoning || ''}</p>
    `;

    const optionsEl = document.getElementById('optionsBreakdown');
    optionsEl.innerHTML = (comparison.options || []).map((opt, i) => {
        const isWinner = opt.id === result.recommended_option_id || opt.name === result.recommended_option_name;
        const alt      = (result.alternative_options || []).find(a => a.id === opt.id || a.name === opt.name);
        const score    = isWinner ? confidence : alt ? Math.round((alt.score || 0) * 100) : '—';
        return `
            <div style="display:flex;align-items:center;justify-content:space-between;padding:1rem;background:rgba(255,255,255,0.03);border:1px solid ${isWinner ? 'rgba(79,140,255,0.4)' : 'rgba(255,255,255,0.08)'};border-radius:8px;margin-bottom:0.75rem;">
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    ${isWinner ? '<span style="font-size:1.2rem;">🏆</span>' : `<span style="width:24px;height:24px;border-radius:50%;background:rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;font-size:0.8rem;">${i + 1}</span>`}
                    <div>
                        <div style="font-weight:600;">${opt.name}</div>
                        <div style="font-size:0.8rem;color:var(--text-secondary);">Quality: ${opt.features?.quality_score || '—'}/10 · Price: $${(opt.features?.price || 0).toLocaleString()}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-weight:700;color:${isWinner ? '#4F8CFF' : 'var(--text-secondary)'};">${score}%</div>
                    ${isWinner ? '<div style="font-size:0.75rem;color:#4F8CFF;">Recommended</div>' : ''}
                </div>
            </div>`;
    }).join('');

    const featEl   = document.getElementById('featureImportance');
    const features = result.feature_importance || [];
    featEl.innerHTML = features.map(f => {
        const pct   = Math.round((f.importance || 0) * 100);
        const label = (f.feature_name || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return `
            <div style="margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="font-weight:500;">${label}</span>
                    <span style="color:var(--text-secondary);">${pct}%</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:linear-gradient(90deg,#4F8CFF,#7B61FF);border-radius:4px;"></div>
                </div>
            </div>`;
    }).join('') || '<p style="color:var(--text-secondary);">No feature importance data.</p>';
}

function _showError(msg) {
    document.getElementById('recommendationCard').innerHTML = `
        <div style="text-align:center;color:var(--text-secondary);padding:2rem;">
            <i class="fas fa-exclamation-circle" style="font-size:2rem;margin-bottom:1rem;color:#ef4444;"></i>
            <p>${msg}</p>
        </div>`;
}

document.addEventListener('DOMContentLoaded', () => {
    if (!isAuthenticated()) { window.location.replace('login.html'); return; }
    loadResults();
});