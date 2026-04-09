/**
 * VisionX — Dashboard (Phase 3)
 * 100% real API data. No localStorage fallback for core stats.
 * Uses /analytics/kpis and /predictions/history endpoints.
 */

let _activityChart = null;
let _categoriesChart = null;

const CLUSTER_ICONS = {
    'Independent Thinker & Risk-Averse':   { icon: '🎯', color: '#4F8CFF' },
    'Growth-Oriented & Value-Conscious':   { icon: '📈', color: '#10b981' },
    'Budget Pragmatist & Stability-Seeker':{ icon: '🏦', color: '#f59e0b' },
    'Socially-Validated & Speed-Driven':   { icon: '⚡', color: '#a855f7' },
};

async function loadDashboardData() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    // Update avatar initial immediately
    document.querySelectorAll('.user-name-initial, #userInitials').forEach(el => {
        el.textContent = user.avatar || user.name?.charAt(0) || 'U';
    });

    // ── Fetch KPIs and history in parallel ──────────────────────────────────
    let kpis = null;
    let predictions = [];

    try {
        const [kpiRes, histRes] = await Promise.all([
            apiCall('/analytics/kpis'),
            apiCall('/predictions/history?limit=50'),
        ]);

        kpis        = kpiRes.data || kpiRes;
        predictions = histRes.predictions || [];

        _setEl('totalPredictions', kpis.total_predictions ?? '—');
        _setEl('avgConfidence',
            kpis.avg_confidence != null ? Math.round(kpis.avg_confidence * 100) + '%' : '—');
        _setEl('modelAccuracy',
            kpis.model_accuracy != null ? (kpis.model_accuracy * 100).toFixed(1) + '%' : '—');

        // Cluster badge
        const cluster = kpis.user_cluster;
        _setEl('userClusterLabel', cluster || 'Not assigned yet');
        _renderClusterBadge(cluster);

    } catch (err) {
        console.error('Dashboard API error:', err);
        showNotification('Could not load dashboard data — ' + err.message, 'error');
        _setEl('totalPredictions', '—');
        _setEl('avgConfidence', '—');
        _setEl('modelAccuracy', '—');
        _setEl('userClusterLabel', '—');
    }

    _renderCharts(predictions);
    _renderRecentPredictions(predictions);
    _renderMLPerformance(kpis);
}

// ── Cluster badge ─────────────────────────────────────────────────────────────
function _renderClusterBadge(clusterLabel) {
    const el = document.getElementById('mlClusterBadge');
    if (!el) return;

    const info = CLUSTER_ICONS[clusterLabel] || { icon: '🧠', color: '#6B7298' };
    el.innerHTML = clusterLabel ? `
        <div style="display:flex;align-items:center;gap:1rem;padding:1rem;background:rgba(255,255,255,0.03);border-radius:12px;border:1px solid rgba(255,255,255,0.08);">
            <div style="width:48px;height:48px;border-radius:50%;background:${info.color}22;display:flex;align-items:center;justify-content:center;font-size:1.5rem;">${info.icon}</div>
            <div>
                <div style="font-weight:600;color:#E6E8F2;">${clusterLabel}</div>
                <div style="font-size:0.8rem;color:#9AA3C7;margin-top:2px;">Your decision-making profile — based on your real usage patterns</div>
            </div>
            <div style="margin-left:auto;background:${info.color}22;border:1px solid ${info.color}44;border-radius:8px;padding:0.4rem 0.8rem;font-size:0.8rem;color:${info.color};">ML Assigned</div>
        </div>
    ` : `<div style="color:#9AA3C7;font-size:0.9rem;padding:1rem;">Make your first comparison to get assigned a decision profile.</div>`;
}

// ── ML performance card ───────────────────────────────────────────────────────
function _renderMLPerformance(kpis) {
    const el = document.getElementById('mlPerformanceStats');
    if (!el || !kpis) return;

    const acc    = kpis.model_accuracy    ? (kpis.model_accuracy * 100).toFixed(2) + '%' : '—';
    const roc    = kpis.model_roc_auc     ? kpis.model_roc_auc.toFixed(4)              : '—';
    const fb     = kpis.feedback;
    const accRate = fb?.acceptance_rate != null ? Math.round(fb.acceptance_rate * 100) + '%' : '—';

    el.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-top:1rem;">
            ${_statPill('Accuracy', acc, '#4F8CFF')}
            ${_statPill('ROC-AUC', roc, '#10b981')}
            ${_statPill('Model', kpis.model_type || 'XGBoost', '#7B61FF')}
            ${_statPill('Acceptance Rate', accRate, '#f59e0b')}
        </div>
    `;
}

function _statPill(label, value, color) {
    return `
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:1rem;text-align:center;">
            <div style="font-size:1.3rem;font-weight:700;color:${color};">${value}</div>
            <div style="font-size:0.75rem;color:#9AA3C7;margin-top:4px;">${label}</div>
        </div>`;
}

// ── Charts ────────────────────────────────────────────────────────────────────
function _renderCharts(predictions) {
    const actCtx = document.getElementById('activityChart');
    if (actCtx) {
        const days = _last7Days();
        const counts = days.map(d =>
            predictions.filter(p => {
                const pd = new Date(p.created_at || p.timestamp);
                return pd.toDateString() === d.date.toDateString();
            }).length
        );
        _activityChart?.destroy();
        _activityChart = new Chart(actCtx, {
            type: 'line',
            data: {
                labels: days.map(d => d.label),
                datasets: [{
                    label: 'Predictions',
                    data: counts,
                    borderColor: '#4F8CFF',
                    backgroundColor: 'rgba(79,140,255,0.1)',
                    tension: 0.4, fill: true,
                    pointRadius: 4, pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false },
                    tooltip: { backgroundColor: 'rgba(11,15,43,0.9)', titleColor: '#E6E8F2', bodyColor: '#9AA3C7' }},
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9AA3C7', precision: 0 }},
                    x: { grid: { display: false }, ticks: { color: '#9AA3C7' }}
                }
            }
        });
    }

    const catCtx = document.getElementById('categoriesChart');
    if (catCtx) {
        const cats = {};
        predictions.forEach(p => {
            const c = p.domain_detected || 'General';
            cats[c] = (cats[c] || 0) + 1;
        });
        const labels = Object.keys(cats).length ? Object.keys(cats) : ['No data yet'];
        const data   = Object.values(cats).length ? Object.values(cats) : [1];

        _categoriesChart?.destroy();
        _categoriesChart = new Chart(catCtx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{ data, backgroundColor: ['#4F8CFF','#7B61FF','#A855F7','#3FA8FF','#6B7298'], borderWidth: 0 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#9AA3C7', padding: 15, font: { size: 12 }}},
                    tooltip: { backgroundColor: 'rgba(11,15,43,0.9)', titleColor: '#E6E8F2', bodyColor: '#9AA3C7' }
                }
            }
        });
    }
}

// ── Recent predictions list ───────────────────────────────────────────────────
function _renderRecentPredictions(predictions) {
    const el = document.getElementById('recentPredictions');
    if (!el) return;

    const recent = predictions.slice(0, 5);

    if (!recent.length) {
        el.innerHTML = `
            <div style="text-align:center;padding:3rem;">
                <i class="fas fa-inbox" style="font-size:3rem;color:#6B7298;margin-bottom:1rem;display:block;"></i>
                <h3 style="color:#E6E8F2;">No predictions yet</h3>
                <p style="color:#9AA3C7;margin-bottom:1.5rem;">Create your first comparison to get started</p>
                <a href="comparison.html" class="btn btn-primary"><i class="fas fa-plus-circle"></i> Create Comparison</a>
            </div>`;
        return;
    }

    el.innerHTML = recent.map(p => {
        const id    = p.prediction_id || p.id;
        const title = p.recommended_option_name ? `${p.recommended_option_name} recommended` : 'Comparison';
        const conf  = p.confidence;
        const ts    = p.created_at;
        return `
        <div onclick="window.location.href='results.html?id=${id}'" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;padding:1rem;border-bottom:1px solid rgba(255,255,255,0.06);">
            <div>
                <div style="font-weight:500;color:#E6E8F2;">${title}</div>
                <div style="font-size:0.8rem;color:#9AA3C7;">${ts ? formatDate(ts) : ''} ${p.domain_detected ? '· ' + p.domain_detected : ''}</div>
            </div>
            <div style="display:flex;align-items:center;gap:0.75rem;">
                ${conf != null ? `<span style="background:rgba(79,140,255,0.2);color:#4F8CFF;padding:0.25rem 0.75rem;border-radius:12px;font-size:0.8rem;">${Math.round(conf * 100)}% conf</span>` : ''}
                <i class="fas fa-chevron-right" style="color:#6B7298;"></i>
            </div>
        </div>`;
    }).join('');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function _setEl(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function _last7Days() {
    const names = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    return Array.from({ length: 7 }, (_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - (6 - i));
        return { date: d, label: i === 6 ? 'Today' : names[d.getDay()] };
    });
}

document.addEventListener('DOMContentLoaded', () => {
    if (!isAuthenticated()) { window.location.replace('login.html'); return; }
    loadDashboardData();
});