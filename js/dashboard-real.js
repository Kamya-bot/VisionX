/**
 * Dashboard — real API data + localStorage fallback
 */

window._dashboardRealLoaded = true;

let activityChart = null;
let categoriesChart = null;

async function loadDashboardData() {
    const user = getCurrentUser();
    if (!user) { window.location.href = 'login.html'; return; }

    // ── Update user name in topbar ────────────────────────
    document.querySelectorAll('.user-name-initial').forEach(el => {
        el.textContent = (user.avatar || user.name?.charAt(0) || 'U');
    });

    // ── Update cluster label ──────────────────────────────
    const clusterEl = document.getElementById('userClusterLabel');
    if (clusterEl) clusterEl.textContent = user.cluster || 'Not assigned';

    let predictions = [];

    // ── 1. Try real API ───────────────────────────────────
    try {
        const [kpiData, histData] = await Promise.all([
            apiCall('/analytics/kpis'),
            apiCall('/predictions/history?limit=50')
        ]);

        predictions = (histData.predictions || histData.data || []);

        const kpis = kpiData.data || kpiData;
        _setEl('totalPredictions',  kpis.total_predictions ?? predictions.length);
        _setEl('avgConfidence',
            kpis.avg_confidence != null ? Math.round(kpis.avg_confidence * 100) + '%' : '—');
        _setEl('modelAccuracy',
            kpis.model_accuracy != null ? Math.round(kpis.model_accuracy * 100) + '%' : '~94%');

    } catch (err) {
        console.warn('API unavailable, using localStorage:', err.message);
        // ── 2. localStorage fallback ──────────────────────
        const stored = JSON.parse(localStorage.getItem('comparisons') || '[]');
        predictions = stored.filter(c => !c.user_id || c.user_id === user.id);

        const withConf = predictions.filter(c => c.result?.confidence);
        const avg = withConf.length
            ? withConf.reduce((s,c) => s + c.result.confidence, 0) / withConf.length : 0;

        _setEl('totalPredictions', predictions.length);
        _setEl('avgConfidence',    Math.round(avg * 100) + '%');
        _setEl('modelAccuracy',    '~94%');
    }

    // ── Charts & recent list ──────────────────────────────
    initializeCharts(predictions);
    renderRecentPredictions(predictions);
}

function _setEl(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function initializeCharts(predictions) {
    // Activity — last 7 days
    const actCtx = document.getElementById('activityChart');
    if (actCtx) {
        const days = _last7Days();
        const counts = days.map(d =>
            predictions.filter(p => {
                const pd = new Date(p.created_at || p.timestamp);
                return pd.toDateString() === d.date.toDateString();
            }).length
        );
        if (activityChart) activityChart.destroy();
        activityChart = new Chart(actCtx, {
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
                    tooltip: { backgroundColor:'rgba(11,15,43,0.9)',
                        titleColor:'#E6E8F2', bodyColor:'#9AA3C7',
                        borderColor:'#4F8CFF', borderWidth:1 }},
                scales: {
                    y: { beginAtZero:true, grid:{ color:'rgba(255,255,255,0.05)' },
                         ticks:{ color:'#9AA3C7', precision:0 }},
                    x: { grid:{ display:false }, ticks:{ color:'#9AA3C7' }}
                }
            }
        });
    }

    // Categories doughnut
    const catCtx = document.getElementById('categoriesChart');
    if (catCtx) {
        const cats = {};
        predictions.forEach(p => {
            const c = p.category || 'General';
            cats[c] = (cats[c] || 0) + 1;
        });
        const labels = Object.keys(cats).length ? Object.keys(cats) : ['No Data'];
        const data   = Object.values(cats).length ? Object.values(cats) : [1];

        if (categoriesChart) categoriesChart.destroy();
        categoriesChart = new Chart(catCtx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: ['#4F8CFF','#7B61FF','#A855F7','#3FA8FF','#6B7298'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position:'bottom',
                        labels:{ color:'#9AA3C7', padding:15, font:{ size:12 }}},
                    tooltip: { backgroundColor:'rgba(11,15,43,0.9)',
                        titleColor:'#E6E8F2', bodyColor:'#9AA3C7' }
                }
            }
        });
    }
}

function renderRecentPredictions(predictions) {
    const container = document.getElementById('recentPredictions');
    if (!container) return;

    const recent = [...predictions]
        .sort((a,b) => new Date(b.created_at||b.timestamp) - new Date(a.created_at||a.timestamp))
        .slice(0, 5);

    if (recent.length === 0) {
        container.innerHTML = `
            <div style="text-align:center;padding:3rem;">
                <i class="fas fa-inbox" style="font-size:3rem;color:#6B7298;margin-bottom:1rem;"></i>
                <h3 style="color:#E6E8F2;">No predictions yet</h3>
                <p style="color:#9AA3C7;margin-bottom:1.5rem;">Create your first comparison to get started</p>
                <a href="comparison.html" class="btn btn-primary">
                    <i class="fas fa-plus-circle"></i> Create Comparison
                </a>
            </div>`;
        return;
    }

    container.innerHTML = recent.map(p => {
        const id    = p.prediction_id || p.id;
        const title = p.title || p.comparison_title ||
                      (p.recommended_option_name ? `${p.recommended_option_name} recommended` : 'Comparison');
        const conf  = p.confidence ?? p.result?.confidence;
        const ts    = p.created_at || p.timestamp;
        return `
        <div class="comparison-item" onclick="window.location.href='results.html?id=${id}'"
             style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;
                    padding:1rem;border-bottom:1px solid rgba(255,255,255,.06);">
            <div>
                <h4 style="color:#E6E8F2;margin:0 0 .25rem;">${title}</h4>
                <span style="color:#9AA3C7;font-size:.875rem;">${ts ? formatDate(ts) : ''}</span>
            </div>
            <div style="display:flex;align-items:center;gap:.75rem;">
                ${conf != null ? `
                <span style="background:rgba(79,140,255,.2);color:#4F8CFF;
                    padding:.25rem .75rem;border-radius:12px;font-size:.875rem;">
                    ${Math.round(conf * 100)}% conf
                </span>` : ''}
                <i class="fas fa-chevron-right" style="color:#6B7298;"></i>
            </div>
        </div>`;
    }).join('');
}

function _last7Days() {
    const names = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    return Array.from({length:7}, (_,i) => {
        const d = new Date();
        d.setDate(d.getDate() - (6 - i));
        return { date: d, label: i === 6 ? 'Today' : names[d.getDay()] };
    });
}
document.addEventListener('DOMContentLoaded', () => {
    if (!isAuthenticated()) {
        window.location.href = 'login.html';
        return;
    }
    loadDashboardData();
});

console.log('✅ Dashboard Real loaded');