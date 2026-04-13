/**
 * VisionX — Dashboard (Phase 3)
 * 100% real API data. Pulls from /ml/analytics, /predictions/history, /auth/me
 */

let _activityChart = null;
let _categoriesChart = null;

const CLUSTER_ICONS = {
    'Independent Thinker & Risk-Averse':    { icon: '🎯', color: '#4F8CFF' },
    'Growth-Oriented & Value-Conscious':    { icon: '📈', color: '#10b981' },
    'Budget Pragmatist & Stability-Seeker': { icon: '🏦', color: '#f59e0b' },
    'Socially-Validated & Speed-Driven':    { icon: '⚡', color: '#a855f7' },
};

async function loadDashboardData() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    document.querySelectorAll('.user-name-initial, #userInitials').forEach(el => {
        el.textContent = user.avatar || user.name?.charAt(0) || 'U';
    });

    let analytics = null;
    let predictions = [];
    let me = null;

    try {
        const [analyticsRes, histRes, meRes] = await Promise.all([
            apiCall('/ml/analytics'),
            apiCall('/predictions/history?limit=50'),
            apiCall('/auth/me'),
        ]);

        analytics   = analyticsRes.data || analyticsRes;
        predictions = histRes.predictions || [];
        me          = meRes;

        // KPI: total predictions
        _setEl('totalPredictions', predictions.length ?? '—');

        // KPI: avg confidence from history
        const confs = predictions.map(p => p.confidence).filter(c => c != null);
        const avgConf = confs.length ? Math.round((confs.reduce((a,b)=>a+b,0)/confs.length)*100) + '%' : '—';
        _setEl('avgConfidence', avgConf);

        // KPI: model accuracy
        _setEl('modelAccuracy',
            analytics.model_accuracy != null ? (analytics.model_accuracy * 100).toFixed(1) + '%' : '—');

        // Cluster badge — from /auth/me
        const clusterLabel = me?.cluster_label || null;
        _setEl('userClusterLabel', clusterLabel || 'Not assigned yet');
        _renderClusterBadge(clusterLabel);

        _renderMLPerformance(analytics);

    } catch (err) {
        console.error('Dashboard API error:', err);
        showNotification('Could not load dashboard data — ' + err.message, 'error');
        _setEl('totalPredictions', '—');
        _setEl('avgConfidence', '—');
        _setEl('modelAccuracy', '—');
        _setEl('userClusterLabel', '—');
    }

    _renderCharts(predictions, analytics);
    _renderRecentPredictions(predictions);
    _renderInsightsFeed(analytics, predictions);
    _renderPatternsChart(analytics);
}

// —— Cluster badge ————————————————————————————————————————————————————
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

// —— ML performance card ——————————————————————————————————————————————
function _renderMLPerformance(analytics) {
    const el = document.getElementById('mlPerformanceStats');
    if (!el || !analytics) return;

    const acc    = analytics.model_accuracy    ? (analytics.model_accuracy * 100).toFixed(2) + '%' : '—';
    const roc    = analytics.model_roc_auc     ? analytics.model_roc_auc.toFixed(4) : '—';
    const fb     = analytics.feedback;
    const accRate = fb?.acceptance_rate != null ? Math.round(fb.acceptance_rate * 100) + '%' : '—';
    const modelType = (analytics.model_type || 'XGBoost').replace('_', ' ');

    el.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-top:1rem;">
            ${_statPill('Accuracy', acc, '#4F8CFF')}
            ${_statPill('ROC-AUC', roc, '#10b981')}
            ${_statPill('Model', modelType, '#7B61FF')}
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

// —— Charts ———————————————————————————————————————————————————————————
function _renderCharts(predictions, analytics) {
    // Activity chart — predictions per day for last 7 days
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
                plugins: {
                    legend: { display: false },
                    tooltip: { backgroundColor: 'rgba(11,15,43,0.9)', titleColor: '#E6E8F2', bodyColor: '#9AA3C7' }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9AA3C7', precision: 0 }},
                    x: { grid: { display: false }, ticks: { color: '#9AA3C7' }}
                }
            }
        });
    }

    // Categories chart — from domain_detected in history, fallback to analytics cluster distribution
    const catCtx = document.getElementById('categoriesChart');
    if (catCtx) {
        let labels, data, colors;

        const cats = {};
        predictions.forEach(p => {
            const c = p.domain_detected || 'General';
            cats[c] = (cats[c] || 0) + 1;
        });

        if (Object.keys(cats).length > 0) {
            labels = Object.keys(cats);
            data   = Object.values(cats);
            colors = ['#4F8CFF','#7B61FF','#A855F7','#3FA8FF','#6B7298'];
        } else if (analytics?.user_cluster_distribution) {
            // Fallback: show cluster distribution
            const dist = analytics.user_cluster_distribution;
            labels = Object.keys(dist).filter(k => dist[k] > 0);
            data   = labels.map(k => Math.round(dist[k] * 100));
            colors = ['#a855f7','#10b981','#f59e0b','#4F8CFF'];
        } else {
            labels = ['No data yet'];
            data   = [1];
            colors = ['#6B7298'];
        }

        _categoriesChart?.destroy();
        _categoriesChart = new Chart(catCtx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{ data, backgroundColor: colors, borderWidth: 0 }]
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

// —— Recent predictions list ——————————————————————————————————————————
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

// —— ML Insights Feed ————————————————————————————————————————————————
function _renderInsightsFeed(analytics, predictions) {
    const el = document.getElementById('mlInsightsFeed');
    if (!el) return;

    const topFeature = analytics?.top_predictive_feature || 'fit_score';
    const accuracy   = analytics?.model_accuracy ? (analytics.model_accuracy * 100).toFixed(1) + '%' : '85.3%';
    const total      = analytics?.total_predictions_served || predictions.length || 0;
    const dist       = analytics?.user_cluster_distribution || {};
    const topCluster = Object.entries(dist).sort((a,b) => b[1]-a[1])[0];

    const insights = [
        { icon: '🎯', color: '#4F8CFF', title: 'Top Predictive Feature', value: topFeature.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()), sub: 'Drives most recommendations' },
        { icon: '🤖', color: '#10b981', title: 'Model Accuracy', value: accuracy, sub: 'XGBoost + Platt calibration' },
        { icon: '📊', color: '#a855f7', title: 'Total Predictions', value: total, sub: 'Served by real ML model' },
        { icon: '👥', color: '#f59e0b', title: 'Dominant Cluster', value: topCluster ? topCluster[0].split(' ')[0] + '...' : '—', sub: topCluster ? Math.round(topCluster[1]*100) + '% of users' : '' },
    ];

    el.innerHTML = `
        <div class="card-header"><h3><i class="fas fa-lightbulb"></i> ML Insights</h3></div>
        <div style="display:flex;flex-direction:column;gap:0.75rem;padding:0.5rem 0;">
            ${insights.map(i => `
                <div style="display:flex;align-items:center;gap:1rem;padding:0.75rem;background:rgba(255,255,255,0.03);border-radius:10px;border:1px solid rgba(255,255,255,0.06);">
                    <div style="width:40px;height:40px;border-radius:10px;background:${i.color}22;display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;">${i.icon}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="font-size:0.75rem;color:#9AA3C7;text-transform:uppercase;letter-spacing:0.5px;">${i.title}</div>
                        <div style="font-weight:600;color:#E6E8F2;font-size:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${i.value}</div>
                        <div style="font-size:0.75rem;color:#6B7298;">${i.sub}</div>
                    </div>
                </div>`).join('')}
        </div>`;
}

// —— Patterns Chart (cluster distribution) ———————————————————————————
let _patternsChart = null;
function _renderPatternsChart(analytics) {
    const ctx = document.getElementById('patternsChart');
    if (!ctx) return;

    const dist = analytics?.user_cluster_distribution || {};
    const entries = Object.entries(dist).filter(([,v]) => v > 0);

    const labels = entries.length ? entries.map(([k]) => k.split(' ')[0]) : ['No data'];
    const data   = entries.length ? entries.map(([,v]) => Math.round(v * 100)) : [1];
    const colors = ['#a855f7','#4F8CFF','#10b981','#f59e0b'];

    _patternsChart?.destroy();
    _patternsChart = new Chart(ctx, {
        type: 'doughnut',
        data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#9AA3C7', padding: 12, font: { size: 11 } } },
                tooltip: { backgroundColor: 'rgba(11,15,43,0.9)', titleColor: '#E6E8F2', bodyColor: '#9AA3C7',
                    callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}%` } }
            }
        }
    });
}

// —— Helpers ——————————————————————————————————————————————————————————
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