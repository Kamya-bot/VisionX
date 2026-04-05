/**
 * VisionX — History Page
 * Loads prediction history from the real database via API.
 */

let allPredictions = [];
let filteredPredictions = [];

async function loadHistory() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    const container = document.getElementById('historyContainer');
    const loading = document.getElementById('loadingHistory');
    const emptyState = document.getElementById('emptyState');

    try {
        const data = await apiCall('/predictions/history?limit=100');
        allPredictions = data.items || [];
        filteredPredictions = [...allPredictions];

        if (loading) loading.style.display = 'none';

        if (allPredictions.length === 0) {
            if (emptyState) emptyState.style.display = 'block';
            return;
        }

        displayPredictions(filteredPredictions);
        updateHistoryStats(allPredictions);

    } catch (error) {
        if (loading) loading.style.display = 'none';
        if (container) container.innerHTML = `
            <div style="text-align:center;padding:2rem;color:#9AA3C7">
                <i class="fas fa-exclamation-circle" style="font-size:2rem;color:#ef4444;margin-bottom:1rem"></i>
                <p>Could not load history: ${error.message}</p>
            </div>`;
    }
}

function displayPredictions(predictions) {
    const container = document.getElementById('historyContainer');
    if (!container) return;

    if (predictions.length === 0) {
        const emptyState = document.getElementById('emptyState');
        if (emptyState) emptyState.style.display = 'block';
        return;
    }

    const clusterColors = { 0: '#6B7298', 1: '#4F8CFF', 2: '#10b981', 3: '#f59e0b' };

    container.innerHTML = predictions.map(pred => {
        const color = clusterColors[pred.cluster_id] || '#6B7298';
        const confidence = Math.round((pred.confidence || 0) * 100);
        const features = pred.features || {};
        const featStr = Object.entries(features).slice(0, 3)
            .map(([k, v]) => `<span style="color:#9AA3C7">${k}:</span> ${typeof v === 'number' ? v.toFixed(1) : v}`)
            .join(' &nbsp;|&nbsp; ');

        return `
            <div class="history-item">
                <div style="display:flex;align-items:flex-start;gap:1rem;flex-wrap:wrap">
                    <div style="flex:1;min-width:200px">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                            <span style="background:${color}22;color:${color};padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500">
                                ${pred.cluster_label || 'Unknown'}
                            </span>
                            <span style="color:#9AA3C7;font-size:12px">${formatDate(pred.created_at)}</span>
                        </div>
                        <div style="color:#9AA3C7;font-size:12px;margin-top:4px">${featStr || 'No feature data'}</div>
                        ${pred.recommendation ? `<div style="color:#E6E8F2;font-size:13px;margin-top:8px;font-style:italic">"${pred.recommendation}"</div>` : ''}
                    </div>
                    <div style="text-align:right;min-width:80px">
                        <div style="font-size:22px;font-weight:500;color:${color}">${confidence}%</div>
                        <div style="font-size:11px;color:#9AA3C7">confidence</div>
                    </div>
                </div>
            </div>`;
    }).join('');
}

function updateHistoryStats(predictions) {
    const totalEl = document.getElementById('totalPredictions');
    const avgEl = document.getElementById('avgConfidence');
    const clusterEl = document.getElementById('topCluster');

    if (totalEl) totalEl.textContent = predictions.length;
    if (avgEl && predictions.length > 0) {
        const avg = predictions.reduce((s, p) => s + (p.confidence || 0), 0) / predictions.length;
        avgEl.textContent = Math.round(avg * 100) + '%';
    }
    if (clusterEl && predictions.length > 0) {
        const counts = {};
        predictions.forEach(p => { counts[p.cluster_label] = (counts[p.cluster_label] || 0) + 1; });
        const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        clusterEl.textContent = top ? top[0] : '—';
    }
}

function filterHistory(query) {
    query = query.toLowerCase();
    filteredPredictions = allPredictions.filter(p =>
        (p.cluster_label || '').toLowerCase().includes(query) ||
        (p.recommendation || '').toLowerCase().includes(query) ||
        JSON.stringify(p.features || {}).toLowerCase().includes(query)
    );
    displayPredictions(filteredPredictions);
}

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    const searchInput = document.getElementById('searchHistory');
    if (searchInput) searchInput.addEventListener('input', debounce(e => filterHistory(e.target.value), 300));
    const refreshBtn = document.getElementById('refreshHistoryBtn');
    if (refreshBtn) refreshBtn.addEventListener('click', loadHistory);
});