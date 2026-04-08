/**
 * VisionX - History Page
 * Loads prediction history from localStorage.
 */

let allPredictions = [];
let filteredPredictions = [];

function loadHistory() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    const container = document.getElementById('historyContainer');
    const loading = document.getElementById('loadingHistory');
    const emptyState = document.getElementById('emptyState');

    if (loading) loading.style.display = 'none';

    // Read from localStorage
    const allComparisons = JSON.parse(localStorage.getItem('comparisons') || '[]');
    // Filter to current user only
    allPredictions = allComparisons.filter(c => c.user_id === user.id);
    filteredPredictions = [...allPredictions];

    if (allPredictions.length === 0) {
        if (emptyState) emptyState.style.display = 'block';
        return;
    }

    displayPredictions(filteredPredictions);
    updateHistoryStats(allPredictions);
}

function displayPredictions(predictions) {
    const container = document.getElementById('historyContainer');
    if (!container) return;

    const emptyState = document.getElementById('emptyState');

    if (predictions.length === 0) {
        if (emptyState) emptyState.style.display = 'block';
        container.innerHTML = '';
        return;
    }

    if (emptyState) emptyState.style.display = 'none';

    container.innerHTML = predictions.map(comp => {
        const result = comp.result || {};
        const confidence = Math.round((result.confidence || 0) * 100);
        const date = new Date(comp.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        const recommended = result.recommended_option_name || 'N/A';
        const options = (comp.options || []).map(o => o.name).join(', ');

        return `
            <div class="history-item" style="padding: 1.25rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; margin-bottom: 0.75rem; cursor: pointer;"
                onclick="window.location.href='results.html?id=${comp.id}'">
                <div style="display:flex;align-items:flex-start;gap:1rem;flex-wrap:wrap">
                    <div style="flex:1;min-width:200px">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                            <span style="background:rgba(79,140,255,0.15);color:#4F8CFF;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500">
                                ${comp.category || 'General'}
                            </span>
                            <span style="color:#9AA3C7;font-size:12px">${date}</span>
                        </div>
                        <div style="font-weight:600;color:#E6E8F2;margin-bottom:4px">${comp.title || 'Untitled Comparison'}</div>
                        <div style="color:#9AA3C7;font-size:12px">Options: ${options}</div>
                        <div style="color:#4F8CFF;font-size:13px;margin-top:6px">
                            <i class="fas fa-trophy" style="margin-right:4px"></i> Recommended: <strong>${recommended}</strong>
                        </div>
                    </div>
                    <div style="text-align:right;min-width:80px">
                        <div style="font-size:22px;font-weight:600;color:#4F8CFF">${confidence}%</div>
                        <div style="font-size:11px;color:#9AA3C7">confidence</div>
                        <div style="margin-top:6px;font-size:11px;color:#9AA3C7">
                            <i class="fas fa-chevron-right"></i>
                        </div>
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
        const avg = predictions.reduce((s, p) => s + ((p.result?.confidence) || 0), 0) / predictions.length;
        avgEl.textContent = Math.round(avg * 100) + '%';
    }
    if (clusterEl && predictions.length > 0) {
        const categories = {};
        predictions.forEach(p => { categories[p.category || 'General'] = (categories[p.category || 'General'] || 0) + 1; });
        const top = Object.entries(categories).sort((a, b) => b[1] - a[1])[0];
        clusterEl.textContent = top ? top[0] : '-';
    }
}

function filterHistory(query) {
    query = query.toLowerCase();
    filteredPredictions = allPredictions.filter(p =>
        (p.title || '').toLowerCase().includes(query) ||
        (p.category || '').toLowerCase().includes(query) ||
        (p.result?.recommended_option_name || '').toLowerCase().includes(query) ||
        (p.options || []).some(o => o.name.toLowerCase().includes(query))
    );
    displayPredictions(filteredPredictions);
}

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    const searchInput = document.getElementById('searchHistory');
    if (searchInput) searchInput.addEventListener('input', e => {
        clearTimeout(window._searchTimer);
        window._searchTimer = setTimeout(() => filterHistory(e.target.value), 300);
    });
    const refreshBtn = document.getElementById('refreshHistoryBtn');
    if (refreshBtn) refreshBtn.addEventListener('click', loadHistory);
});