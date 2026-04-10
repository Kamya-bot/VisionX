/**
 * VisionX – History Page
 * Reads prediction history from the real API (/api/v1/predictions/history).
 * localStorage is used only as a display-while-loading cache.
 */

let allPredictions = [];
let filteredPredictions = [];
let currentOffset = 0;
const PAGE_SIZE = 50;

async function loadHistory(append = false) {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    const container   = document.getElementById('historyContainer');
    const loading     = document.getElementById('loadingHistory');
    const emptyState  = document.getElementById('emptyState');
    const loadMoreBtn = document.getElementById('loadMoreBtn');

    if (loading) loading.style.display = 'block';
    if (emptyState) emptyState.style.display = 'none';

    try {
        const data = await apiCall(
            `/predictions/history?limit=${PAGE_SIZE}&offset=${currentOffset}`
        );

        const predictions = data.predictions || data.data || [];
        const total       = data.total || predictions.length;

        if (!append) {
            allPredictions = predictions;
        } else {
            allPredictions = [...allPredictions, ...predictions];
        }
        filteredPredictions = [...allPredictions];

        if (loading) loading.style.display = 'none';

        if (allPredictions.length === 0) {
            if (emptyState) emptyState.style.display = 'block';
            return;
        }

        displayPredictions(filteredPredictions);
        updateHistoryStats(allPredictions);

        // Show/hide "load more" button
        if (loadMoreBtn) {
            loadMoreBtn.style.display =
                allPredictions.length < total ? 'block' : 'none';
        }

        // Update cache for instant display on next load
        try {
            localStorage.setItem(
                'visionx_history_cache',
                JSON.stringify({ ts: Date.now(), predictions: allPredictions })
            );
        } catch (_) { /* quota exceeded – ignore */ }

    } catch (err) {
        console.error('Failed to load history:', err);
        if (loading) loading.style.display = 'none';

        // Graceful degradation: show cached data if available
        const cached = _loadCache();
        if (cached.length > 0) {
            allPredictions = cached;
            filteredPredictions = [...cached];
            displayPredictions(filteredPredictions);
            updateHistoryStats(allPredictions);
            _showBanner('Showing cached history — could not reach server.', 'warning');
        } else if (emptyState) {
            emptyState.style.display = 'block';
        }
    }
}

function _loadCache() {
    try {
        const raw = localStorage.getItem('visionx_history_cache');
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        // Cache expires after 5 minutes
        if (Date.now() - parsed.ts > 5 * 60 * 1000) return [];
        return parsed.predictions || [];
    } catch (_) { return []; }
}

function _showBanner(msg, type = 'info') {
    const colors = { warning: '#f59e0b', error: '#ef4444', info: '#4F8CFF' };
    const banner = document.createElement('div');
    banner.style.cssText = `
        padding:10px 16px;
        background:rgba(${type === 'warning' ? '245,158,11' : '79,140,255'},.15);
        border:1px solid ${colors[type]};
        border-radius:8px;
        color:${colors[type]};
        font-size:13px;
        margin-bottom:12px;
    `;
    banner.textContent = msg;
    const container = document.getElementById('historyContainer');
    if (container) container.insertAdjacentElement('beforebegin', banner);
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

    container.innerHTML = predictions.map(pred => {
        const confidence = Math.round((pred.confidence || 0) * 100);
        const date = pred.created_at
            ? new Date(pred.created_at).toLocaleDateString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric'
              })
            : '—';
        const recommended = pred.recommended_option_name || pred.recommendation || 'N/A';
        const clusterLabel = pred.cluster_label || (pred.cluster_id != null ? `Cluster ${pred.cluster_id}` : 'Unknown');
        const title = pred.title || pred.recommendation || 'Prediction';

        // Confidence colour
        const confColor = confidence >= 70 ? '#10b981' : confidence >= 40 ? '#f59e0b' : '#ef4444';

        return `
            <div class="history-item"
                style="padding:1.25rem;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:10px;margin-bottom:.75rem;cursor:pointer;transition:border-color .2s;"
                onmouseenter="this.style.borderColor='rgba(79,140,255,.4)'"
                onmouseleave="this.style.borderColor='rgba(255,255,255,.08)'"
                onclick="window.location.href='results.html?id=${pred.id || pred.prediction_id}'">
                <div style="display:flex;align-items:flex-start;gap:1rem;flex-wrap:wrap">
                    <div style="flex:1;min-width:200px">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                            <span style="background:rgba(79,140,255,.15);color:#4F8CFF;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500">
                                ${clusterLabel}
                            </span>
                            <span style="color:#9AA3C7;font-size:12px">${date}</span>
                        </div>
                        <div style="font-weight:600;color:#E6E8F2;margin-bottom:4px">${title}</div>
                        <div style="color:#4F8CFF;font-size:13px;margin-top:6px">
                            <i class="fas fa-trophy" style="margin-right:4px"></i>
                            Recommended: <strong>${recommended}</strong>
                        </div>
                        ${pred.model_version ? `<div style="color:#9AA3C7;font-size:11px;margin-top:4px">Model v${pred.model_version}</div>` : ''}
                    </div>
                    <div style="text-align:right;min-width:80px">
                        <div style="font-size:22px;font-weight:600;color:${confColor}">${confidence}%</div>
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
    const totalEl   = document.getElementById('totalPredictions');
    const avgEl     = document.getElementById('avgConfidence');
    const clusterEl = document.getElementById('topCluster');

    if (totalEl) totalEl.textContent = predictions.length;

    if (avgEl && predictions.length > 0) {
        const avg = predictions.reduce((s, p) => s + (p.confidence || 0), 0) / predictions.length;
        avgEl.textContent = Math.round(avg * 100) + '%';
    }

    if (clusterEl && predictions.length > 0) {
        const counts = {};
        predictions.forEach(p => {
            const key = p.cluster_label || (p.cluster_id != null ? `Cluster ${p.cluster_id}` : 'Unknown');
            counts[key] = (counts[key] || 0) + 1;
        });
        const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        clusterEl.textContent = top ? top[0] : '—';
    }
}

function filterHistory(query) {
    query = query.toLowerCase();
    filteredPredictions = allPredictions.filter(p =>
        (p.title || '').toLowerCase().includes(query) ||
        (p.recommendation || '').toLowerCase().includes(query) ||
        (p.cluster_label || '').toLowerCase().includes(query) ||
        (p.recommended_option_name || '').toLowerCase().includes(query)
    );
    displayPredictions(filteredPredictions);
}

async function loadMore() {
    currentOffset += PAGE_SIZE;
    await loadHistory(true);
}

document.addEventListener('DOMContentLoaded', () => {
    // Show cached data immediately while API loads
    const cached = _loadCache();
    if (cached.length > 0) {
        allPredictions = cached;
        filteredPredictions = [...cached];
        displayPredictions(filteredPredictions);
        updateHistoryStats(allPredictions);
    }

    loadHistory();

    const searchInput = document.getElementById('searchHistory');
    if (searchInput) {
        searchInput.addEventListener('input', e => {
            clearTimeout(window._searchTimer);
            window._searchTimer = setTimeout(() => filterHistory(e.target.value), 300);
        });
    }

    const refreshBtn = document.getElementById('refreshHistoryBtn');
    if (refreshBtn) refreshBtn.addEventListener('click', () => {
        currentOffset = 0;
        loadHistory();
    });

    const loadMoreBtn = document.getElementById('loadMoreBtn');
    if (loadMoreBtn) loadMoreBtn.addEventListener('click', loadMore);
});