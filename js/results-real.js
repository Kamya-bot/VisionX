/**
 * Results Page - Real Data
 * Reads comparison result from localStorage and displays it
 */

async function loadResults() {
    const user = getCurrentUser();
    if (!user) {
        window.location.href = 'login.html';
        return;
    }

    // Set user initials
    const initials = document.getElementById('userInitials');
    if (initials && user.full_name) {
        initials.textContent = user.full_name.charAt(0).toUpperCase();
    }

    // Get comparison ID from URL
    const params = new URLSearchParams(window.location.search);
    const compId = params.get('id');

    // Load from localStorage
    const allComparisons = JSON.parse(localStorage.getItem('comparisons') || '[]');
    const comparison = compId
        ? allComparisons.find(c => c.id === compId)
        : allComparisons[allComparisons.length - 1];

    if (!comparison) {
        document.getElementById('recommendationCard').innerHTML = `
            <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                <i class="fas fa-exclamation-circle" style="font-size: 2rem; margin-bottom: 1rem;"></i>
                <p>No comparison found. <a href="comparison.html" style="color: var(--primary);">Create one now</a></p>
            </div>`;
        return;
    }

    const result = comparison.result;

    // Show fallback warning if ML was unavailable
    if (result.is_fallback) {
        document.getElementById('fallbackWarning').style.display = 'block';
    }

    // Render recommendation card
    const confidence = Math.round((result.confidence || 0) * 100);
    document.getElementById('recommendationCard').innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="width: 56px; height: 56px; border-radius: 50%; background: linear-gradient(135deg, #4F8CFF, #7B61FF); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0;">
                🤖
            </div>
            <div>
                <h2 style="margin: 0;">AI Recommendation</h2>
                <p style="color: var(--text-secondary); margin: 0;">${comparison.title || 'Comparison'}</p>
            </div>
        </div>
        <div style="background: rgba(79,140,255,0.1); border: 1px solid rgba(79,140,255,0.3); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.25rem;">RECOMMENDED CHOICE</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #4F8CFF;">${result.recommended_option_name || 'N/A'}</div>
            <div style="margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                <div style="flex: 1; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                    <div style="width: ${confidence}%; height: 100%; background: linear-gradient(90deg, #4F8CFF, #7B61FF); border-radius: 4px;"></div>
                </div>
                <span style="font-weight: 600; color: #4F8CFF;">${confidence}% confidence</span>
            </div>
        </div>
        <p style="color: var(--text-secondary); line-height: 1.6;">${result.reasoning || ''}</p>
        ${result.user_cluster ? `<div style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-secondary);">Your profile: <strong style="color: var(--text-primary);">${result.user_cluster}</strong></div>` : ''}
    `;

    // Render all options breakdown
    const optionsEl = document.getElementById('optionsBreakdown');
    const allOptions = comparison.options || [];
    const alts = result.alternative_options || [];

    optionsEl.innerHTML = allOptions.map((opt, i) => {
        const isWinner = opt.id === result.recommended_option_id || opt.name === result.recommended_option_name;
        const alt = alts.find(a => a.id === opt.id || a.name === opt.name);
        const score = isWinner ? confidence : alt ? Math.round((alt.score || 0) * 100) : '—';

        return `
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem; background: rgba(255,255,255,0.03); border: 1px solid ${isWinner ? 'rgba(79,140,255,0.4)' : 'rgba(255,255,255,0.08)'}; border-radius: 8px; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    ${isWinner ? '<span style="font-size: 1.2rem;">🏆</span>' : `<span style="width: 24px; height: 24px; border-radius: 50%; background: rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; font-size: 0.8rem;">${i + 1}</span>`}
                    <div>
                        <div style="font-weight: 600;">${opt.name}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            Quality: ${opt.features?.quality_score || '—'}/10 &nbsp;·&nbsp;
                            Price: $${opt.features?.price?.toLocaleString() || '—'}
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 700; color: ${isWinner ? '#4F8CFF' : 'var(--text-secondary)'};">${score}%</div>
                    ${isWinner ? '<div style="font-size: 0.75rem; color: #4F8CFF;">Recommended</div>' : ''}
                </div>
            </div>
        `;
    }).join('');

    // Render feature importance
    const featEl = document.getElementById('featureImportance');
    const features = result.feature_importance || [];
    featEl.innerHTML = features.length ? features.map(f => {
        const pct = Math.round((f.importance || 0) * 100);
        const label = (f.feature_name || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return `
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-weight: 500;">${label}</span>
                    <span style="color: var(--text-secondary);">${pct}%</span>
                </div>
                <div style="height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                    <div style="width: ${pct}%; height: 100%; background: linear-gradient(90deg, #4F8CFF, #7B61FF); border-radius: 4px;"></div>
                </div>
            </div>
        `;
    }).join('') : '<p style="color: var(--text-secondary);">No feature importance data available.</p>';

    // Render score chart
    const ctx = document.getElementById('scoreChart')?.getContext('2d');
    if (ctx && allOptions.length) {
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: allOptions.map(o => o.name),
                datasets: [
                    {
                        label: 'Quality Score',
                        data: allOptions.map(o => o.features?.quality_score || 0),
                        backgroundColor: 'rgba(79, 140, 255, 0.7)',
                        borderRadius: 6
                    },
                    {
                        label: 'Price (÷1000)',
                        data: allOptions.map(o => (o.features?.price || 0) / 1000),
                        backgroundColor: 'rgba(123, 97, 255, 0.7)',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#a0aec0' } } },
                scales: {
                    x: { ticks: { color: '#a0aec0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#a0aec0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (!isAuthenticated()) {
        window.location.href = 'login.html';
        return;
    }
    loadResults();
});

console.log('✅ Results Real Data loaded');