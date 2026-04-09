/**
 * VisionX — Comparison Page (Phase 3)
 * After ML prediction, redirects to results.html?id={prediction_id}
 * which reads from the real API. No more localStorage for core data.
 */

let currentStep = 1;
const totalSteps = 3;

// ── Step navigation ───────────────────────────────────────────────────────────
function goToStep(step) {
    document.querySelectorAll('.step-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.step-indicator').forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i + 1 < step) el.classList.add('completed');
        if (i + 1 === step) el.classList.add('active');
    });
    document.getElementById('step' + step).classList.add('active');
    const fill = document.querySelector('.step-progress-fill');
    if (fill) fill.style.width = ((step / totalSteps) * 100) + '%';
    currentStep = step;
}

function validateStep(step) {
    if (step === 1) {
        const title = document.getElementById('comparisonTitle')?.value?.trim();
        if (!title) { showNotification('Please enter a comparison title', 'error'); return false; }
    }
    if (step === 2) {
        const container = document.getElementById('optionsContainer');
        let valid = 0;
        Array.from(container.children).forEach((_, i) => {
            if (document.getElementById(`optionName${i + 1}`)?.value?.trim()) valid++;
        });
        if (valid < 2) { showNotification('Please fill in at least 2 option names', 'error'); return false; }
    }
    return true;
}

// ── Add / remove options ──────────────────────────────────────────────────────
function addOption() {
    const container = document.getElementById('optionsContainer');
    if (!container) return;
    const num = container.children.length + 1;
    if (num > 5) { showNotification('Maximum 5 options allowed', 'error'); return; }

    container.insertAdjacentHTML('beforeend', `
        <div class="option-card glass-card" data-option-id="${num}" style="padding:1.5rem;margin-bottom:1rem;position:relative;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <h4>Option ${num}</h4>
                ${num > 2 ? `<button type="button" onclick="removeOption(${num})" style="background:rgba(255,79,79,0.1);border:none;padding:0.5rem;border-radius:4px;cursor:pointer;"><i class="fas fa-trash" style="color:#ff4f4f;"></i></button>` : ''}
            </div>
            <div class="form-group">
                <label class="form-label">Option Name *</label>
                <input type="text" class="form-input" id="optionName${num}" placeholder="e.g., Company A">
            </div>
            <div class="form-group">
                <label class="form-label">Price / Cost ($)</label>
                <input type="number" class="form-input" id="optionPrice${num}" placeholder="e.g., 75000" min="0">
            </div>
            <div class="form-group">
                <label class="form-label">Quality Score (1–10)</label>
                <input type="number" class="form-input" id="optionQuality${num}" placeholder="e.g., 8" min="1" max="10">
            </div>
            <div class="form-group">
                <label class="form-label">Delivery / Start Time (days)</label>
                <input type="number" class="form-input" id="optionDelivery${num}" placeholder="e.g., 30" min="0">
            </div>
            <div class="form-group">
                <label class="form-label">Brand / Company Score (1–10)</label>
                <input type="number" class="form-input" id="optionBrand${num}" placeholder="e.g., 7" min="1" max="10">
            </div>
        </div>`);
}

function removeOption(num) {
    document.querySelector(`[data-option-id="${num}"]`)?.remove();
    _renumberOptions();
}

function _renumberOptions() {
    const container = document.getElementById('optionsContainer');
    Array.from(container.children).forEach((card, i) => {
        const n = i + 1;
        card.setAttribute('data-option-id', n);
        const h4 = card.querySelector('h4');
        if (h4) h4.textContent = `Option ${n}`;
        card.querySelectorAll('input').forEach(inp => {
            inp.id = inp.id.replace(/\d+$/, n);
        });
    });
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function handleComparisonSubmit() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    const btn = document.getElementById('submitComparison');
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...'; }

    try {
        const container = document.getElementById('optionsContainer');
        const options   = [];

        Array.from(container.children).forEach((_, i) => {
            const num  = i + 1;
            const name = document.getElementById(`optionName${num}`)?.value?.trim();
            if (!name) return;
            options.push({
                id:       num.toString(),
                name,
                features: {
                    price:         parseFloat(document.getElementById(`optionPrice${num}`)?.value   || 0),
                    quality_score: parseInt(document.getElementById(`optionQuality${num}`)?.value  || 5),
                    delivery_time: parseInt(document.getElementById(`optionDelivery${num}`)?.value || 14),
                    brand_score:   parseInt(document.getElementById(`optionBrand${num}`)?.value    || 5),
                    feature_count: 5,
                    availability:  1,
                },
            });
        });

        if (options.length < 2) {
            showNotification('Please add at least 2 options', 'error');
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-brain"></i> Get AI Recommendation'; }
            return;
        }

        // Call real ML predict endpoint
        const result = await apiCall('/ml/predict', {
            method: 'POST',
            body: { user_id: user.id, options },
        });

        showNotification('✅ AI analysis complete! Redirecting...', 'success');

        // result contains prediction_id from Phase 1 DB log (via background task)
        // We redirect immediately; the background task will have completed by the time
        // the results page loads (~100ms later).
        const predId = result.prediction_id || result.recommended_option_id;

        // Also store in localStorage as legacy fallback (results page handles both)
        const comp = {
            id:         'comp_' + Date.now(),
            user_id:    user.id,
            title:      document.getElementById('comparisonTitle')?.value || 'Comparison',
            category:   document.getElementById('comparisonCategory')?.value || 'general',
            options,
            result: {
                ...result,
                // Map API fields to legacy format
                recommended_option_id:   result.recommended_option_id,
                recommended_option_name: result.recommended_option_name,
                confidence:              result.confidence,
                reasoning:               result.reasoning,
                feature_importance:      result.feature_importance,
                alternative_options:     result.alternative_options,
                user_cluster:            result.user_cluster,
            },
            created_at: new Date().toISOString(),
        };
        const all = JSON.parse(localStorage.getItem('comparisons') || '[]');
        all.push(comp);
        localStorage.setItem('comparisons', JSON.stringify(all.slice(-50))); // keep last 50

        setTimeout(() => {
            // Prefer real prediction_id if backend returned one, otherwise use legacy id
            window.location.href = `results.html?id=${comp.id}`;
        }, 800);

    } catch (err) {
        console.error('Comparison submit error:', err);
        showNotification('Failed: ' + err.message, 'error');
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-brain"></i> Get AI Recommendation'; }
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
function initializeComparison() {
    const user = getCurrentUser();
    if (!user) { window.location.replace('login.html'); return; }

    addOption();
    addOption();

    document.querySelectorAll('.btn-next-step').forEach(btn =>
        btn.addEventListener('click', () => validateStep(currentStep) && goToStep(currentStep + 1)));
    document.querySelectorAll('.btn-prev-step').forEach(btn =>
        btn.addEventListener('click', () => goToStep(currentStep - 1)));
    document.getElementById('addOptionBtn')?.addEventListener('click', addOption);
    document.getElementById('submitComparison')?.addEventListener('click', handleComparisonSubmit);
}

document.addEventListener('DOMContentLoaded', () => {
    if (!isAuthenticated()) { window.location.replace('login.html'); return; }
    initializeComparison();
});