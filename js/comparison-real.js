/**
 * Comparison Page with Backend Integration
 * Handles comparison creation and ML prediction
 */

let comparisonOptions = [];
let currentStep = 1;

/**
 * Initialize Comparison Page
 */
function initializeComparison() {
    const user = getCurrentUser();
    
    if (!user) {
        window.location.href = 'login.html';
        return;
    }
    
    // Initialize first two options
    addOption();
    addOption();
    
    // Bind form submission
    const form = document.getElementById('comparisonForm');
    if (form) {
        form.addEventListener('submit', handleComparisonSubmit);
    }
    
    // Bind add option button
    const addBtn = document.getElementById('addOptionBtn');
    if (addBtn) {
        addBtn.addEventListener('click', addOption);
    }
}

/**
 * Add Option
 */
function addOption() {
    const container = document.getElementById('optionsContainer');
    if (!container) return;
    
    const optionNumber = container.children.length + 1;
    
    if (optionNumber > 5) {
        showNotification('Maximum 5 options allowed', 'error');
        return;
    }
    
    const optionHtml = `
        <div class="option-card glass-card" data-option-id="${optionNumber}" style="padding: 1.5rem; margin-bottom: 1rem; position: relative;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4>Option ${optionNumber}</h4>
                ${optionNumber > 2 ? `
                    <button type="button" class="btn-icon-danger" onclick="removeOption(${optionNumber})" style="background: rgba(255, 79, 79, 0.1); border: none; padding: 0.5rem; border-radius: 4px; cursor: pointer;">
                        <i class="fas fa-trash" style="color: #ff4f4f;"></i>
                    </button>
                ` : ''}
            </div>
            
            <div class="form-group">
                <label class="form-label">Option Name *</label>
                <input type="text" class="form-input" id="optionName${optionNumber}" placeholder="e.g., Job at Company A" required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Price/Cost ($)</label>
                <input type="number" class="form-input" id="optionPrice${optionNumber}" placeholder="e.g., 75000" min="0">
            </div>
            
            <div class="form-group">
                <label class="form-label">Quality Score (1-10)</label>
                <input type="number" class="form-input" id="optionQuality${optionNumber}" placeholder="e.g., 8" min="1" max="10">
            </div>
            
            <div class="form-group">
                <label class="form-label">Delivery/Start Time (days)</label>
                <input type="number" class="form-input" id="optionDelivery${optionNumber}" placeholder="e.g., 30" min="0">
            </div>
            
            <div class="form-group">
                <label class="form-label">Additional Notes</label>
                <textarea class="form-input" id="optionNotes${optionNumber}" rows="2" placeholder="Any additional information..."></textarea>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', optionHtml);
}

/**
 * Remove Option
 */
function removeOption(optionNumber) {
    const option = document.querySelector(`[data-option-id="${optionNumber}"]`);
    if (option) {
        option.remove();
        renumberOptions();
    }
}

/**
 * Renumber Options after removal
 */
function renumberOptions() {
    const container = document.getElementById('optionsContainer');
    if (!container) return;
    
    Array.from(container.children).forEach((option, index) => {
        const optionNumber = index + 1;
        option.setAttribute('data-option-id', optionNumber);
        
        const title = option.querySelector('h4');
        if (title) title.textContent = `Option ${optionNumber}`;
        
        option.querySelectorAll('input, textarea').forEach(input => {
            const idPrefix = input.id.replace(/\d+$/, '');
            input.id = idPrefix + optionNumber;
        });
    });
}

/**
 * Handle Comparison Submit
 */
async function handleComparisonSubmit(event) {
    event.preventDefault();
    
    const user = getCurrentUser();
    if (!user) {
        window.location.href = 'login.html';
        return;
    }
    
    const submitBtn = event.target.querySelector('button[type="submit"]');
    
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    }
    
    try {
        const comparisonTitle = document.getElementById('comparisonTitle')?.value || 'Untitled Comparison';
        const comparisonCategory = document.getElementById('comparisonCategory')?.value || 'Other';
        
        const container = document.getElementById('optionsContainer');
        const options = [];
        
        Array.from(container.children).forEach((optionCard, index) => {
            const num = index + 1;
            const name = document.getElementById(`optionName${num}`)?.value;
            if (!name) return;
            
            options.push({
                id: num.toString(),
                name: name,
                features: {
                    price: parseFloat(document.getElementById(`optionPrice${num}`)?.value || 0),
                    quality_score: parseInt(document.getElementById(`optionQuality${num}`)?.value || 5),
                    delivery_time: parseInt(document.getElementById(`optionDelivery${num}`)?.value || 0),
                    feature_count: parseInt(document.getElementById(`optionDelivery${num}`)?.value || 0),
                    brand_score: 5.0,
                    availability: 1.0
                },
                notes: document.getElementById(`optionNotes${num}`)?.value || ''
            });
        });
        
        if (options.length < 2) {
            showNotification('Please add at least 2 options', 'error');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-magic"></i> Get AI Recommendation';
            }
            return;
        }
        
        // Call ML prediction API
        let result;
        let usedFallback = false;

        try {
            result = await apiCall('/ml/predict', {
                method: 'POST',
                body: { user_id: user.id, options: options }
            });
            console.log('ML prediction result:', result);
        } catch (error) {
            console.error('ML API failed:', error);
            usedFallback = true;

            // Score options by quality/price ratio as a real fallback
            const scored = options.map(opt => {
                const q = opt.features.quality_score || 5;
                const p = opt.features.price || 1;
                return { opt, score: q / Math.log(p + 2) };
            }).sort((a, b) => b.score - a.score);

            const winner = scored[0].opt;
            result = {
                recommended_option_id: winner.id,
                recommended_option_name: winner.name,
                confidence: scored[0].score / (scored[0].score + scored[1].score),
                reasoning: "ML service temporarily unavailable. This recommendation is based on quality-to-cost ratio analysis.",
                alternative_options: scored.slice(1).map(s => ({
                    id: s.opt.id,
                    name: s.opt.name,
                    score: s.score / (scored[0].score + s.score),
                    reason: 'Alternative based on quality-to-cost analysis'
                })),
                feature_importance: [
                    { feature_name: 'quality_score', importance: 0.45 },
                    { feature_name: 'price', importance: 0.35 },
                    { feature_name: 'delivery_time', importance: 0.20 }
                ],
                user_cluster: 'Analytical Researcher',
                is_fallback: true
            };
        }
        
        // Save to localStorage
        const comparison = {
            id: 'comp_' + Date.now(),
            user_id: user.id,
            title: comparisonTitle,
            category: comparisonCategory,
            options: options,
            result: result,
            created_at: new Date().toISOString()
        };
        
        const allComparisons = JSON.parse(localStorage.getItem('comparisons') || '[]');
        allComparisons.push(comparison);
        localStorage.setItem('comparisons', JSON.stringify(allComparisons));
        
        if (usedFallback) {
            showNotification('⚠️ ML service unavailable — used local analysis. Results may vary.', 'warning');
        } else {
            showNotification('✅ AI analysis complete! Redirecting...', 'success');
        }
        
        setTimeout(() => {
            window.location.href = `results.html?id=${comparison.id}`;
        }, 1000);
        
    } catch (error) {
        console.error('Failed to process comparison:', error);
        showNotification('Failed to analyze comparison: ' + error.message, 'error');
        
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-magic"></i> Get AI Recommendation';
        }
    }
}

/**
 * Initialize on page load
 * FIX: replaced broken checkAuth() with isAuthenticated() from api-config.js
 */
document.addEventListener('DOMContentLoaded', () => {
    if (!isAuthenticated()) {
        window.location.href = 'login.html';
        return;
    }
    initializeComparison();
});

console.log('✅ Comparison Real Integration loaded');