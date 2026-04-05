// ============================================
// VisionX - Working Comparison Form with ML Backend
// ============================================

class ComparisonForm {
    constructor() {
        this.options = [];
        this.currentStep = 1;
        this.totalSteps = 3;
        this.init();
    }

    init() {
        this.setupStepNavigation();
        this.setupOptionManagement();
        this.setupFormSubmission();
        console.log('📋 Comparison form initialized');
    }

    setupStepNavigation() {
        const nextBtns = document.querySelectorAll('.btn-next-step');
        const prevBtns = document.querySelectorAll('.btn-prev-step');

        nextBtns.forEach(btn => {
            btn.addEventListener('click', () => this.nextStep());
        });

        prevBtns.forEach(btn => {
            btn.addEventListener('click', () => this.prevStep());
        });
    }

    setupOptionManagement() {
        const addOptionBtn = document.getElementById('addOptionBtn');
        if (addOptionBtn) {
            addOptionBtn.addEventListener('click', () => this.addOption());
        }

        // Add initial 2 options
        this.addOption();
        this.addOption();
    }

    addOption() {
        const optionId = 'option_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
        const optionIndex = this.options.length;

        this.options.push({ id: optionId, name: '', features: {} });

        const optionHTML = `
            <div class="option-card glass-card" data-option-id="${optionId}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0;">Option ${optionIndex + 1}</h3>
                    <button type="button" class="btn btn-danger btn-sm remove-option-btn" data-option-id="${optionId}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>

                <div class="form-group">
                    <label class="form-label">Option Name *</label>
                    <input type="text" class="form-input option-name"
                           placeholder="e.g., Google, iPhone 15, Company A"
                           data-option-id="${optionId}" required>
                </div>

                <div class="form-group">
                    <label class="form-label">Price / Salary / Cost ($) *</label>
                    <input type="number" class="form-input option-price"
                           placeholder="e.g., 120000"
                           min="1" data-option-id="${optionId}" required>
                </div>

                <div class="form-group">
                    <label class="form-label">Quality Score (1-10) *</label>
                    <input type="number" class="form-input option-quality"
                           placeholder="Rate quality from 1-10"
                           min="1" max="10" data-option-id="${optionId}" required>
                </div>

                <div class="form-group">
                    <label class="form-label">Number of Features / Benefits *</label>
                    <input type="number" class="form-input option-feature-count"
                           placeholder="e.g., how many perks/features does it have"
                           min="0" data-option-id="${optionId}" required>
                </div>

                <div class="form-group">
                    <label class="form-label">Brand / Reputation Score (1-10)</label>
                    <input type="number" class="form-input option-brand"
                           placeholder="Rate brand reputation 1-10 (default: 5)"
                           min="1" max="10" data-option-id="${optionId}">
                </div>
            </div>
        `;

        const container = document.getElementById('optionsContainer');
        if (container) {
            container.insertAdjacentHTML('beforeend', optionHTML);
            this.setupRemoveButton(optionId);
            this.setupOptionInputs(optionId);
        }
    }

    setupRemoveButton(optionId) {
        const btn = document.querySelector(`.remove-option-btn[data-option-id="${optionId}"]`);
        if (btn) {
            btn.addEventListener('click', () => {
                if (this.options.length > 2) {
                    this.options = this.options.filter(o => o.id !== optionId);
                    document.querySelector(`.option-card[data-option-id="${optionId}"]`)?.remove();
                } else {
                    alert('You must have at least 2 options to compare');
                }
            });
        }
    }

    setupOptionInputs(optionId) {
        const get = (cls) => document.querySelector(`.${cls}[data-option-id="${optionId}"]`);

        get('option-name')?.addEventListener('input', e => {
            const opt = this.options.find(o => o.id === optionId);
            if (opt) opt.name = e.target.value;
        });

        get('option-price')?.addEventListener('input', e => {
            this.setFeature(optionId, 'price', parseFloat(e.target.value) || 0);
        });

        get('option-quality')?.addEventListener('input', e => {
            this.setFeature(optionId, 'quality_score', parseFloat(e.target.value) || 0);
        });

        get('option-feature-count')?.addEventListener('input', e => {
            this.setFeature(optionId, 'feature_count', parseInt(e.target.value) || 0);
        });

        get('option-brand')?.addEventListener('input', e => {
            this.setFeature(optionId, 'brand_score', parseFloat(e.target.value) || 5.0);
        });
    }

    setFeature(optionId, key, value) {
        const opt = this.options.find(o => o.id === optionId);
        if (opt) opt.features[key] = value;
    }

    nextStep() {
        if (this.currentStep < this.totalSteps && this.validateStep(this.currentStep)) {
            this.currentStep++;
            this.updateStepDisplay();
        }
    }

    prevStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateStepDisplay();
        }
    }

    validateStep(step) {
        if (step === 1) {
            const title = document.getElementById('comparisonTitle');
            if (!title?.value.trim()) {
                alert('Please enter a comparison title');
                return false;
            }
        } else if (step === 2) {
            if (this.options.length < 2) {
                alert('Please add at least 2 options');
                return false;
            }
            for (const opt of this.options) {
                if (!opt.name?.trim()) {
                    alert('Please fill in all option names');
                    return false;
                }
                if (!opt.features.price || opt.features.price <= 0) {
                    alert(`Please enter a price for "${opt.name || 'option'}"`);
                    return false;
                }
                if (!opt.features.quality_score) {
                    alert(`Please enter a quality score for "${opt.name || 'option'}"`);
                    return false;
                }
                if (opt.features.feature_count === undefined || opt.features.feature_count === null || opt.features.feature_count === '') {
                    alert(`Please enter number of features for "${opt.name || 'option'}"`);
                    return false;
                }
            }
        }
        return true;
    }

    updateStepDisplay() {
        for (let i = 1; i <= this.totalSteps; i++) {
            const el = document.getElementById(`step${i}`);
            if (el) el.style.display = i === this.currentStep ? 'block' : 'none';
        }

        const progress = (this.currentStep / this.totalSteps) * 100;
        const bar = document.querySelector('.step-progress-fill');
        if (bar) bar.style.width = progress + '%';

        document.querySelectorAll('.step-indicator').forEach((ind, idx) => {
            ind.classList.toggle('active', idx < this.currentStep);
            ind.classList.toggle('completed', idx < this.currentStep - 1);
        });
    }

    setupFormSubmission() {
        const btn = document.getElementById('submitComparison');
        if (btn) btn.addEventListener('click', () => this.submitComparison());
    }

    async submitComparison() {
        const submitBtn = document.getElementById('submitComparison');
        const originalText = submitBtn.innerHTML;

        try {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Getting ML Prediction...';
            submitBtn.disabled = true;

            // Get user ID from localStorage (set during login/register)
            const currentUser = getCurrentUser();
            const userId = currentUser?.id || 'guest_' + Date.now();
            const title = document.getElementById('comparisonTitle')?.value || 'Comparison';

            // Build options with all required fields
            const options = this.options.map(opt => ({
                id: opt.id,
                name: opt.name,
                features: {
                    price: opt.features.price || 1,
                    quality_score: opt.features.quality_score || 5,
                    feature_count: opt.features.feature_count || 0,
                    brand_score: opt.features.brand_score || 5.0,
                    availability: 1.0
                }
            }));

            const requestData = {
                user_id: userId,
                options: options
            };

            console.log('📤 Sending to ML API:', requestData);

            const token = localStorage.getItem('auth_token');
            const response = await fetch(`${API_CONFIG.BASE_URL}/ml/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token && { 'Authorization': `Bearer ${token}` })
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || `API error: ${response.status}`);
            }

            const result = await response.json();
            console.log('📊 ML Prediction received:', result);

            // Store result for results page
            sessionStorage.setItem('visionx_last_prediction', JSON.stringify(result));
            sessionStorage.setItem('visionx_comparison_title', title);
            sessionStorage.setItem('visionx_comparison_options', JSON.stringify(this.options));

            showNotification('✅ AI Prediction complete! Redirecting...', 'success');

            setTimeout(() => {
                window.location.href = 'results.html';
            }, 1000);

        } catch (error) {
            console.error('❌ Prediction error:', error);
            showNotification('Prediction failed: ' + error.message, 'error');
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ComparisonForm();
});