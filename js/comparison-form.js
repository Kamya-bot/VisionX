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

        // Add initial options
        this.addOption();
        this.addOption();
    }

    addOption() {
        const optionId = 'option_' + Date.now();
        const optionIndex = this.options.length;

        const optionData = {
            id: optionId,
            name: '',
            features: {}
        };

        this.options.push(optionData);

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
                           placeholder="e.g., Job at Company A" 
                           data-option-id="${optionId}" required>
                </div>

                <div class="form-group">
                    <label class="form-label">Price / Cost</label>
                    <input type="number" class="form-input option-price" 
                           placeholder="e.g., 75000" 
                           data-option-id="${optionId}">
                </div>

                <div class="form-group">
                    <label class="form-label">Quality Score (1-10)</label>
                    <input type="number" class="form-input option-quality" 
                           placeholder="Rate from 1-10" 
                           min="1" max="10"
                           data-option-id="${optionId}">
                </div>

                <div class="form-group">
                    <label class="form-label">Satisfaction Score (1-10)</label>
                    <input type="number" class="form-input option-satisfaction" 
                           placeholder="Expected satisfaction 1-10" 
                           min="1" max="10"
                           data-option-id="${optionId}">
                </div>
            </div>
        `;

        const optionsContainer = document.getElementById('optionsContainer');
        if (optionsContainer) {
            optionsContainer.insertAdjacentHTML('beforeend', optionHTML);
            this.setupRemoveButton(optionId);
            this.setupOptionInputs(optionId);
        }
    }

    setupRemoveButton(optionId) {
        const removeBtn = document.querySelector(`.remove-option-btn[data-option-id="${optionId}"]`);
        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                if (this.options.length > 2) {
                    this.removeOption(optionId);
                } else {
                    alert('You must have at least 2 options to compare');
                }
            });
        }
    }

    setupOptionInputs(optionId) {
        const nameInput = document.querySelector(`.option-name[data-option-id="${optionId}"]`);
        const priceInput = document.querySelector(`.option-price[data-option-id="${optionId}"]`);
        const qualityInput = document.querySelector(`.option-quality[data-option-id="${optionId}"]`);
        const satisfactionInput = document.querySelector(`.option-satisfaction[data-option-id="${optionId}"]`);

        if (nameInput) {
            nameInput.addEventListener('input', (e) => {
                this.updateOptionData(optionId, 'name', e.target.value);
            });
        }

        if (priceInput) {
            priceInput.addEventListener('input', (e) => {
                this.updateOptionFeature(optionId, 'price', parseFloat(e.target.value) || 0);
            });
        }

        if (qualityInput) {
            qualityInput.addEventListener('input', (e) => {
                this.updateOptionFeature(optionId, 'quality_score', parseInt(e.target.value) || 0);
            });
        }

        if (satisfactionInput) {
            satisfactionInput.addEventListener('input', (e) => {
                this.updateOptionFeature(optionId, 'satisfaction_score', parseInt(e.target.value) || 0);
            });
        }
    }

    updateOptionData(optionId, field, value) {
        const option = this.options.find(opt => opt.id === optionId);
        if (option) {
            option[field] = value;
        }
    }

    updateOptionFeature(optionId, feature, value) {
        const option = this.options.find(opt => opt.id === optionId);
        if (option) {
            option.features[feature] = value;
        }
    }

    removeOption(optionId) {
        this.options = this.options.filter(opt => opt.id !== optionId);
        const optionCard = document.querySelector(`.option-card[data-option-id="${optionId}"]`);
        if (optionCard) {
            optionCard.remove();
        }
    }

    nextStep() {
        if (this.currentStep < this.totalSteps) {
            if (this.validateStep(this.currentStep)) {
                this.currentStep++;
                this.updateStepDisplay();
            }
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
            // Validate basic info
            const title = document.getElementById('comparisonTitle');
            if (!title || !title.value.trim()) {
                alert('Please enter a comparison title');
                return false;
            }
        } else if (step === 2) {
            // Validate options
            if (this.options.length < 2) {
                alert('Please add at least 2 options');
                return false;
            }

            for (const option of this.options) {
                if (!option.name || !option.name.trim()) {
                    alert('Please fill in all option names');
                    return false;
                }
            }
        }
        return true;
    }

    updateStepDisplay() {
        // Hide all steps
        for (let i = 1; i <= this.totalSteps; i++) {
            const step = document.getElementById(`step${i}`);
            if (step) {
                step.style.display = 'none';
            }
        }

        // Show current step
        const currentStepEl = document.getElementById(`step${this.currentStep}`);
        if (currentStepEl) {
            currentStepEl.style.display = 'block';
        }

        // Update progress bar
        const progress = (this.currentStep / this.totalSteps) * 100;
        const progressBar = document.querySelector('.step-progress-fill');
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }

        // Update step indicators
        document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
            if (index < this.currentStep) {
                indicator.classList.add('active');
                indicator.classList.add('completed');
            } else if (index === this.currentStep) {
                indicator.classList.add('active');
                indicator.classList.remove('completed');
            } else {
                indicator.classList.remove('active');
                indicator.classList.remove('completed');
            }
        });
    }

    setupFormSubmission() {
        const submitBtn = document.getElementById('submitComparison');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitComparison());
        }
    }

    async submitComparison() {
        if (!this.validateStep(3)) return;

        const submitBtn = document.getElementById('submitComparison');
        const originalText = submitBtn.innerHTML;
        
        try {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Getting ML Prediction...';
            submitBtn.disabled = true;

            // Get user ID from session
            const userId = sessionStorage.getItem('visionx_user_id') || 'demo_user_' + Date.now();
            const title = document.getElementById('comparisonTitle')?.value || 'Comparison';

            // Prepare request for backend ML API
            const requestData = {
                user_id: userId,
                decision_type: document.getElementById('comparisonCategory')?.value || 'general',
                options: this.options
            };

            console.log('📤 Sending to ML API:', requestData);

            // Call the real backend ML API
            const response = await fetch(`${API_CONFIG.BASE_URL}/ml/predict`, {
                method: 'POST',
                headers: API_CONFIG.HEADERS,
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const result = await response.json();
            console.log('📥 ML Prediction received:', result);

            // Store result in session
            sessionStorage.setItem('visionx_last_prediction', JSON.stringify(result));
            sessionStorage.setItem('visionx_comparison_title', title);

            // Show success message
            alert('✅ ML Prediction Complete!\\n\\nRecommendation: ' + result.recommendation?.name + '\\nConfidence: ' + (result.confidence * 100).toFixed(1) + '%');

            // Redirect to results
            setTimeout(() => {
                window.location.href = 'results.html';
            }, 1000);

        } catch (error) {
            console.error('❌ Prediction error:', error);
            alert('Failed to get ML prediction: ' + error.message + '\\n\\nPlease ensure the backend is running on port 8000.');
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ComparisonForm();
});
