/**
 * Results Page with Real Data
 * Displays comparison results with ML insights
 */

let currentComparison = null;

/**
 * Load Comparison Results
 */
async function loadResults() {
    const user = getCurrentUser();
    
    if (!user) {
        window.location.href = 'login.html';
        return;
    }
    
    // Get comparison ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const comparisonId = urlParams.get('id');
    
    if (!comparisonId) {
        showError('resultsContainer', 'No comparison specified');
        return;
    }
    
    console.log('Loading results for comparison:', comparisonId);
    
    try {
        // Load comparison from localStorage
        const allComparisons = JSON.parse(localStorage.getItem('comparisons') || '[]');
        currentComparison = allComparisons.find(c => c.id === comparisonId);
        
        if (!currentComparison) {
            showError('resultsContainer', 'Comparison not found');
            return;
        }
        
        // Check if user owns this comparison
        if (currentComparison.user_id !== user.id) {
            showError('resultsContainer', 'Access denied');
            return;
        }
        
        // Display results
        displayResults(currentComparison);
        
    } catch (error) {
        console.error('Failed to load results:', error);
        showError('resultsContainer', 'Failed to load results');
    }
}

/**
 * Display Results
 */
function displayResults(comparison) {
    // Update title
    const titleEl = document.getElementById('comparisonTitle');
    if (titleEl) {
        titleEl.textContent = comparison.title || 'Comparison Results';
    }
    
    // Update metadata
    const metaEl = document.getElementById('comparisonMeta');
    if (metaEl) {
        metaEl.innerHTML = `
            <span><i class="fas fa-clock"></i> ${formatDate(comparison.created_at)}</span>
            <span><i class="fas fa-list"></i> ${comparison.options?.length || 0} options</span>
            ${comparison.category ? `<span><i class="fas fa-tag"></i> ${comparison.category}</span>` : ''}
        `;
    }
    
    if (!comparison.result) {
        showError('resultsContainer', 'No results available');
        return;
    }
    
    // Display winner
    displayWinner(comparison);
    
    // Display all options comparison
    displayOptionsComparison(comparison);
    
    // Display feature importance
    displayFeatureImportance(comparison.result);
    
    // Display alternatives
    displayAlternatives(comparison.result);
}

/**
 * Display Winner
 */
function displayWinner(comparison) {
    const container = document.getElementById('winnerCard');
    if (!container) return;
    
    const result = comparison.result;
    const winningOption = comparison.options.find(opt => opt.id === result.recommended_option_id);
    
    container.innerHTML = `
        <div class="glass-card" style="padding: 2rem; text-align: center; background: linear-gradient(135deg, rgba(79, 140, 255, 0.1), rgba(123, 97, 255, 0.1));">
            <div style="font-size: 3rem; margin-bottom: 1rem;">
                <i class="fas fa-trophy" style="color: #FFD700;"></i>
            </div>
            <h2 style="color: #E6E8F2; margin-bottom: 0.5rem;">Recommended Choice</h2>
            <h3 style="color: #4F8CFF; font-size: 2rem; margin-bottom: 1rem;">${result.recommended_option_name}</h3>
            
            <div style="display: inline-block; background: rgba(79, 140, 255, 0.2); padding: 0.75rem 2rem; border-radius: 20px; margin-bottom: 1.5rem;">
                <span style="color: #4F8CFF; font-size: 1.5rem; font-weight: 600;">${Math.round(result.confidence * 100)}%</span>
                <span style="color: #9AA3C7; margin-left: 0.5rem;">confidence</span>
            </div>
            
            ${result.user_cluster ? `
                <div style="margin-bottom: 1.5rem;">
                    <span style="background: rgba(123, 97, 255, 0.2); color: #7B61FF; padding: 0.5rem 1rem; border-radius: 12px; font-size: 0.875rem;">
                        <i class="fas fa-user-tag"></i> ${result.user_cluster}
                    </span>
                </div>
            ` : ''}
            
            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px; text-align: left;">
                <h4 style="color: #E6E8F2; margin-bottom: 1rem;">
                    <i class="fas fa-lightbulb" style="color: #4F8CFF;"></i> Why This Choice?
                </h4>
                <p style="color: #9AA3C7; line-height: 1.6;">${result.reasoning}</p>
            </div>
            
            ${winningOption ? `
                <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; text-align: center;">
                        ${winningOption.features.price ? `
                            <div>
                                <div style="color: #9AA3C7; font-size: 0.875rem; margin-bottom: 0.25rem;">Price</div>
                                <div style="color: #E6E8F2; font-size: 1.25rem; font-weight: 600;">$${winningOption.features.price.toLocaleString()}</div>
                            </div>
                        ` : ''}
                        ${winningOption.features.quality_score ? `
                            <div>
                                <div style="color: #9AA3C7; font-size: 0.875rem; margin-bottom: 0.25rem;">Quality</div>
                                <div style="color: #E6E8F2; font-size: 1.25rem; font-weight: 600;">${winningOption.features.quality_score}/10</div>
                            </div>
                        ` : ''}
                        ${winningOption.features.delivery_time ? `
                            <div>
                                <div style="color: #9AA3C7; font-size: 0.875rem; margin-bottom: 0.25rem;">Time</div>
                                <div style="color: #E6E8F2; font-size: 1.25rem; font-weight: 600;">${winningOption.features.delivery_time} days</div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Display Options Comparison
 */
function displayOptionsComparison(comparison) {
    const container = document.getElementById('optionsComparison');
    if (!container) return;
    
    container.innerHTML = `
        <h3 style="color: #E6E8F2; margin-bottom: 1.5rem;">All Options Comparison</h3>
        <div style="display: grid; gap: 1rem;">
            ${comparison.options.map((option, index) => {
                const isWinner = option.id === comparison.result.recommended_option_id;
                return `
                    <div class="glass-card" style="padding: 1.5rem; ${isWinner ? 'border: 2px solid #4F8CFF;' : ''}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h4 style="color: #E6E8F2;">${option.name}</h4>
                            ${isWinner ? '<span style="background: #4F8CFF; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem;">WINNER</span>' : ''}
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem;">
                            ${option.features.price ? `
                                <div>
                                    <div style="color: #9AA3C7; font-size: 0.875rem;">Price</div>
                                    <div style="color: #E6E8F2; font-weight: 500;">$${option.features.price.toLocaleString()}</div>
                                </div>
                            ` : ''}
                            ${option.features.quality_score ? `
                                <div>
                                    <div style="color: #9AA3C7; font-size: 0.875rem;">Quality</div>
                                    <div style="color: #E6E8F2; font-weight: 500;">${option.features.quality_score}/10</div>
                                </div>
                            ` : ''}
                            ${option.features.delivery_time ? `
                                <div>
                                    <div style="color: #9AA3C7; font-size: 0.875rem;">Time</div>
                                    <div style="color: #E6E8F2; font-weight: 500;">${option.features.delivery_time} days</div>
                                </div>
                            ` : ''}
                        </div>
                        ${option.notes ? `
                            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                                <div style="color: #9AA3C7; font-size: 0.875rem;">${option.notes}</div>
                            </div>
                        ` : ''}
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

/**
 * Display Feature Importance
 */
function displayFeatureImportance(result) {
    const container = document.getElementById('featureImportance');
    if (!container || !result.feature_importance) return;
    
    const sorted = [...result.feature_importance].sort((a, b) => b.importance - a.importance);
    
    container.innerHTML = `
        <h3 style="color: #E6E8F2; margin-bottom: 1.5rem;">Key Decision Factors</h3>
        <div class="glass-card" style="padding: 1.5rem;">
            ${sorted.map(feature => `
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #E6E8F2; text-transform: capitalize;">${feature.feature_name.replace('_', ' ')}</span>
                        <span style="color: #4F8CFF; font-weight: 600;">${Math.round(feature.importance * 100)}%</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #4F8CFF, #7B61FF); height: 100%; width: ${feature.importance * 100}%; transition: width 0.5s ease;"></div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Display Alternatives
 */
function displayAlternatives(result) {
    const container = document.getElementById('alternatives');
    if (!container || !result.alternative_options || result.alternative_options.length === 0) return;
    
    container.innerHTML = `
        <h3 style="color: #E6E8F2; margin-bottom: 1.5rem;">Alternative Options</h3>
        <div style="display: grid; gap: 1rem;">
            ${result.alternative_options.map((alt, index) => `
                <div class="glass-card" style="padding: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="color: #E6E8F2;">${alt.name}</h4>
                        <span style="color: #9AA3C7; font-size: 0.875rem;">${Math.round((alt.score || 0) * 100)}% match</span>
                    </div>
                    <p style="color: #9AA3C7; font-size: 0.875rem;">${alt.reason || 'Good alternative option'}</p>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Initialize Results Page
 */
document.addEventListener('DOMContentLoaded', () => {
    // Check auth first
    if (!checkAuth()) return;
    
    // Load results
    loadResults();
    
    // Bind back button
    const backBtn = document.getElementById('backBtn');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            window.location.href = 'history.html';
        });
    }
    
    // Bind new comparison button
    const newBtn = document.getElementById('newComparisonBtn');
    if (newBtn) {
        newBtn.addEventListener('click', () => {
            window.location.href = 'comparison.html';
        });
    }
});

console.log('✅ Results Real Data loaded');
