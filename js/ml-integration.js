/**
 * ============================================
 * VisionX ML Integration - Core API Layer
 * ============================================
 * 
 * Purpose: Provides centralized API communication with ML backend
 * Backend: FastAPI ML service with 3 models (K-Means, RandomForest, Recommender)
 * 
 * Features:
 * - User clustering and segmentation
 * - Decision pattern analysis
 * - AI-powered recommendations
 * - Predictive scoring
 * - Model performance metrics
 * 
 * @version 1.0.0
 * @author VisionX Team
 */

// ============================================
// Configuration
// ============================================

const ML_CONFIG = {
    // Backend API base URL - UPDATE THIS to your deployed backend URL
    baseURL: 'http://localhost:8000',
    
    // API endpoints
    endpoints: {
        // User & Clustering
        userCluster: '/api/v1/ml/user-cluster',
        clusterStats: '/api/v1/ml/cluster-stats',
        
        // Predictions & Recommendations
        predict: '/api/v1/ml/predict',
        recommend: '/api/v1/ml/recommend',
        
        // Analytics
        patterns: '/api/v1/ml/decision-patterns',
        insights: '/api/v1/ml/insights',
        performance: '/api/v1/ml/model-performance',
        
        // Training & Optimization
        train: '/api/v1/ml/train',
        optimize: '/api/v1/ml/optimize',
        retrain: '/api/v1/ml/retrain'
    },
    
    // Request timeout (ms)
    timeout: 10000,
    
    // Retry configuration
    retry: {
        attempts: 3,
        delay: 1000
    }
};

// ============================================
// Utility Functions
// ============================================

/**
 * Creates a timeout promise
 * @param {number} ms - Timeout in milliseconds
 * @returns {Promise}
 */
function timeout(ms) {
    return new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), ms)
    );
}

/**
 * Delays execution
 * @param {number} ms - Delay in milliseconds
 * @returns {Promise}
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Makes HTTP request with timeout and retry
 * @param {string} url - Request URL
 * @param {object} options - Fetch options
 * @param {number} retries - Number of retry attempts
 * @returns {Promise<object>}
 */
async function fetchWithRetry(url, options = {}, retries = ML_CONFIG.retry.attempts) {
    try {
        const response = await Promise.race([
            fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            }),
            timeout(ML_CONFIG.timeout)
        ]);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        return data;
        
    } catch (error) {
        if (retries > 0) {
            console.warn(`Request failed, retrying... (${retries} attempts left)`);
            await delay(ML_CONFIG.retry.delay);
            return fetchWithRetry(url, options, retries - 1);
        }
        throw error;
    }
}

/**
 * Constructs full API URL
 * @param {string} endpoint - Endpoint path
 * @returns {string}
 */
function getAPIUrl(endpoint) {
    return `${ML_CONFIG.baseURL}${endpoint}`;
}

// ============================================
// API Client Class
// ============================================

class MLAPIClient {
    constructor() {
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }
    
    /**
     * Gets cached data if available and not expired
     * @param {string} key - Cache key
     * @returns {any|null}
     */
    getCache(key) {
        const cached = this.cache.get(key);
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }
        this.cache.delete(key);
        return null;
    }
    
    /**
     * Sets cache data
     * @param {string} key - Cache key
     * @param {any} data - Data to cache
     */
    setCache(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }
    
    /**
     * Clears cache
     */
    clearCache() {
        this.cache.clear();
    }
    
    // ============================================
    // User Clustering APIs
    // ============================================
    
    /**
     * Gets user cluster information
     * @param {string} userId - User ID
     * @returns {Promise<object>} Cluster data
     */
    async getUserCluster(userId) {
        const cacheKey = `cluster_${userId}`;
        const cached = this.getCache(cacheKey);
        if (cached) return cached;
        
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.userCluster);
            const data = await fetchWithRetry(`${url}/${userId}`);
            
            this.setCache(cacheKey, data);
            return data;
            
        } catch (error) {
            console.error('Error fetching user cluster:', error);
            throw new Error(`Failed to fetch user cluster: ${error.message}`);
        }
    }
    
    /**
     * Gets cluster statistics
     * @returns {Promise<object>} Cluster stats
     */
    async getClusterStats() {
        const cacheKey = 'cluster_stats';
        const cached = this.getCache(cacheKey);
        if (cached) return cached;
        
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.clusterStats);
            const data = await fetchWithRetry(url);
            
            this.setCache(cacheKey, data);
            return data;
            
        } catch (error) {
            console.error('Error fetching cluster stats:', error);
            throw new Error(`Failed to fetch cluster stats: ${error.message}`);
        }
    }
    
    // ============================================
    // Prediction APIs
    // ============================================
    
    /**
     * Gets AI prediction for comparison
     * @param {object} comparisonData - Comparison data
     * @returns {Promise<object>} Prediction result
     */
    async getPrediction(comparisonData) {
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.predict);
            const data = await fetchWithRetry(url, {
                method: 'POST',
                body: JSON.stringify(comparisonData)
            });
            
            return data;
            
        } catch (error) {
            console.error('Error fetching prediction:', error);
            throw new Error(`Failed to get prediction: ${error.message}`);
        }
    }
    
    /**
     * Gets AI recommendation
     * @param {object} comparisonData - Comparison data
     * @returns {Promise<object>} Recommendation result
     */
    async getRecommendation(comparisonData) {
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.recommend);
            const data = await fetchWithRetry(url, {
                method: 'POST',
                body: JSON.stringify(comparisonData)
            });
            
            return data;
            
        } catch (error) {
            console.error('Error fetching recommendation:', error);
            throw new Error(`Failed to get recommendation: ${error.message}`);
        }
    }
    
    // ============================================
    // Analytics APIs
    // ============================================
    
    /**
     * Gets decision patterns for user
     * @param {string} userId - User ID
     * @returns {Promise<object>} Decision patterns
     */
    async getDecisionPatterns(userId) {
        const cacheKey = `patterns_${userId}`;
        const cached = this.getCache(cacheKey);
        if (cached) return cached;
        
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.patterns);
            const data = await fetchWithRetry(`${url}/${userId}`);
            
            this.setCache(cacheKey, data);
            return data;
            
        } catch (error) {
            console.error('Error fetching decision patterns:', error);
            throw new Error(`Failed to fetch patterns: ${error.message}`);
        }
    }
    
    /**
     * Gets AI insights
     * @param {string} userId - User ID
     * @returns {Promise<object>} AI insights
     */
    async getInsights(userId) {
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.insights);
            const data = await fetchWithRetry(`${url}/${userId}`);
            
            return data;
            
        } catch (error) {
            console.error('Error fetching insights:', error);
            throw new Error(`Failed to fetch insights: ${error.message}`);
        }
    }
    
    /**
     * Gets model performance metrics
     * @returns {Promise<object>} Performance metrics
     */
    async getModelPerformance() {
        const cacheKey = 'model_performance';
        const cached = this.getCache(cacheKey);
        if (cached) return cached;
        
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.performance);
            const data = await fetchWithRetry(url);
            
            this.setCache(cacheKey, data);
            return data;
            
        } catch (error) {
            console.error('Error fetching model performance:', error);
            throw new Error(`Failed to fetch performance: ${error.message}`);
        }
    }
    
    // ============================================
    // Training APIs (Admin)
    // ============================================
    
    /**
     * Trains all ML models
     * @returns {Promise<object>} Training result
     */
    async trainModels() {
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.train);
            const data = await fetchWithRetry(url, {
                method: 'POST'
            });
            
            this.clearCache();
            return data;
            
        } catch (error) {
            console.error('Error training models:', error);
            throw new Error(`Failed to train models: ${error.message}`);
        }
    }
    
    /**
     * Optimizes model hyperparameters
     * @returns {Promise<object>} Optimization result
     */
    async optimizeModels() {
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.optimize);
            const data = await fetchWithRetry(url, {
                method: 'POST'
            });
            
            this.clearCache();
            return data;
            
        } catch (error) {
            console.error('Error optimizing models:', error);
            throw new Error(`Failed to optimize models: ${error.message}`);
        }
    }
    
    /**
     * Retrains models with new data
     * @param {object} newData - New training data
     * @returns {Promise<object>} Retrain result
     */
    async retrainModels(newData) {
        try {
            const url = getAPIUrl(ML_CONFIG.endpoints.retrain);
            const data = await fetchWithRetry(url, {
                method: 'POST',
                body: JSON.stringify(newData)
            });
            
            this.clearCache();
            return data;
            
        } catch (error) {
            console.error('Error retraining models:', error);
            throw new Error(`Failed to retrain models: ${error.message}`);
        }
    }
}

// ============================================
// Global Instance
// ============================================

const mlAPI = new MLAPIClient();

// ============================================
// Convenience Functions
// ============================================

/**
 * Formats cluster label for display
 * @param {number} clusterId - Cluster ID (0-2)
 * @returns {object} Formatted cluster info
 */
function formatClusterLabel(clusterId) {
    const labels = {
        0: {
            name: 'Strategic Thinker',
            color: '#4F8CFF',
            icon: '🎯',
            description: 'Data-driven decision maker who values analytics'
        },
        1: {
            name: 'Balanced Decider',
            color: '#7B61FF',
            icon: '⚖️',
            description: 'Weighs multiple factors carefully'
        },
        2: {
            name: 'Quick Decider',
            color: '#A855F7',
            icon: '⚡',
            description: 'Efficient and action-oriented'
        }
    };
    
    return labels[clusterId] || {
        name: 'Unknown',
        color: '#6B7298',
        icon: '❓',
        description: 'Cluster not determined'
    };
}

/**
 * Formats confidence score as percentage
 * @param {number} confidence - Confidence score (0-1)
 * @returns {string} Formatted percentage
 */
function formatConfidence(confidence) {
    return `${(confidence * 100).toFixed(1)}%`;
}

/**
 * Gets confidence level label
 * @param {number} confidence - Confidence score (0-1)
 * @returns {string} Confidence level
 */
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
}

// ============================================
// Export
// ============================================

// Make available globally
window.mlAPI = mlAPI;
window.ML_CONFIG = ML_CONFIG;
window.formatClusterLabel = formatClusterLabel;
window.formatConfidence = formatConfidence;
window.getConfidenceLevel = getConfidenceLevel;

console.log('✅ ML Integration loaded successfully');
