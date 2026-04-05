/**
 * VisionX — API Configuration
 * Handles all backend communication with real JWT authentication.
 * Base URL is read from window.VISIONX_API_URL (set by env) or falls back to Render.
 */

const API_CONFIG = {
    BASE_URL: (() => {
        if (window.VISIONX_API_URL) return window.VISIONX_API_URL + '/api/v1';
        // Local dev only
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8000/api/v1';
        }
        // All deployed environments (Vercel, etc.) use Render backend
        return 'https://visionx-mzqc.onrender.com/api/v1';
    })(),
    TIMEOUT: 10000
};

/**
 * Core API call — attaches JWT, handles errors.
 * Shows a visible banner if the backend is unreachable (no silent fake data).
 */
async function apiCall(endpoint, options = {}) {
    const token = localStorage.getItem('auth_token');

    const config = {
        method: options.method || 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` }),
            ...options.headers
        },
        ...(options.body && {
            body: typeof options.body === 'string' ? options.body : JSON.stringify(options.body)
        })
    };

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT);
        const response = await fetch(`${API_CONFIG.BASE_URL}${endpoint}`, { ...config, signal: controller.signal });
        clearTimeout(timeoutId);

        if (response.status === 204) return { success: true };

        const data = await response.json();

        if (response.status === 401) {
            throw new Error('Authentication required');
        }

        if (!response.ok) {
            throw new Error(data.detail || data.message || `API error: ${response.status}`);
        }

        return data;

    } catch (error) {
        if (error.name === 'AbortError' || error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showBackendOfflineBanner();
            throw new Error('Backend is not reachable. Start the server on port 8000.');
        }
        throw error;
    }
}

function showBackendOfflineBanner() {
    if (document.getElementById('vx-offline-banner')) return;
    const banner = document.createElement('div');
    banner.id = 'vx-offline-banner';
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#A32D2D;color:#fff;padding:10px 20px;font-size:13px;text-align:center;font-family:sans-serif;';
    banner.innerHTML = '⚠ <strong>Backend offline</strong> — start the server: <code style="background:rgba(255,255,255,0.2);padding:2px 6px;border-radius:3px;">cd backend && uvicorn app.main:app --reload</code>';
    document.body.prepend(banner);
}

// ─── User session helpers ────────────────────────────────────────────────────

function getCurrentUser() {
    try { const s = localStorage.getItem('currentUser'); return s ? JSON.parse(s) : null; }
    catch { return null; }
}
function setCurrentUser(user) { localStorage.setItem('currentUser', JSON.stringify(user)); }
function clearCurrentUser() {
    localStorage.removeItem('currentUser');
    localStorage.removeItem('auth_token');
    localStorage.removeItem('isAuthenticated');
}
function isAuthenticated() { return !!localStorage.getItem('auth_token') && getCurrentUser() !== null; }

// ─── UI helpers ──────────────────────────────────────────────────────────────

function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = '<div style="text-align:center;padding:2rem"><div style="border:3px solid rgba(79,140,255,0.2);border-top:3px solid #4F8CFF;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto"></div><p style="margin-top:1rem;color:#9AA3C7">Loading...</p></div>';
}

function showError(elementId, message) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = `<div style="text-align:center;padding:2rem"><i class="fas fa-exclamation-circle" style="font-size:3rem;color:#ff4f4f;margin-bottom:1rem"></i><h3 style="color:#E6E8F2">Error</h3><p style="color:#9AA3C7">${message}</p></div>`;
}

function showEmptyState(elementId, icon, title, message, actionText, actionHref) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = `<div style="text-align:center;padding:3rem"><i class="${icon}" style="font-size:4rem;color:#6B7298;margin-bottom:1rem"></i><h3 style="color:#E6E8F2;margin-bottom:0.5rem">${title}</h3><p style="color:#9AA3C7;margin-bottom:1.5rem">${message}</p>${actionText ? `<a href="${actionHref}" class="btn btn-primary">${actionText}</a>` : ''}</div>`;
}

function showNotification(message, type = 'info') {
    const existing = document.querySelector('.vx-notification');
    if (existing) existing.remove();
    const colors = { success: '#10b981', error: '#ef4444', info: '#4F8CFF', warning: '#f59e0b' };
    const n = document.createElement('div');
    n.className = 'vx-notification';
    n.style.cssText = `position:fixed;top:20px;right:20px;z-index:10000;background:${colors[type]||colors.info};color:#fff;padding:12px 20px;border-radius:8px;font-size:14px;box-shadow:0 4px 12px rgba(0,0,0,0.3);max-width:360px;`;
    n.textContent = message;
    document.body.appendChild(n);
    setTimeout(() => n.remove(), 4000);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const diffDays = Math.floor((new Date() - date) / 86400000);
    if (diffDays === 0) return 'Today at ' + date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    if (diffDays === 1) return 'Yesterday at ' + date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    if (diffDays < 7) return diffDays + ' days ago';
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function debounce(func, wait) {
    let t; return (...args) => { clearTimeout(t); t = setTimeout(() => func(...args), wait); };
}

if (!document.getElementById('vx-anim-styles')) {
    const s = document.createElement('style');
    s.id = 'vx-anim-styles';
    s.textContent = '@keyframes spin{to{transform:rotate(360deg)}}@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}';
    document.head.appendChild(s);
}

console.log('✅ API config loaded — base:', API_CONFIG.BASE_URL);