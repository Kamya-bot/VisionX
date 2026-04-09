/**
 * VisionX — API Configuration
 * Phase 3: adds refresh token rotation, per-call auth header injection,
 * and the apiCall wrapper used by all pages.
 */

const API_BASE = (() => {
    if (window.VISIONX_API_URL) return window.VISIONX_API_URL + '/api/v1';
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:8000/api/v1';
    }
    return 'https://visionx-mzqc.onrender.com/api/v1';
})();

// ── Session helpers ───────────────────────────────────────────────────────────

function getToken()       { return localStorage.getItem('auth_token'); }
function getRefreshToken(){ return localStorage.getItem('refresh_token'); }

function getCurrentUser() {
    try { return JSON.parse(localStorage.getItem('currentUser')); } catch { return null; }
}
function setCurrentUser(u) { localStorage.setItem('currentUser', JSON.stringify(u)); }
function clearCurrentUser() {
    ['currentUser','auth_token','refresh_token','isAuthenticated'].forEach(k => localStorage.removeItem(k));
}
function isAuthenticated() { return !!getToken() && !!getCurrentUser(); }

// ── Token refresh ─────────────────────────────────────────────────────────────

let _refreshing = false;
let _refreshQueue = [];

async function _doRefresh() {
    const rt = getRefreshToken();
    if (!rt) throw new Error('No refresh token');
    const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: rt }),
    });
    if (!res.ok) throw new Error('Refresh failed');
    const data = await res.json();
    localStorage.setItem('auth_token', data.access_token);
    if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
    return data.access_token;
}

async function getValidToken() {
    if (_refreshing) {
        return new Promise((resolve, reject) => _refreshQueue.push({ resolve, reject }));
    }
    return getToken();
}

// ── Core API call ─────────────────────────────────────────────────────────────

async function apiCall(endpoint, options = {}) {
    const token = await getValidToken();

    const config = {
        method: options.method || 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` }),
            ...options.headers,
        },
        ...(options.body && {
            body: typeof options.body === 'string' ? options.body : JSON.stringify(options.body),
        }),
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 55000);

    try {
        const res = await fetch(`${API_BASE}${endpoint}`, { ...config, signal: controller.signal });
        clearTimeout(timeoutId);

        if (res.status === 204) return { success: true };

        const data = await res.json();

        if (res.status === 401) {
            // Try refresh once
            try {
                _refreshing = true;
                const newToken = await _doRefresh();
                _refreshQueue.forEach(p => p.resolve(newToken));
                _refreshQueue = [];
                _refreshing = false;
                // Retry original call with new token
                return apiCall(endpoint, options);
            } catch {
                _refreshQueue.forEach(p => p.reject(new Error('Session expired')));
                _refreshQueue = [];
                _refreshing = false;
                clearCurrentUser();
                window.location.replace('login.html');
                throw new Error('Session expired');
            }
        }

        if (!res.ok) throw new Error(data.detail || data.message || `API error ${res.status}`);
        return data;

    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError' || err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            _showOfflineBanner();
            throw new Error('Backend unreachable');
        }
        throw err;
    }
}

function _showOfflineBanner() {
    if (document.getElementById('vx-offline-banner')) return;
    const b = document.createElement('div');
    b.id = 'vx-offline-banner';
    b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#A32D2D;color:#fff;padding:10px 20px;font-size:13px;text-align:center;';
    b.innerHTML = '⚠ <strong>Backend offline</strong> — start server: <code style="background:rgba(255,255,255,0.2);padding:2px 6px;border-radius:3px;">cd backend && uvicorn app.main:app --reload</code>';
    document.body.prepend(b);
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function showNotification(message, type = 'info') {
    document.querySelector('.vx-notification')?.remove();
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
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return diffDays + ' days ago';
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = '<div style="text-align:center;padding:2rem"><div style="border:3px solid rgba(79,140,255,0.2);border-top:3px solid #4F8CFF;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto"></div><p style="margin-top:1rem;color:#9AA3C7">Loading...</p></div>';
}

if (!document.getElementById('vx-anim-styles')) {
    const s = document.createElement('style');
    s.id = 'vx-anim-styles';
    s.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
    document.head.appendChild(s);
}

function logout() { clearCurrentUser(); window.location.replace('login.html'); }

console.log('✅ API config loaded — base:', API_BASE);