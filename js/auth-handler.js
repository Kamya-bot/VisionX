/**
 * VisionX — Auth Handler (Phase 3)
 * - Email/password login & register → real backend
 * - Google & GitHub OAuth → real backend redirect (no more "coming soon")
 * - Stores access_token + refresh_token on success
 * - Auth guard redirects unauthenticated users
 */

// ── Auth guard ────────────────────────────────────────────────────────────────
(function () {
    const PUBLIC = ['login.html', 'register.html', 'index.html', ''];
    const page = window.location.pathname.split('/').pop();
    if (!PUBLIC.includes(page) && !isAuthenticated()) {
        window.location.replace('login.html');
    }
})();

// ── Wake server on page load ──────────────────────────────────────────────────
fetch('https://visionx-mzqc.onrender.com/health').catch(() => {});

// ── Cold-start banner ─────────────────────────────────────────────────────────
function showWakingBanner() {
    if (document.getElementById('vx-waking-banner')) return;
    const b = document.createElement('div');
    b.id = 'vx-waking-banner';
    b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#1a56db;color:#fff;padding:10px 20px;font-size:13px;text-align:center;';
    b.innerHTML = '⏳ <strong>Starting server...</strong> — Free server wakes from sleep. This takes up to 30 seconds on first use.';
    document.body.prepend(b);
}
function hideWakingBanner() { document.getElementById('vx-waking-banner')?.remove(); }

// ── Save session ──────────────────────────────────────────────────────────────
function saveUserSession(data, emailFallback) {
    const user = {
        id:            data.user_id,
        email:         data.email || emailFallback,
        name:          data.full_name || (emailFallback || '').split('@')[0],
        avatar:        (data.full_name || emailFallback || 'U').charAt(0).toUpperCase(),
        cluster_id:    data.cluster_id,
        cluster_label: data.cluster_label,
        created_at:    new Date().toISOString(),
    };
    setCurrentUser(user);
    localStorage.setItem('auth_token', data.access_token);
    if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
    localStorage.setItem('isAuthenticated', 'true');
}

// ── Login ─────────────────────────────────────────────────────────────────────
async function handleLogin(event) {
    if (event) event.preventDefault();
    const email    = document.getElementById('email')?.value?.trim();
    const password = document.getElementById('password')?.value;
    const btn      = document.querySelector('#loginForm button[type="submit"]');
    if (!email || !password) { showNotification('Email and password required', 'error'); return; }

    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging in...'; }
    showWakingBanner();

    try {
        const data = await apiCall('/auth/login', { method: 'POST', body: { email, password } });
        hideWakingBanner();
        saveUserSession(data, email);
        showNotification('Login successful! Redirecting...', 'success');
        window.location.replace('dashboard.html');
    } catch (err) {
        hideWakingBanner();
        showNotification(err.message || 'Login failed', 'error');
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Sign In'; }
    }
}

// ── Register ──────────────────────────────────────────────────────────────────
async function handleRegister(event) {
    if (event) event.preventDefault();
    const name     = document.getElementById('name')?.value?.trim();
    const email    = document.getElementById('email')?.value?.trim();
    const password = document.getElementById('password')?.value;
    const confirm  = document.getElementById('confirmPassword')?.value;
    const btn      = document.querySelector('#registerForm button[type="submit"]');

    if (!email || !password) { showNotification('Email and password required', 'error'); return; }
    if (confirm && password !== confirm) { showNotification('Passwords do not match', 'error'); return; }
    if (password.length < 6) { showNotification('Password must be at least 6 characters', 'error'); return; }

    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...'; }
    showWakingBanner();

    try {
        const data = await apiCall('/auth/register', { method: 'POST', body: { email, password, full_name: name || null } });
        hideWakingBanner();
        saveUserSession(data, email);
        showNotification('Account created! Welcome to VisionX 🎉', 'success');
        window.location.replace('dashboard.html');
    } catch (err) {
        hideWakingBanner();
        showNotification(err.message || 'Registration failed', 'error');
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-user-plus"></i> Create Account'; }
    }
}

// ── OAuth ─────────────────────────────────────────────────────────────────────
// Backend Phase 1 added /auth/google and /auth/github redirect endpoints.
// These redirect to the provider and return back to /auth/callback.
// We just redirect the browser to the backend OAuth entry point.

function handleGoogleLogin() {
    showWakingBanner();
    window.location.href = 'https://visionx-mzqc.onrender.com/api/v1/auth/google';
}

function handleGithubLogin() {
    showWakingBanner();
    window.location.href = 'https://visionx-mzqc.onrender.com/api/v1/auth/github';
}

// ── OAuth callback handler ────────────────────────────────────────────────────
// After OAuth the backend redirects to /oauth-callback.html?token=...&refresh=...
// That page calls this function to save the session.
function handleOAuthCallback() {
    const params = new URLSearchParams(window.location.search);
    const token   = params.get('token');
    const refresh = params.get('refresh');
    const userId  = params.get('user_id');
    const email   = params.get('email');
    const name    = params.get('name');

    if (!token) {
        showNotification('OAuth login failed — no token received', 'error');
        setTimeout(() => window.location.replace('login.html'), 2000);
        return;
    }

    saveUserSession({
        access_token:  token,
        refresh_token: refresh,
        user_id:       userId,
        email:         email,
        full_name:     name,
    }, email || 'user');

    showNotification('Logged in successfully!', 'success');
    window.location.replace('dashboard.html');
}

// ── Logout ────────────────────────────────────────────────────────────────────
function handleLogout() { clearCurrentUser(); window.location.replace('login.html'); }

// ── Update nav ────────────────────────────────────────────────────────────────
function updateNavUser() {
    const user = getCurrentUser();
    if (!user) return;
    document.querySelectorAll('.user-name, .user-name-initial').forEach(el => {
        el.textContent = user.name || user.email;
    });
    document.querySelectorAll('.user-avatar span').forEach(el => {
        el.textContent = user.avatar || 'U';
    });
    document.querySelectorAll('#userInitials').forEach(el => {
        el.textContent = user.avatar || 'U';
    });
}

// ── Wire everything up ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateNavUser();

    document.getElementById('loginForm')?.addEventListener('submit', handleLogin);
    document.getElementById('registerForm')?.addEventListener('submit', handleRegister);

    document.querySelectorAll('.logout-btn, [data-action="logout"]')
        .forEach(b => b.addEventListener('click', handleLogout));
    document.querySelectorAll('.google-btn')
        .forEach(b => b.addEventListener('click', handleGoogleLogin));
    document.querySelectorAll('.github-btn')
        .forEach(b => b.addEventListener('click', handleGithubLogin));

    // Toggle password visibility
    document.getElementById('togglePassword')?.addEventListener('click', () => {
        const pw = document.getElementById('password');
        if (!pw) return;
        pw.type = pw.type === 'password' ? 'text' : 'password';
    });
});