/**
 * VisionX — Auth Handler
 * Connects to the real FastAPI backend for login and registration.
 * Handles Render free-tier cold starts gracefully.
 */

// ─── Guard: redirect logged-in users away from login/register pages ──────────
(function() {
    const publicPages = ['login.html', 'register.html', 'index.html', ''];
    const page = window.location.pathname.split('/').pop();
    if (!publicPages.includes(page) && !isAuthenticated()) {
        window.location.href = 'login.html';
    }
})();


// ─── Server wake-up: ping backend on page load so it's ready ─────────────────
(function wakeUpServer() {
    fetch('https://visionx-mzqc.onrender.com/').catch(() => {});
})();


// ─── apiCall with longer timeout for cold starts ──────────────────────────────
async function apiCallWithRetry(endpoint, options = {}, timeoutMs = 55000) {
    const token = localStorage.getItem('auth_token');
    const config = {
        method: options.method || 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` }),
        },
        ...(options.body && { body: JSON.stringify(options.body) })
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(
            `https://visionx-mzqc.onrender.com/api/v1${endpoint}`,
            { ...config, signal: controller.signal }
        );
        clearTimeout(timeoutId);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || data.error || data.message || `Error ${response.status}`);
        }
        return data;

    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Server is taking too long to respond. Please try again.');
        }
        throw error;
    }
}


// ─── Cold-start banner ────────────────────────────────────────────────────────
function showWakingBanner() {
    if (document.getElementById('vx-waking-banner')) return;
    const banner = document.createElement('div');
    banner.id = 'vx-waking-banner';
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#1a56db;color:#fff;padding:10px 20px;font-size:13px;text-align:center;font-family:sans-serif;';
    banner.innerHTML = '⏳ <strong>Starting server...</strong> — Our free server sleeps when idle. This takes up to 30 seconds on first use. Please wait.';
    document.body.prepend(banner);
}

function hideWakingBanner() {
    const banner = document.getElementById('vx-waking-banner');
    if (banner) banner.remove();
}


// ─── Login ────────────────────────────────────────────────────────────────────

async function handleLogin(event) {
    if (event) event.preventDefault();

    const email    = document.getElementById('email')?.value?.trim();
    const password = document.getElementById('password')?.value;
    const submitBtn = document.querySelector('#loginForm button[type="submit"], button#loginBtn');

    if (!email || !password) {
        showNotification('Please enter your email and password', 'error');
        return;
    }

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging in...';
    }
    showWakingBanner();

    try {
        const data = await apiCallWithRetry('/auth/login', {
            method: 'POST',
            body: { email, password }
        });

        hideWakingBanner();
        saveUserSession(data, email);
        showNotification('Login successful! Redirecting...', 'success');
        window.location.replace('dashboard.html');

    } catch (error) {
        hideWakingBanner();
        showNotification(error.message || 'Login failed', 'error');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
        }
    }
}


// ─── Register ─────────────────────────────────────────────────────────────────

async function handleRegister(event) {
    if (event) event.preventDefault();

    const name            = document.getElementById('name')?.value?.trim();
    const email           = document.getElementById('email')?.value?.trim();
    const password        = document.getElementById('password')?.value;
    const confirmPassword = document.getElementById('confirmPassword')?.value;
    const submitBtn       = document.querySelector('#registerForm button[type="submit"]');

    if (!email || !password) {
        showNotification('Email and password are required', 'error');
        return;
    }
    if (confirmPassword && password !== confirmPassword) {
        showNotification('Passwords do not match', 'error');
        return;
    }
    if (password.length < 6) {
        showNotification('Password must be at least 6 characters', 'error');
        return;
    }

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';
    }
    showWakingBanner();

    try {
        const data = await apiCallWithRetry('/auth/register', {
            method: 'POST',
            body: { email, password, full_name: name || null }
        });

        hideWakingBanner();
        saveUserSession(data, email);
        showNotification('Account created! Welcome to VisionX 🎉', 'success');
        window.location.replace('dashboard.html');

    } catch (error) {
        hideWakingBanner();
        showNotification(error.message || 'Registration failed', 'error');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-user-plus"></i> Create Account';
        }
    }
}


// ─── Save session helper ──────────────────────────────────────────────────────

function saveUserSession(data, emailFallback) {
    const user = {
        id: data.user_id,
        email: data.email,
        name: data.full_name || emailFallback.split('@')[0],
        avatar: (data.full_name || emailFallback).charAt(0).toUpperCase(),
        cluster_id: data.cluster_id,
        cluster_label: data.cluster_label,
        created_at: new Date().toISOString()
    };
    setCurrentUser(user);
    localStorage.setItem('auth_token', data.access_token);
    localStorage.setItem('isAuthenticated', 'true');
}


// ─── Google / GitHub OAuth (not yet supported by backend) ────────────────────

function handleGoogleLogin() {
    showNotification('Google login coming soon! Please use email & password for now.', 'info');
}

function handleGithubLogin() {
    showNotification('GitHub login coming soon! Please use email & password for now.', 'info');
}


// ─── Logout ───────────────────────────────────────────────────────────────────

function handleLogout() {
    clearCurrentUser();
    window.location.replace('login.html');
}


// ─── Update nav with real user name/avatar ────────────────────────────────────

function updateNavUser() {
    const user = getCurrentUser();
    if (!user) return;
    document.querySelectorAll('.user-name').forEach(el => {
        el.textContent = user.name || user.email;
    });
    document.querySelectorAll('.user-avatar span').forEach(el => {
        el.textContent = user.avatar || user.name?.charAt(0).toUpperCase() || 'U';
    });
}
document.addEventListener('DOMContentLoaded', updateNavUser);


// ─── Wire up forms and buttons ────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Forms
    const loginForm = document.getElementById('loginForm');
    if (loginForm) loginForm.addEventListener('submit', handleLogin);

    const registerForm = document.getElementById('registerForm');
    if (registerForm) registerForm.addEventListener('submit', handleRegister);

    // Logout buttons
    document.querySelectorAll('.logout-btn, [data-action="logout"]')
        .forEach(btn => btn.addEventListener('click', handleLogout));

    // Google buttons (both login + register pages)
    document.querySelectorAll('.google-btn')
        .forEach(btn => btn.addEventListener('click', handleGoogleLogin));

    // GitHub buttons
    document.querySelectorAll('.github-btn')
        .forEach(btn => btn.addEventListener('click', handleGithubLogin));
});