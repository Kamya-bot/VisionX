/**
 * VisionX — Auth Handler
 * Connects to the real FastAPI backend for login and registration.
 */

// ─── Guard: redirect logged-in users away from login/register pages ──────────
(function() {
    const publicPages = ['login.html', 'register.html', 'index.html', ''];
    const page = window.location.pathname.split('/').pop();
    if (!publicPages.includes(page) && !isAuthenticated()) {
        window.location.href = 'login.html';
    }
})();


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

    try {
        const data = await apiCall('/auth/login', {
            method: 'POST',
            body: { email, password }
        });

        const user = {
            id: data.user_id,
            email: data.email,
            name: data.full_name || email.split('@')[0],
            avatar: (data.full_name || email).charAt(0).toUpperCase(),
            cluster_id: data.cluster_id,
            cluster_label: data.cluster_label,
            created_at: new Date().toISOString()
        };
        setCurrentUser(user);
        localStorage.setItem('auth_token', data.access_token);
        localStorage.setItem('isAuthenticated', 'true');

        showNotification('Login successful! Redirecting...', 'success');
        window.location.replace('dashboard.html');

    } catch (error) {
        showNotification(error.message || 'Login failed', 'error');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Login';
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

    try {
        const data = await apiCall('/auth/register', {
            method: 'POST',
            body: { email, password, full_name: name || null }
        });

        const user = {
            id: data.user_id,
            email: data.email,
            name: data.full_name || email.split('@')[0],
            avatar: (data.full_name || email).charAt(0).toUpperCase(),
            cluster_id: data.cluster_id,
            cluster_label: data.cluster_label,
            created_at: new Date().toISOString()
        };
        setCurrentUser(user);
        localStorage.setItem('auth_token', data.access_token);
        localStorage.setItem('isAuthenticated', 'true');

        showNotification('Account created! Welcome to VisionX', 'success');
        window.location.replace('dashboard.html');

    } catch (error) {
        showNotification(error.message || 'Registration failed', 'error');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Create Account';
        }
    }
}


// ─── Logout ───────────────────────────────────────────────────────────────────

function handleLogout() {
    clearCurrentUser();
    window.location.replace('login.html');
}


// ─── Update nav with real user name/avatar ───────────────────────────────────

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


// ─── Wire up forms ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    if (loginForm) loginForm.addEventListener('submit', handleLogin);

    const registerForm = document.getElementById('registerForm');
    if (registerForm) registerForm.addEventListener('submit', handleRegister);

    const logoutBtns = document.querySelectorAll('.logout-btn, [data-action="logout"]');
    logoutBtns.forEach(btn => btn.addEventListener('click', handleLogout));
});