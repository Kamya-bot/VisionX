// ============================================
// VisionX - Core Application JavaScript
// ============================================

// ===== Global State Management =====
const appState = {
    currentUser: null,
    comparisons: [],
    currentComparison: null
};

// ===== Sidebar Toggle =====
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.querySelector('.sidebar');
const mainContent = document.querySelector('.main-content');

if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        if (mainContent) {
            mainContent.classList.toggle('expanded');
        }
    });
}

// ===== Active Navigation Link =====
function setActiveNavLink() {
    const currentPage = window.location.pathname.split('/').pop() || 'dashboard.html';
    const navLinks = document.querySelectorAll('.sidebar-nav a');
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPage) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

setActiveNavLink();

// ===== User Dropdown =====
const userAvatar = document.querySelector('.user-avatar');
const userDropdown = document.querySelector('.user-dropdown');

if (userAvatar && userDropdown) {
    userAvatar.addEventListener('click', (e) => {
        e.stopPropagation();
        userDropdown.classList.toggle('show');
    });
    document.addEventListener('click', (e) => {
        if (!userDropdown.contains(e.target) && !userAvatar.contains(e.target)) {
            userDropdown.classList.remove('show');
        }
    });
}

// ===== Notification Center =====
const notificationBtn = document.querySelector('.notification-btn');
const notificationDropdown = document.querySelector('.notification-dropdown');

if (notificationBtn && notificationDropdown) {
    notificationBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        notificationDropdown.classList.toggle('show');
    });
    document.addEventListener('click', (e) => {
        if (!notificationDropdown.contains(e.target) && !notificationBtn.contains(e.target)) {
            notificationDropdown.classList.remove('show');
        }
    });
}

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    }[type];
    toast.innerHTML = `<i class="fas ${icon}"></i><span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ===== Modal Functions =====
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('show');
        document.body.style.overflow = '';
    }
}

document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal(modal.id);
    });
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.show').forEach(modal => closeModal(modal.id));
    }
});

// ===== Local Storage Helpers =====
const storage = {
    save(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('Error saving to localStorage:', e);
            return false;
        }
    },
    load(key) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('Error loading from localStorage:', e);
            return null;
        }
    },
    remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('Error removing from localStorage:', e);
            return false;
        }
    }
};

// ===== Comparison Data Management =====
const comparisonManager = {
    getAll() {
        return storage.load('visionx_comparisons') || [];
    },
    getById(id) {
        return this.getAll().find(c => c.id === id);
    },
    save(comparison) {
        const comparisons = this.getAll();
        const index = comparisons.findIndex(c => c.id === comparison.id);
        if (index >= 0) {
            comparisons[index] = comparison;
        } else {
            comparisons.push(comparison);
        }
        storage.save('visionx_comparisons', comparisons);
        return comparison;
    },
    delete(id) {
        const comparisons = this.getAll().filter(c => c.id !== id);
        storage.save('visionx_comparisons', comparisons);
    },
    create(title, description, category) {
        const comparison = {
            id: 'comp_' + Date.now(),
            title, description, category,
            options: [], criteria: [], scores: {},
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        return this.save(comparison);
    }
};

// ===== Utility Functions =====
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function generateId(prefix = 'id') {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => { clearTimeout(timeout); func(...args); };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ===== Logout Function =====
function logout() {
    if (typeof clearCurrentUser === 'function') {
        clearCurrentUser();
    } else {
        localStorage.removeItem('currentUser');
        localStorage.removeItem('auth_token');
        localStorage.removeItem('isAuthenticated');
    }
    window.location.replace('login.html');
}

window.logout = logout;

// ===== Update user avatar in nav =====
function updateNavAvatar() {
    try {
        const userStr = localStorage.getItem('currentUser');
        if (!userStr) return;
        const user = JSON.parse(userStr);
        const initial = (user.avatar || user.name || user.email || 'U').charAt(0).toUpperCase();
        document.querySelectorAll('.user-avatar span, .user-name-initial').forEach(el => {
            el.textContent = initial;
        });
        document.querySelectorAll('.user-name').forEach(el => {
            el.textContent = user.name || user.email || 'User';
        });
    } catch (e) {}
}

// ===== Initialize App =====
document.addEventListener('DOMContentLoaded', () => {
    updateNavAvatar();
    console.log('✨ VisionX App Initialized');
});

// ===== Export for other modules =====
window.VisionX = {
    appState, storage, comparisonManager,
    showToast, openModal, closeModal,
    formatDate, generateId, debounce
};

console.log('📊 VisionX App loaded');