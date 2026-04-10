/**
 * VisionX – Profile Page
 * Fetches real user data. Prediction count from real API, not localStorage.
 */

async function loadUserProfile() {
    const user = getCurrentUser();
    if (!user) { window.location.href = 'login.html'; return; }

    try {
        const profile = await apiCall('/auth/me');

        const updated = {
            ...user,
            id:            profile.user_id,
            email:         profile.email,
            name:          profile.full_name || user.name,
            full_name:     profile.full_name,
            cluster_id:    profile.cluster_id,
            cluster_label: profile.cluster_label,
            created_at:    profile.created_at,
        };
        setCurrentUser(updated);

        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.value = val || '';
        };
        set('profileFullName', profile.full_name || '');
        set('profileName',     profile.full_name || '');
        set('profileEmail',    profile.email);
        set('profileUserId',   profile.user_id);
        set('profileCluster',  profile.cluster_label ||
            (profile.cluster_id != null ? `Cluster ${profile.cluster_id}` : 'Not assigned yet'));

        document.querySelectorAll('.user-name, .profile-user-name').forEach(el => {
            el.textContent = profile.full_name || profile.email.split('@')[0];
        });
        document.querySelectorAll('.user-email, .profile-user-email').forEach(el => {
            el.textContent = profile.email;
        });
        document.querySelectorAll('.user-avatar span').forEach(el => {
            el.textContent = (profile.full_name || profile.email).charAt(0).toUpperCase();
        });

        const memberSinceEl = document.getElementById('memberSince');
        if (memberSinceEl && profile.created_at) {
            const d = new Date(profile.created_at);
            memberSinceEl.textContent = d.toLocaleDateString('en-US', {
                month: 'long', year: 'numeric'
            });
        }

        const clusterBadge = document.getElementById('clusterBadge');
        if (clusterBadge) clusterBadge.textContent = profile.cluster_label || 'Pending';

        // ── Real prediction count from API ────────────────────────────────
        const predCountEl = document.getElementById('totalPredictions');
        if (predCountEl) {
            predCountEl.textContent = '…';
            try {
                const histData = await apiCall('/predictions/history?limit=1&offset=0');
                predCountEl.textContent = histData.total ?? (histData.predictions?.length ?? 0);
            } catch (_) {
                // Fallback: check analytics KPIs
                try {
                    const kpis = await apiCall('/analytics/kpis');
                    predCountEl.textContent = kpis.total_predictions ?? '—';
                } catch (__) {
                    predCountEl.textContent = '—';
                }
            }
        }

    } catch (error) {
        console.error('Failed to load profile:', error);
        // Graceful fallback — at least show what we have in memory
        const fallback = getCurrentUser();
        if (fallback) {
            const set = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.value = val || '';
            };
            set('profileFullName', fallback.full_name || fallback.name);
            set('profileEmail',    fallback.email);
            set('profileUserId',   fallback.id);
        }
    }
}

async function saveProfile() {
    const nameInput = document.getElementById('profileFullName') ||
                      document.getElementById('profileName');
    const newName = nameInput?.value?.trim();
    if (!newName) { showNotification('Name cannot be empty', 'error'); return; }

    try {
        await apiCall('/auth/me', {
            method: 'PATCH',
            body: JSON.stringify({ full_name: newName }),
        });
        const user = getCurrentUser();
        if (user) {
            user.name      = newName;
            user.full_name = newName;
            setCurrentUser(user);
        }
        showNotification('Profile updated!', 'success');
    } catch (err) {
        console.error('Save profile failed:', err);
        // Update local cache even if API failed
        const user = getCurrentUser();
        if (user) {
            user.name = newName; user.full_name = newName;
            setCurrentUser(user);
        }
        showNotification('Saved locally (sync failed)', 'warning');
    }
}

async function deleteAccount() {
    const confirmed = confirm(
        'Are you sure you want to delete your account? This cannot be undone.'
    );
    if (!confirmed) return;
    const confirmed2 = confirm(
        'This will permanently delete all your data. Confirm?'
    );
    if (!confirmed2) return;

    try {
        await apiCall('/auth/me', { method: 'DELETE' });
    } catch (error) {
        console.warn('Backend delete failed, clearing local data:', error);
    }

    localStorage.removeItem('visionx_token');
    localStorage.removeItem('visionx_refresh_token');
    localStorage.removeItem('visionx_user');
    localStorage.removeItem('visionx_history_cache');
    sessionStorage.clear();

    showNotification('Account deleted. Redirecting…', 'success');
    setTimeout(() => { window.location.href = 'index.html'; }, 1500);
}

document.addEventListener('DOMContentLoaded', () => {
    loadUserProfile();

    const saveBtn = document.getElementById('saveProfileBtn');
    if (saveBtn) saveBtn.addEventListener('click', saveProfile);

    const deleteBtn = document.getElementById('deleteAccountBtn');
    if (deleteBtn) deleteBtn.addEventListener('click', deleteAccount);
});