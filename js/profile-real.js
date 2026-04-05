/**
 * VisionX — Profile Page
 * Fetches real user data from /auth/me and populates all profile fields.
 */

async function loadUserProfile() {
    const user = getCurrentUser();
    if (!user) { window.location.href = 'login.html'; return; }

    try {
        // Fetch fresh data from backend
        const profile = await apiCall('/auth/me');

        // Merge into localStorage so nav updates too
        const updated = {
            ...user,
            id: profile.user_id,
            email: profile.email,
            name: profile.full_name || user.name,
            cluster_id: profile.cluster_id,
            cluster_label: profile.cluster_label,
            created_at: profile.created_at
        };
        setCurrentUser(updated);

        // ── Form fields ──────────────────────────────────────────────────────
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };

        set('profileFullName',  profile.full_name || '');
        set('profileName',      profile.full_name || '');
        set('profileEmail',     profile.email);
        set('profileUserId',    profile.user_id);
        set('profileCluster',   profile.cluster_label || (profile.cluster_id !== null ? `Cluster ${profile.cluster_id}` : 'Not assigned yet'));

        // ── Display text ─────────────────────────────────────────────────────
        document.querySelectorAll('.user-name, .profile-user-name').forEach(el => {
            el.textContent = profile.full_name || profile.email.split('@')[0];
        });
        document.querySelectorAll('.user-email, .profile-user-email').forEach(el => {
            el.textContent = profile.email;
        });
        document.querySelectorAll('.user-avatar span').forEach(el => {
            el.textContent = (profile.full_name || profile.email).charAt(0).toUpperCase();
        });

        // Member since
        const memberSinceEl = document.getElementById('memberSince');
        if (memberSinceEl && profile.created_at) {
            const d = new Date(profile.created_at);
            memberSinceEl.textContent = d.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
        }

        // Total predictions stat
        const predCountEl = document.getElementById('totalPredictions');
        if (predCountEl) predCountEl.textContent = profile.total_predictions ?? 0;

        // Cluster badge
        const clusterBadge = document.getElementById('clusterBadge');
        if (clusterBadge) {
            clusterBadge.textContent = profile.cluster_label || 'Pending';
        }

    } catch (error) {
        console.error('Failed to load profile:', error);
        showNotification('Could not load profile: ' + error.message, 'error');

        // Fallback: show whatever is in localStorage
        const fallback = getCurrentUser();
        if (fallback) {
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };
            set('profileFullName', fallback.name);
            set('profileEmail',    fallback.email);
            set('profileUserId',   fallback.id);
        }
    }
}

async function saveProfile() {
    const nameInput = document.getElementById('profileFullName') || document.getElementById('profileName');
    const newName = nameInput?.value?.trim();
    if (!newName) { showNotification('Name cannot be empty', 'error'); return; }

    // Update localStorage immediately
    const user = getCurrentUser();
    if (user) {
        user.name = newName;
        user.full_name = newName;
        setCurrentUser(user);
    }

    // Note: a PATCH /auth/me endpoint could be added later for persistence
    showNotification('Profile updated!', 'success');
}

document.addEventListener('DOMContentLoaded', () => {
    loadUserProfile();

    const saveBtn = document.getElementById('saveProfileBtn');
    if (saveBtn) saveBtn.addEventListener('click', saveProfile);
});
