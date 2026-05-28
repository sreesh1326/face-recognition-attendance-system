/* ═══════════════════════════════════════════════════════
   FeD — app.js
   App Module: state management via server REST API,
               student registration, attendance, DOM rendering
   ═══════════════════════════════════════════════════════ */

const App = (() => {

  const API_BASE = window.location.origin + '/api';

  /* ── Local state (synced from server) ──────────────── */
  let students = [];
  let attendance = [];
  let _capturedImage = null;

  /* ── Captured image (temp, between capture & register) ─ */
  function setCapturedImage(dataUrl) { _capturedImage = dataUrl; }
  function getCapturedImage() { return _capturedImage; }

  /* ── Getters ─────────────────────────────────────────── */
  function getStudents() { return students; }
  function getAttendance() { return attendance; }

  /* ══════════════ SERVER HELPERS ═══════════════════════ */
  async function _api(endpoint, options = {}) {
    try {
      const resp = await fetch(`${API_BASE}${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || `Server error ${resp.status}`);
      }
      return data;
    } catch (err) {
      console.error(`[App] API ${endpoint}:`, err);
      throw err;
    }
  }

  /* ══════════════ FETCH DATA FROM SERVER ═══════════════ */
  async function fetchStudents() {
    try {
      const data = await _api('/users');
      students = data.users || [];
      _renderStudentList();
    } catch (err) {
      console.warn('[App] Could not fetch students:', err.message);
    }
  }

  async function fetchAttendance() {
    try {
      const data = await _api('/attendance');
      attendance = data.records || [];
    } catch (err) {
      console.warn('[App] Could not fetch attendance:', err.message);
    }
  }

  /* ══════════════ STUDENT REGISTRATION ════════════════ */
  async function registerStudent() {
    const name = document.getElementById('regName')?.value.trim();
    const roll = document.getElementById('regRoll')?.value.trim();
    const dept = document.getElementById('regDept')?.value.trim() || 'N/A';

    // Validation
    if (!name) { UI.toast('Please enter the student\'s full name.', 'error'); return; }
    if (!roll) { UI.toast('Please enter the roll number.', 'error'); return; }
    if (!_capturedImage) { UI.toast('Please capture the student\'s face first.', 'error'); return; }

    UI.toast('Registering student & processing face…', 'info');

    try {
      const data = await _api('/users/register', {
        method: 'POST',
        body: JSON.stringify({ name, roll, dept, photo: _capturedImage })
      });

      // Reset form
      ['regName', 'regRoll', 'regDept'].forEach(id => {
        const el = document.getElementById(id); if (el) el.value = '';
      });
      _capturedImage = null;
      const preview = document.getElementById('capturePreview');
      if (preview) preview.style.display = 'none';

      UI.toast(`${name} registered successfully!`, 'success');

      // Refresh student list from server
      await fetchStudents();

    } catch (err) {
      UI.toast(err.message || 'Registration failed — check server.', 'error');
    }
  }

  async function deleteStudent(roll) {
    const student = students.find(s => s.roll_number === roll);
    if (!student) return;
    if (!confirm(`Remove "${student.name}" (${roll}) from the system?`)) return;

    try {
      await _api(`/users/${roll}`, { method: 'DELETE' });
      UI.toast(`${student.name} removed from system.`, 'info');
      await fetchStudents();
    } catch (err) {
      UI.toast(err.message || 'Delete failed.', 'error');
    }
  }

  /* ── Student list DOM ───────────────────────────────── */
  function _renderStudentList() {
    const list = document.getElementById('studentList');
    const count = document.getElementById('studentCount');
    if (!list) return;

    if (count) count.textContent = students.length;

    if (students.length === 0) {
      list.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.2">
            <circle cx="32" cy="24" r="12"/>
            <path d="M8 56c0-13.25 10.75-24 24-24s24 10.75 24 24"/>
          </svg>
          <p>No students registered yet</p>
          <small>Register a student above to get started</small>
        </div>`;
      return;
    }

    list.innerHTML = `<div class="student-chips">
      ${students.map(s => `
        <div class="student-chip">
          <div class="s-avatar">${UI.initials(s.name)}</div>
          <div class="s-info">
            <div class="s-name">${_esc(s.name)}</div>
            <div class="s-roll">${_esc(s.roll_number)} &middot; ${_esc(s.department)}</div>
          </div>
          <button class="s-delete" onclick="App.deleteStudent('${_esc(s.roll_number)}')" title="Remove student">&times;</button>
        </div>`).join('')}
    </div>`;
  }

  /* ══════════════ ATTENDANCE ═══════════════════════════ */

  /* ── Records panel DOM ──────────────────────────────── */
  async function updateRecords() {
    try {
      // Fetch stats from server
      const stats = await _api('/stats');
      _setText('statTotal', stats.totalAttendance);
      _setText('statToday', stats.todayAttendance);
      _setText('statStudents', stats.totalUsers);

      // Fetch attendance records
      await fetchAttendance();
    } catch (err) {
      console.warn('[App] Stats fetch failed:', err.message);
    }

    const logEl = document.getElementById('attendanceLog');
    if (!logEl) return;

    if (attendance.length === 0) {
      logEl.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.2">
            <rect x="8" y="8" width="48" height="48" rx="4"/>
            <line x1="8" y1="24" x2="56" y2="24"/>
            <line x1="24" y1="8" x2="24" y2="56"/>
          </svg>
          <p>No attendance records yet</p>
          <small>Mark attendance to see records here</small>
        </div>`;
      return;
    }

    logEl.innerHTML = `
      <div class="table-wrap">
        <table class="att-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Roll No.</th>
              <th>Department</th>
              <th>Date &amp; Time (IST)</th>
              <th>Confidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${attendance.map((r, i) => `
              <tr>
                <td class="td-num">${String(attendance.length - i).padStart(2, '0')}</td>
                <td class="td-name">${_esc(r.name)}</td>
                <td class="td-roll">${_esc(r.roll_number)}</td>
                <td class="td-dept">${_esc(r.department)}</td>
                <td class="td-time">${UI.formatIST(r.marked_at)}</td>
                <td class="td-conf">${r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '—'}</td>
                <td><span class="badge badge-present">Present</span></td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  async function clearRecords() {
    if (!confirm('Clear ALL attendance records? This cannot be undone.')) return;
    try {
      await _api('/attendance', { method: 'DELETE' });
      attendance = [];
      await updateRecords();
      UI.toast('All attendance records cleared.', 'info');
    } catch (err) {
      UI.toast('Failed to clear records.', 'error');
    }
  }

  /* ── Helpers ─────────────────────────────────────────── */
  function _setText(id, val) {
    const el = document.getElementById(id); if (el) el.textContent = val;
  }
  function _esc(str) {
    return String(str || '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /* ── Bootstrap ──────────────────────────────────────── */
  async function init() {
    UI.startClock();
    Recognition.loadModels();
    await fetchStudents();
    await updateRecords();
  }

  /* ── Public API ─────────────────────────────────────── */
  return {
    init,
    setCapturedImage, getCapturedImage,
    getStudents, getAttendance,
    registerStudent, deleteStudent,
    updateRecords, clearRecords,
    fetchStudents
  };

})();

/* ── Entry point ─────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  App.init()
  document.getElementById('tab-register').addEventListener('click', () => UI.switchTab('register'));
  document.getElementById('tab-attendance').addEventListener('click', () => UI.switchTab('attendance'));
  document.getElementById('tab-records').addEventListener('click', () => UI.switchTab('records'));
});
