/* ═══════════════════════════════════════════════════════
   FeD — app.js
   App Module: state management, student registration,
               attendance logging, DOM rendering
   ═══════════════════════════════════════════════════════ */

const App = (() => {

  /* ── Persistent state (localStorage) ─────────────────── */
  const KEYS = { students: 'fed_v2_students', attendance: 'fed_v2_attendance' };

  let students = _load(KEYS.students, []);
  let attendance = _load(KEYS.attendance, []);
  let _capturedImage = null;

  function _load(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) || fallback; } catch { return fallback; }
  }
  function _save(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {
      console.warn('[App] localStorage write failed:', e);
    }
  }

  /* ── Captured image (temp, between capture & register) ─ */
  function setCapturedImage(dataUrl) { _capturedImage = dataUrl; }
  function getCapturedImage() { return _capturedImage; }

  /* ── Getters ─────────────────────────────────────────── */
  function getStudents() { return students; }
  function getAttendance() { return attendance; }

  /* ══════════════ STUDENT REGISTRATION ════════════════ */
  async function registerStudent() {
    const name = document.getElementById('regName')?.value.trim();
    const roll = document.getElementById('regRoll')?.value.trim();
    const dept = document.getElementById('regDept')?.value.trim() || 'N/A';

    // Validation
    if (!name) { UI.toast('Please enter the student\'s full name.', 'error'); return; }
    if (!roll) { UI.toast('Please enter the roll number.', 'error'); return; }
    if (!_capturedImage) { UI.toast('Please capture the student\'s face first.', 'error'); return; }
    if (students.find(s => s.roll === roll)) {
      UI.toast(`Roll number "${roll}" is already registered.`, 'error'); return;
    }

    UI.toast('Processing face data…', 'info');

    // Try extracting face descriptor
    let descriptor = null;
    if (Recognition.modelsReady()) {
      descriptor = await Recognition.extractDescriptor(_capturedImage);
      if (!descriptor) {
        UI.toast('No face detected in captured image — try again.', 'error');
        return;
      }
    }

    const student = {
      id: Date.now(),
      name, roll, dept,
      photo: _capturedImage,
      descriptor: descriptor,
      registeredAt: new Date().toISOString()
    };

    students.push(student);
    _save(KEYS.students, students);

    // Reset form
    ['regName', 'regRoll', 'regDept'].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = '';
    });
    _capturedImage = null;
    const preview = document.getElementById('capturePreview');
    if (preview) preview.style.display = 'none';

    UI.toast(`${name} registered successfully!`, 'success');
    _renderStudentList();
  }

  function deleteStudent(roll) {
    const student = students.find(s => s.roll === roll);
    if (!student) return;
    if (!confirm(`Remove "${student.name}" (${roll}) from the system?`)) return;
    students = students.filter(s => s.roll !== roll);
    _save(KEYS.students, students);
    _renderStudentList();
    UI.toast(`${student.name} removed from system.`, 'info');
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
            <div class="s-roll">${_esc(s.roll)} &middot; ${_esc(s.dept)}</div>
          </div>
          <button class="s-delete" onclick="App.deleteStudent('${_esc(s.roll)}')" title="Remove student">&times;</button>
        </div>`).join('')}
    </div>`;
  }

  /* ══════════════ ATTENDANCE ═══════════════════════════ */
  function markAttendance(student) {
    const ist = UI.getIST();
    const alreadyMarked = attendance.some(r => r.roll === student.roll && new Date(r.timestamp).toDateString() === todayStr);
    if (alreadyMarked) return;
    const record = {
      id: Date.now(),
      name: student.name,
      roll: student.roll,
      dept: student.dept,
      timestamp: ist.toISOString(),
      formatted: UI.formatIST(ist.toISOString())
    };

    attendance.unshift(record);
    _save(KEYS.attendance, attendance);

    // Update result banner
    const res = document.getElementById('recResult');
    const rn = document.getElementById('recName');
    const rm = document.getElementById('recMeta');
    const rt = document.getElementById('recTime');
    if (res && rn && rm && rt) {
      rn.textContent = student.name;
      rm.textContent = `${student.roll}  ·  ${student.dept}`;
      rt.textContent = `Attendance marked at ${record.formatted}`;
      res.classList.add('show');
    }

    UI.toast(`✓ ${student.name} — attendance marked`, 'success');
    updateRecords();
  }

  /* ── Records panel DOM ──────────────────────────────── */
  function updateRecords() {
    const todayStr = UI.getIST().toDateString();
    const todayLogs = attendance.filter(r =>
      new Date(new Date(r.timestamp).toLocaleString('en-US', { timezone: 'Asia/Kolkata' })).toDateString() === todayStr
    );

    _setText('statTotal', attendance.length);
    _setText('statToday', todayLogs.length);
    _setText('statStudents', students.length);

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
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${attendance.map((r, i) => `
              <tr>
                <td class="td-num">${String(attendance.length - i).padStart(2, '0')}</td>
                <td class="td-name">${_esc(r.name)}</td>
                <td class="td-roll">${_esc(r.roll)}</td>
                <td class="td-dept">${_esc(r.dept)}</td>
                <td class="td-time">${_esc(r.formatted)}</td>
                <td><span class="badge badge-present">Present</span></td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  function clearRecords() {
    if (!confirm('Clear ALL attendance records? This cannot be undone.')) return;
    attendance = [];
    _save(KEYS.attendance, attendance);
    updateRecords();
    UI.toast('All attendance records cleared.', 'info');
  }

  /* ── Helpers ─────────────────────────────────────────── */
  function _setText(id, val) {
    const el = document.getElementById(id); if (el) el.textContent = val;
  }
  function _esc(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /* ── Bootstrap ──────────────────────────────────────── */
  function init() {
    UI.startClock();
    Recognition.loadModels();
    _renderStudentList();
    updateRecords();
  }

  /* ── Public API ─────────────────────────────────────── */
  return {
    init,
    setCapturedImage, getCapturedImage,
    getStudents, getAttendance,
    registerStudent, deleteStudent,
    markAttendance, updateRecords, clearRecords
  };

})();

/* ── Entry point ─────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  App.init()
  document.getElementById('tab-register').addEventListener('click', () => UI.switchTab('register'));
  document.getElementById('tab-attendance').addEventListener('click', () => UI.switchTab('attendance'));
  document.getElementById('tab-records').addEventListener('click', () => UI.switchTab('records'));
});
