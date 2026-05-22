/* ═══════════════════════════════════════════════════════
   FeD — recognition.js
   Recognition Module: communicates with backend ML pipeline
   No client-side face-api.js — all ML runs on the server
   with custom-trained CNN models and PostgreSQL storage
   ═══════════════════════════════════════════════════════ */

const Recognition = (() => {

  const API_BASE = window.location.origin + '/api';

  /* status: 'loading' | 'ready' | 'error' */
  let _status = 'loading';

  /* cooldown: prevent spamming the same attendance */
  let _lastMarkedRoll = null;
  let _lastMarkedAt   = 0;
  const COOLDOWN_MS   = 6000;

  /* recognition interval */
  let _recognitionInterval = null;

  function setScanStatus(state, message) {
    const badge = document.getElementById('scanState');
    const copy = document.getElementById('scanCopy');
    if (!badge || !copy) return;

    badge.className = `scan-pill ${state}`;
    const labels = {
      offline: 'Camera idle',
      searching: 'Scanning',
      detected: 'Face detected',
      matched: 'Match found',
      error: 'Scan blocked'
    };

    badge.textContent = labels[state] || 'Scanning';
    copy.textContent = message;
  }

  /* ── Check backend health + ML pipeline status ─────── */
  async function loadModels() {
    const dot    = document.getElementById('loaderDot');
    const text   = document.getElementById('loaderText');
    const fill   = document.getElementById('loaderFill');

    const step = (pct, msg) => {
      if (fill) fill.style.width = pct + '%';
      if (text) text.textContent = msg;
    };

    try {
      step(20, 'Connecting to ML backend server…');

      const resp = await fetch(`${API_BASE}/stats`);
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);

      step(60, 'Loading custom CNN embedding model…');
      const data = await resp.json();

      step(80, 'Initializing face recognition pipeline…');

      if (data.success) {
        step(100, `ML pipeline ready — ${data.totalUsers} users, ${data.pipeline.totalEmbeddings} embeddings`);
        if (dot) { dot.className = 'loader-dot ready'; }
        _status = 'ready';
        UI.setStatus('online', 'ONLINE');
        UI.toast('Custom face recognition engine ready.', 'success');
      } else {
        throw new Error('Pipeline not ready');
      }

    } catch (err) {
      console.error('[Recognition] Backend connection failed:', err);
      if (dot)  dot.className = 'loader-dot error';
      if (text) text.textContent = 'Backend offline — start the server with: cd server && npm start';
      _status = 'error';
      UI.setStatus('offline', 'OFFLINE');
      UI.toast('Backend server not reachable. Start the server first.', 'error');
    }
  }

  function modelsReady() { return _status === 'ready'; }

  /* ── Extract face embedding via backend ────────────── */
  async function extractDescriptor(imageDataUrl) {
    if (_status !== 'ready') return null;
    try {
      const resp = await fetch(`${API_BASE}/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo: imageDataUrl })
      });
      const data = await resp.json();
      // For registration, we just need to confirm a face was processed
      return data.success ? 'valid' : null;
    } catch (err) {
      console.error('[Recognition] extractDescriptor error:', err);
      return null;
    }
  }

  /* ── Attendance recognition loop (called per frame) ── */
  async function runDetection(video, canvas, ctx) {
    if (_status !== 'ready') {
      setScanStatus('searching', 'Connecting to ML backend…');
      return;
    }
    await _serverDetection(video, canvas, ctx);
  }

  async function _serverDetection(video, canvas, ctx) {
    // Capture current frame as JPEG data URL
    const W = video.videoWidth, H = video.videoHeight;
    canvas.width  = W;
    canvas.height = H;
    ctx.drawImage(video, 0, 0, W, H);
    const frameDataUrl = canvas.toDataURL('image/jpeg', 0.7);

    try {
      const resp = await fetch(`${API_BASE}/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo: frameDataUrl })
      });
      const data = await resp.json();

      if (!data.success) {
        setScanStatus('searching', 'Scanning live feed for faces…');
        return;
      }

      if (data.matched) {
        const user = data.user;
        const conf = (data.confidence * 100).toFixed(1);

        // Draw bounding box (centered estimate since server returns match only)
        const bw = 180, bh = 220;
        const bx = (W - bw) / 2, by = (H - bh) / 2;
        _drawBox(ctx, bx, by, bw, bh, `${user.name} (${conf}%)`, true);

        setScanStatus('matched', `${user.name} recognized with ${conf}% confidence.`);
        _tryMarkAttendance(user, data.confidence);
      } else {
        // Draw unknown box
        const bw = 180, bh = 220;
        const bx = (W - bw) / 2, by = (H - bh) / 2;
        _drawBox(ctx, bx, by, bw, bh, 'Unknown', false);

        const reason = data.bestScore
          ? `Face detected but no match (best: ${(data.bestScore * 100).toFixed(1)}%)`
          : 'No registered faces found';
        setScanStatus('detected', reason);
      }
    } catch (err) {
      console.error('[Recognition] Detection error:', err);
      setScanStatus('error', 'Recognition request failed — check server connection.');
    }
  }

  /* ── Drawing helpers ────────────────────────────────── */
  function _drawBox(ctx, x, y, w, h, label, matched) {
    const color = matched ? '#34d399' : '#f87171';
    const labelH = 24;

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.strokeRect(x, y, w, h);

    // Corner accents
    const cs = 14;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx,cy], i) => {
      ctx.beginPath();
      ctx.moveTo(cx + (i%2===0 ? cs : -cs), cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + (i<2 ? cs : -cs));
      ctx.stroke();
    });

    // Label background
    ctx.fillStyle = matched ? 'rgba(52,211,153,0.88)' : 'rgba(248,113,113,0.88)';
    ctx.fillRect(x, y - labelH, w, labelH);

    // Label text
    ctx.fillStyle = '#fff';
    ctx.font      = 'bold 12px "Syne", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + 6, y - labelH / 2);
  }

  /* ── Attendance marker ──────────────────────────────── */
  async function _tryMarkAttendance(user, confidence) {
    const now = Date.now();
    if (user.roll === _lastMarkedRoll && now - _lastMarkedAt < COOLDOWN_MS) return;
    _lastMarkedRoll = user.roll;
    _lastMarkedAt   = now;

    // Mark via backend API
    try {
      await fetch(`${API_BASE}/attendance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId:     user.userId,
          name:       user.name,
          roll:       user.roll,
          dept:       user.dept,
          confidence: confidence
        })
      });
    } catch (err) {
      console.error('[Recognition] Attendance marking error:', err);
    }

    App.showAttendanceResult(user, confidence);
    UI.toast(`✓ ${user.name} — attendance marked`, 'success');

    setTimeout(() => Camera.stopAttendanceCamera(), 800);
  }

  /* ── Public API ─────────────────────────────────────── */
  return { loadModels, modelsReady, extractDescriptor, runDetection, setScanStatus };

})();
