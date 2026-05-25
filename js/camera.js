/* ═══════════════════════════════════════════════════════
   FeD — camera.js
   Camera Module: webcam control for registration & attendance
   ═══════════════════════════════════════════════════════ */

const Camera = (() => {

  let regStream = null;
  let attStream = null;
  let attRafId = null;
  let regFaceLoopOn = false;
  let attFaceLoopOn = false;

  function clearRegOverlay() {
    const canvas = document.getElementById('regOverlay');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 0;
    canvas.height = 0;
  }

  /* ── Shared: getUserMedia wrapper ─────────────────── */
  async function openCamera() {
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
    });
  }

  /* ══════════════ REGISTRATION CAMERA ══════════════════ */
  async function startRegCamera() {
    if (regStream) return;                   // already open
    try {
      regStream = await openCamera();

      const video = document.getElementById('regVideo');
      const preview = document.getElementById('capturePreview');

      clearRegOverlay();
      if (preview) preview.style.display = 'none';
      video.srcObject = regStream;
      video.style.display = 'block';
      await video.play().catch(() => { });

      document.getElementById('regCamIdle').style.display = 'none';
      document.getElementById('regFrame').style.visibility = 'visible';
      document.getElementById('regScan').style.display = 'block';
      document.getElementById('btnCapture').disabled = false;

      video.addEventListener('play', () => _regFaceLoop(video), { once: true });
    } catch (err) {
      UI.toast('Camera access denied — please allow camera permission.', 'error');
      console.error('[Camera] startRegCamera:', err);
    }
  }

  async function _regFaceLoop(video) {
    regFaceLoopOn = true;
    const badge = document.getElementById('regFaceBadge');

    const loop = async () => {
      if (!regFaceLoopOn || !regStream) return;
      try {
        if (Recognition.modelsReady()) {
          const det = await faceapi.detectSingleFace(
            video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.4 })
          );
          badge.classList.toggle('show', !!det);
        }
      } catch (_) { /* silently ignore transient errors */ }
      if (regFaceLoopOn && regStream) requestAnimationFrame(loop);
    };
    loop();
  }

  function stopRegCamera() {
    regFaceLoopOn = false;
    if (regStream) { regStream.getTracks().forEach(t => t.stop()); regStream = null; }

    const video = document.getElementById('regVideo');
    if (video) { video.srcObject = null; video.style.display = 'none'; }
    clearRegOverlay();

    document.getElementById('regCamIdle').style.display = 'flex';
    document.getElementById('regFrame').style.visibility = 'hidden';
    document.getElementById('regScan').style.display = 'none';
    document.getElementById('regFaceBadge').classList.remove('show');
    document.getElementById('btnCapture').disabled = true;
  }

  async function captureForReg() {
    const video = document.getElementById('regVideo');
    if (!video || !regStream) { UI.toast('Open the camera first.', 'error'); return; }

    // Draw full frame to hidden canvas
    const canvas = document.getElementById('regOverlay');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    // Draw cropped square to preview
    const preview = document.getElementById('previewCanvas');
    preview.width = 200; preview.height = 200;
    const ctx = preview.getContext('2d');
    const side = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, 200, 200);

    // Store data URI for descriptor extraction
    App.setCapturedImage(canvas.toDataURL('image/jpeg', 0.85));

    stopRegCamera();

    document.getElementById('capturePreview').style.display = 'block';
    UI.toast('Face captured — fill in details and register.', 'success');
  }

  /* ══════════════ ATTENDANCE CAMERA ═══════════════════ */
  async function startAttendanceCamera() {
    const students = App.getStudents();
    if (students.length === 0) {
      UI.toast('No students registered yet!', 'error');
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      UI.toast('This browser does not support camera access.', 'error');
      Recognition.setScanStatus('error', 'Your browser does not expose camera APIs for live face scanning.');
      return;
    }

    if (attStream) return;

    try {
      Recognition.setScanStatus('searching', 'Requesting camera access so live face scanning can start.');
      attStream = await openCamera();

      const video = document.getElementById('attVideo');
      video.srcObject = attStream;
      video.style.display = 'block';

      document.getElementById('attCamIdle').style.display = 'none';
      document.getElementById('attScan').style.display = 'block';
      document.getElementById('btnStartAtt').disabled = true;
      document.getElementById('btnStopAtt').disabled = false;

      Recognition.setScanStatus('searching', 'Camera enabled. Scanning the live feed for registered faces.');

      video.addEventListener('play', () => _attRecognitionLoop(video), { once: true });
    } catch (err) {
      UI.toast('Camera access denied — please allow camera permission.', 'error');
      Recognition.setScanStatus('error', 'Camera permission was denied. Allow access and try again.');
      console.error('[Camera] startAttendanceCamera:', err);
    }
  }

  async function _attRecognitionLoop(video) {
    const canvas = document.getElementById('attCanvas');
    const ctx = canvas.getContext('2d');

    const loop = async () => {
      if (!attStream || video.paused || video.ended)
        attRafId = null; s
      return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      await Recognition.runDetection(video, canvas, ctx);

      attRafId = requestAnimationFrame(loop);
    };
    loop();
  }

  function stopAttendanceCamera() {
    if (attRafId) { cancelAnimationFrame(attRafId); attRafId = null; }
    if (attStream) { attStream.getTracks().forEach(t => t.stop()); attStream = null; }

    const video = document.getElementById('attVideo');
    const canvas = document.getElementById('attCanvas');
    if (video) { video.srcObject = null; video.style.display = 'none'; }
    if (canvas) { canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height); }

    document.getElementById('attCamIdle').style.display = 'flex';
    document.getElementById('attScan').style.display = 'none';
    document.getElementById('btnStartAtt').disabled = false;
    document.getElementById('btnStopAtt').disabled = true;
    Recognition.setScanStatus('offline', 'Enable the camera to start real-time face scanning.');

    const res = document.getElementById('recResult');
    if (res) res.classList.remove('show');
  }

  /* ── Public API ─────────────────────────────────────── */
  return {
    startRegCamera,
    stopRegCamera,
    captureForReg,
    startAttendanceCamera,
    stopAttendanceCamera
  };

})();
