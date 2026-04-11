/* ═══════════════════════════════════════════════
   RespiAI — main.js
   Interactive dashboard logic with browser mic recording
   ═══════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  // ─── Drag & Drop File Upload ───
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileInfo = document.getElementById('file-info');
  const uploadForm = document.getElementById('upload-form');
  const analyzeBtn = document.getElementById('analyze-btn');
  const loader = document.getElementById('loader');

  if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        showFileInfo(fileInput.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) {
        showFileInfo(fileInput.files[0]);
      }
    });
  }

  function showFileInfo(file) {
    if (!fileInfo) return;
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    fileInfo.innerHTML = `
      <div class="mt-3 p-3 glass-card">
        <i class="fa-solid fa-file-audio text-info me-2"></i>
        <strong>${file.name}</strong>
        <span class="text-muted ms-2">${sizeMB} MB</span>
      </div>`;
    fileInfo.style.display = 'block';
    previewWaveform(file);
  }

  function previewWaveform(file) {
    const canvas = document.getElementById('waveform-preview');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const reader = new FileReader();
    reader.onload = function (e) {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtx.decodeAudioData(e.target.result, (buffer) => {
        const data = buffer.getChannelData(0);
        canvas.style.display = 'block';
        drawStaticWaveform(ctx, canvas, data);
        audioCtx.close();
      });
    };
    reader.readAsArrayBuffer(file);
  }

  function drawStaticWaveform(ctx, canvas, data) {
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 120;
    ctx.clearRect(0, 0, w, h);

    const step = Math.ceil(data.length / w);
    const amp = h / 2;

    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#00d4ff');
    gradient.addColorStop(1, '#7b2fff');

    ctx.beginPath();
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 1.5;

    for (let i = 0; i < w; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const val = data[(i * step) + j] || 0;
        if (val < min) min = val;
        if (val > max) max = val;
      }
      ctx.moveTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }
    ctx.stroke();
  }

  // ─── Form Submit with AJAX ───
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!fileInput || !fileInput.files.length) {
        alert('Please select an audio file first.');
        return;
      }
      if (analyzeBtn) analyzeBtn.style.display = 'none';
      if (loader) loader.style.display = 'block';

      const formData = new FormData(uploadForm);
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData,
          headers: { 'Accept': 'application/json' }
        });

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const data = await response.json();
          if (!response.ok || data.error) {
            alert('Error: ' + (data.error || 'Upload failed'));
            resetUploadUI();
            return;
          }
          if (data.status === 'success') {
            submitToRender(data.result);
          }
        } else {
          const html = await response.text();
          document.open();
          document.write(html);
          document.close();
        }
      } catch (err) {
        console.error(err);
        alert('Network error during prediction.');
        resetUploadUI();
      }
    });
  }

  function resetUploadUI() {
    if (analyzeBtn) analyzeBtn.style.display = 'block';
    if (loader) loader.style.display = 'none';
  }

  function submitToRender(resultData) {
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = '/render-result';
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'result_data';
    input.value = JSON.stringify(resultData);
    form.appendChild(input);
    document.body.appendChild(form);
    form.submit();
  }


  // ═══════════════════════════════════════════════
  //  BROWSER MICROPHONE RECORDING
  //  Uses MediaRecorder API + Web Audio AnalyserNode
  // ═══════════════════════════════════════════════

  const micBtn = document.getElementById('mic-btn');
  const micStatus = document.getElementById('mic-status');
  const countdownEl = document.getElementById('countdown');
  const countdownNum = document.getElementById('countdown-num');
  const progressRing = document.getElementById('progress-ring');
  const micLoader = document.getElementById('mic-loader');
  const liveWaveformBox = document.getElementById('live-waveform-box');
  const liveWaveformCanvas = document.getElementById('live-waveform');

  let mediaRecorder = null;
  let audioChunks = [];
  let analyserNode = null;
  let audioContext = null;
  let waveformAnimId = null;
  let recognition = null;
  let lastSpeechTime = 0;
  const liveCaption = document.getElementById('live-caption');

  if (micBtn) {
    micBtn.addEventListener('click', startRecording);
  }

  async function startRecording() {
    // Request microphone permission
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      alert('Microphone access denied. Please allow microphone permission and try again.');
      console.error('getUserMedia error:', err);
      return;
    }

    // Disable button
    micBtn.disabled = true;
    micBtn.classList.add('recording');
    if (micStatus) micStatus.textContent = '🔴 Recording...';

    // Set up Web Audio API for real-time waveform
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    source.connect(analyserNode);

    lastSpeechTime = 0;

    // Show waveform canvas
    if (liveWaveformBox) liveWaveformBox.style.display = 'block';

    // Start drawing real-time waveform
    drawLiveWaveform();

    // Set up MediaRecorder
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      // Stop all tracks
      stream.getTracks().forEach(t => t.stop());
      // Stop waveform animation
      if (waveformAnimId) cancelAnimationFrame(waveformAnimId);
      if (audioContext) audioContext.close();
      if (recognition) recognition.stop();

      // Create audio blob and send to server
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      sendRecordedAudio(audioBlob);
    };

    // Start recording & transcription
    mediaRecorder.start();

    // ── Siri-style Live Transcription ──
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        if (liveCaption) {
          liveCaption.style.display = 'block';
          liveCaption.classList.add('fade-in');
          liveCaption.textContent = 'Listening...';
        }
      };

      recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }
        
        const currentTranscript = (finalTranscript + interimTranscript).trim();
        if (liveCaption && currentTranscript !== '') {
          liveCaption.textContent = `"${currentTranscript}"`;
          lastSpeechTime = Date.now();
        }
      };

      try {
        recognition.start();
      } catch (e) {
        console.warn('Speech recognition start failed', e);
      }
    }

    // Countdown timer (5 seconds)
    const totalSeconds = 5;
    let remaining = totalSeconds;
    const circumference = 339.292; // 2πr for r=54

    if (countdownEl) countdownEl.style.display = 'flex';
    if (progressRing) progressRing.style.strokeDashoffset = '0';
    if (countdownNum) countdownNum.textContent = remaining;

    const timer = setInterval(() => {
      remaining--;
      if (countdownNum) countdownNum.textContent = remaining;
      if (progressRing) {
        const offset = circumference * ((totalSeconds - remaining) / totalSeconds);
        progressRing.style.strokeDashoffset = offset;
      }
      if (remaining <= 0) {
        clearInterval(timer);
        // Stop recording after 5 seconds
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
        if (micStatus) micStatus.textContent = '⏳ Processing...';
        if (countdownEl) countdownEl.style.display = 'none';
        if (liveWaveformBox) liveWaveformBox.style.display = 'none';
        if (micLoader) micLoader.style.display = 'block';
      }
    }, 1000);
  }

  function drawLiveWaveform() {
    if (!analyserNode || !liveWaveformCanvas) return;

    const canvas = liveWaveformCanvas;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      waveformAnimId = requestAnimationFrame(draw);
      analyserNode.getByteTimeDomainData(dataArray);

      const w = canvas.width = canvas.offsetWidth * 2;
      const h = canvas.height = 160;
      ctx.clearRect(0, 0, w, h);

      // Gradient stroke
      const gradient = ctx.createLinearGradient(0, 0, w, 0);
      gradient.addColorStop(0, '#00d4ff');
      gradient.addColorStop(0.5, '#7b2fff');
      gradient.addColorStop(1, '#ef4444');

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = gradient;
      ctx.beginPath();

      const sliceWidth = w / bufferLength;
      let x = 0;
      let maxVol = 0;

      for (let i = 0; i < bufferLength; i++) {
        const val = dataArray[i];
        const v = val / 128.0;
        const diff = Math.abs(val - 128);
        if (diff > maxVol) maxVol = diff;

        const y = (v * h) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }

      ctx.lineTo(w, h / 2);
      ctx.stroke();

      // Breathing detection logic
      if (liveCaption) {
        const timeSinceSpeech = lastSpeechTime === 0 ? 0 : Date.now() - lastSpeechTime;
        if (maxVol > 12) { // Ambient/breathing noise threshold
          if (lastSpeechTime === 0 || timeSinceSpeech > 1200) {
            liveCaption.textContent = "Breathing...";
          }
        } else if (lastSpeechTime === 0 || timeSinceSpeech > 2000) {
          liveCaption.textContent = "Listening...";
        }
      }

      // Subtle glow effect
      ctx.shadowColor = '#00d4ff';
      ctx.shadowBlur = 8;
    }

    draw();
  }

  async function sendRecordedAudio(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');

    try {
      const response = await fetch('/predict-live', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.status === 'success') {
        submitToRender(data.result);
      } else {
        alert('Error: ' + (data.error || 'Prediction failed'));
        resetMicUI();
      }
    } catch (err) {
      console.error('Live prediction error:', err);
      alert('Error sending recording to server.');
      resetMicUI();
    }
  }

  function resetMicUI() {
    if (micBtn) {
      micBtn.disabled = false;
      micBtn.classList.remove('recording');
    }
    if (micStatus) micStatus.textContent = 'Click to start recording';
    if (countdownEl) countdownEl.style.display = 'none';
    if (micLoader) micLoader.style.display = 'none';
    if (liveWaveformBox) liveWaveformBox.style.display = 'none';
    if (liveCaption) {
      liveCaption.style.display = 'none';
      liveCaption.textContent = '';
    }
  }


  // ═══════════════════════════════════════════════
  //  ANIMATED COUNTERS (Intersection Observer)
  // ═══════════════════════════════════════════════

  const counters = document.querySelectorAll('.stat-number');
  const observerOptions = { threshold: 0.5 };

  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.dataset.target) || 0;
        const suffix = el.dataset.suffix || '';
        animateCounter(el, 0, target, 1500, suffix);
        counterObserver.unobserve(el);
      }
    });
  }, observerOptions);

  counters.forEach(c => counterObserver.observe(c));

  function animateCounter(el, start, end, duration, suffix) {
    const range = end - start;
    const startTime = performance.now();
    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(start + range * eased) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }


  // ═══════════════════════════════════════════════
  //  CONFIDENCE GAUGE (Result Page)
  // ═══════════════════════════════════════════════

  const gaugeFill = document.getElementById('gauge-fill');
  const gaugePct = document.getElementById('gauge-pct');
  if (gaugeFill && gaugePct) {
    const confidence = parseFloat(gaugePct.dataset.value) || 0;
    const circumference = 565;
    const offset = circumference - (circumference * confidence / 100);

    let color = '#ef4444';
    if (confidence >= 80) color = '#22c55e';
    else if (confidence >= 60) color = '#eab308';

    setTimeout(() => {
      gaugeFill.style.strokeDashoffset = offset;
      gaugeFill.style.stroke = color;
      animateCounter(gaugePct, 0, Math.round(confidence), 1500, '%');
    }, 300);
  }


  // ═══════════════════════════════════════════════
  //  CHART.JS TOP 3 BAR CHART (Result Page)
  // ═══════════════════════════════════════════════

  const chartCanvas = document.getElementById('top3-chart');
  if (chartCanvas && window.chartData) {
    const ctx = chartCanvas.getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: window.chartData.labels,
        datasets: [{
          label: 'Probability (%)',
          data: window.chartData.values,
          backgroundColor: ['rgba(0,212,255,0.6)', 'rgba(123,47,255,0.6)', 'rgba(100,116,139,0.4)'],
          borderColor: ['#00d4ff', '#7b2fff', '#64748b'],
          borderWidth: 2,
          borderRadius: 8,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            max: 100,
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            ticks: { color: '#f1f5f9', font: { weight: 'bold' } },
            grid: { display: false }
          }
        },
        animation: { duration: 1500, easing: 'easeOutQuart' }
      }
    });
  }

});
