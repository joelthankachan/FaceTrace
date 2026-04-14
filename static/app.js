// FaceTrace — app.js
// Handles file upload, webcam capture, Flask API calls and result rendering

let selectedImage = null;
let webcamStream  = null;

const dropZone = document.getElementById('drop-zone');


// load known identity chips on page load
window.addEventListener('load', async () => {
    try {
        const res  = await fetch('/classes');
        const data = await res.json();
        const container = document.getElementById('identity-chips');
        data.classes.forEach(name => {
            const chip = document.createElement('div');
            chip.className   = 'chip';
            chip.textContent = name.replace(/_/g, ' ');
            container.appendChild(chip);
        });
    } catch {
        console.log('Could not load identity list.');
    }
});


// ── File upload ────────────────────────────────────────────────────────────

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) loadImageFile(e.dataTransfer.files[0]);
});

// the file input covers the full drop zone (position: absolute, inset: 0)
// clicking the zone opens the browser natively — no extra .click() call needed
document.getElementById('file-input').addEventListener('change', e => {
    if (e.target.files[0]) loadImageFile(e.target.files[0]);
});

function loadImageFile(file) {
    const reader = new FileReader();
    reader.onload = e => {
        selectedImage = e.target.result;
        document.getElementById('preview-img').src           = selectedImage;
        document.getElementById('preview-box').style.display = 'block';
        dropZone.style.display = 'none';
        document.getElementById('predict-btn').disabled = false;
        showPlaceholder();
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedImage = null;
    document.getElementById('preview-box').style.display = 'none';
    document.getElementById('preview-img').src  = '';
    document.getElementById('file-input').value = '';
    dropZone.style.display = 'block';
    document.getElementById('predict-btn').disabled = true;
    showPlaceholder();
}


// ── Webcam ─────────────────────────────────────────────────────────────────

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        document.getElementById('webcam-video').srcObject = webcamStream;
        document.getElementById('webcam-box').style.display = 'block';
        document.getElementById('start-btn').disabled   = true;
        document.getElementById('capture-btn').disabled = false;
        document.getElementById('stop-btn').disabled    = false;
    } catch (err) {
        alert('Could not access camera: ' + err.message);
    }
}

function capturePhoto() {
    const video  = document.getElementById('webcam-video');
    const canvas = document.getElementById('capture-canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    selectedImage = canvas.toDataURL('image/jpeg', 0.92);
    document.getElementById('webcam-preview-img').src           = selectedImage;
    document.getElementById('webcam-preview-box').style.display = 'block';
    document.getElementById('predict-btn').disabled = false;
    showPlaceholder();
}

function clearWebcamCapture() {
    selectedImage = null;
    document.getElementById('webcam-preview-box').style.display = 'none';
    document.getElementById('webcam-preview-img').src = '';
    document.getElementById('predict-btn').disabled   = true;
    showPlaceholder();
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    document.getElementById('webcam-box').style.display         = 'none';
    document.getElementById('webcam-preview-box').style.display = 'none';
    document.getElementById('start-btn').disabled   = false;
    document.getElementById('capture-btn').disabled = true;
    document.getElementById('stop-btn').disabled    = true;
    selectedImage = null;
    document.getElementById('predict-btn').disabled = true;
    showPlaceholder();
}


// ── Reset ──────────────────────────────────────────────────────────────────

function resetAll() {
    clearImage();
    stopWebcam();
    dropZone.style.display = 'block';
    showPlaceholder();
}


// ── Predict ────────────────────────────────────────────────────────────────

async function predict() {
    if (!selectedImage) return;

    document.getElementById('placeholder').style.display    = 'none';
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('spinner-box').style.display    = 'block';
    document.getElementById('predict-btn').disabled         = true;

    try {
        const res  = await fetch('/predict', {
            method  : 'POST',
            headers : { 'Content-Type': 'application/json' },
            body    : JSON.stringify({ image: selectedImage })
        });
        const data = await res.json();

        document.getElementById('spinner-box').style.display = 'none';
        document.getElementById('predict-btn').disabled      = false;

        if (data.error) {
            alert('Error: ' + data.error);
            showPlaceholder();
        } else {
            showResult(data);
        }

    } catch (err) {
        document.getElementById('spinner-box').style.display = 'none';
        document.getElementById('predict-btn').disabled      = false;
        alert('Could not reach server. Make sure server.py is running.');
        showPlaceholder();
    }
}


// ── Show result ────────────────────────────────────────────────────────────

function showResult(data) {
    const isKnown = data.predicted_name !== 'Unknown';
    const nameEl  = document.getElementById('predicted-name');

    nameEl.textContent = isKnown ? data.predicted_name.replace(/_/g, ' ') : 'Unknown Person';
    nameEl.className   = 'predicted-name ' + (isKnown ? 'known' : 'unknown');

    const badgeEl     = document.getElementById('face-badge');
    badgeEl.textContent = data.face_detected
        ? '✓ Face detected in image'
        : '⚠ No face detected — full image used';
    badgeEl.className = 'face-badge ' + (data.face_detected ? 'detected' : 'not-detected');

    document.getElementById('confidence-value').textContent = data.confidence + '%';

    const barsContainer = document.getElementById('prob-bars');
    barsContainer.innerHTML = '';

    const sorted  = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);
    const topName = sorted[0][0];

    sorted.forEach(([name, pct]) => {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-label-row">
                <span class="name">${name.replace(/_/g, ' ')}</span>
                <span class="pct">${pct}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill ${name === topName ? 'top-bar' : ''}"
                     style="width: ${pct}%"></div>
            </div>`;
        barsContainer.appendChild(row);
    });

    document.getElementById('result-section').style.display = 'block';
}


// ── Helper ─────────────────────────────────────────────────────────────────

function showPlaceholder() {
    document.getElementById('placeholder').style.display    = 'block';
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('spinner-box').style.display    = 'none';
}
