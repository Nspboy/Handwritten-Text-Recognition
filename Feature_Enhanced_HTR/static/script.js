document.addEventListener('DOMContentLoaded', () => {
    // Basic elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const resetBtn = document.getElementById('resetBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Tab buttons & contents
    const tabUploadBtn = document.getElementById('tabUploadBtn');
    const tabDrawBtn = document.getElementById('tabDrawBtn');
    const uploadTabContent = document.getElementById('uploadTabContent');
    const drawTabContent = document.getElementById('drawTabContent');

    // Canvas elements
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const clearCanvasBtn = document.getElementById('clearCanvasBtn');
    const analyzeCanvasBtn = document.getElementById('analyzeCanvasBtn');

    // Settings elements
    const nlpMethodSelect = document.getElementById('nlpMethodSelect');
    const nlpHelpText = document.getElementById('nlpHelpText');

    // Pipeline steps
    const steps = {
        prep: document.getElementById('step-prep'),
        cnn: document.getElementById('step-cnn'),
        ctc: document.getElementById('step-ctc'),
        nlp: document.getElementById('step-nlp')
    };

    let selectedFile = null;
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // --- Tab Switching ---
    tabUploadBtn.addEventListener('click', () => {
        tabUploadBtn.classList.add('active');
        tabDrawBtn.classList.remove('active');
        uploadTabContent.classList.remove('hidden');
        drawTabContent.classList.add('hidden');
        resetUI();
    });

    tabDrawBtn.addEventListener('click', () => {
        tabDrawBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        drawTabContent.classList.remove('hidden');
        uploadTabContent.classList.add('hidden');
        resetUI();
        resizeCanvas();
    });

    // --- Drawing Board Setup ---
    function initCanvas() {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.strokeStyle = '#000000';
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.lineWidth = parseInt(brushSizeInput.value);
    }

    function resizeCanvas() {
        // Adjust canvas visual scale but preserve drawing buffer width/height
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        initCanvas();
    }

    // Brush size controller
    brushSizeInput.addEventListener('input', (e) => {
        brushSizeVal.textContent = e.target.value + 'px';
        ctx.lineWidth = parseInt(e.target.value);
    });

    // Clear Canvas
    clearCanvasBtn.addEventListener('click', initCanvas);

    // Get exact offsets for mouse and touch
    function getOffset(e) {
        const rect = canvas.getBoundingClientRect();
        if (e.touches && e.touches.length > 0) {
            return {
                x: e.touches[0].clientX - rect.left,
                y: e.touches[0].clientY - rect.top
            };
        } else {
            return {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        }
    }

    // Start drawing
    function startDrawing(e) {
        isDrawing = true;
        const coords = getOffset(e);
        lastX = coords.x;
        lastY = coords.y;
    }

    // Draw lines
    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getOffset(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        
        lastX = coords.x;
        lastY = coords.y;
    }

    // Stop drawing
    function stopDrawing() {
        isDrawing = false;
    }

    // Canvas listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    // Canvas Submission
    analyzeCanvasBtn.addEventListener('click', () => {
        canvas.toBlob((blob) => {
            const file = new File([blob], "canvas_drawing.png", { type: "image/png" });
            selectedFile = file;

            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                drawTabContent.classList.add('hidden');
                previewArea.classList.remove('hidden');
                
                // Automatically run recognition
                analyzeBtn.click();
            };
            reader.readAsDataURL(file);
        }, 'image/png');
    });

    // --- NLP Configuration Helper Text ---
    nlpMethodSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'simple') {
            nlpHelpText.textContent = 'Standardizes whitespaces and removes trailing syntax anomalies.';
        } else if (val === 'symspell') {
            nlpHelpText.textContent = 'Applies dictionary lookup to correct typos at character level (+5-10% word accuracy).';
        } else if (val === 'transformer') {
            nlpHelpText.textContent = 'Runs google/flan-t5-small seq2seq correction for contextual and grammatical fixes.';
        }
    });

    // --- Upload Drag and Drop ---
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener('click', resetUI);

    // --- Main Inference Orchestration ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="ph ph-spinner spin"></i> Processing...';
        
        resetSteps();

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('nlp_method', nlpMethodSelect.value);

        try {
            // Trigger first step preprocessing visually
            activateStep('prep');
            
            const response = await fetch('/api/recognize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const data = await response.json();
            
            // Step 1: Preprocessing completed
            completeStep('prep');
            document.getElementById('processedPreview').src = data.processed_image;
            document.getElementById('prepResult').classList.remove('hidden');

            // Step 2: CNN & BiLSTM & HRNN
            activateStep('cnn');
            document.getElementById('modelProgress').classList.remove('hidden');
            await animateProgress();
            completeStep('cnn');

            // Step 3: CTC
            activateStep('ctc');
            await sleep(400);
            completeStep('ctc');
            document.getElementById('rawTextValue').textContent = data.raw_text || '[No text detected]';
            document.getElementById('rawTextResult').classList.remove('hidden');

            // Step 4: NLP
            activateStep('nlp');
            await sleep(400);
            completeStep('nlp');
            document.getElementById('finalTextValue').textContent = data.corrected_text || '[No text detected]';
            document.getElementById('digitizedPreview').src = data.digitized_image || '';
            document.getElementById('timeValue').textContent = data.inference_time;
            document.getElementById('finalTextResult').classList.remove('hidden');

        } catch (error) {
            console.error('Error during recognition:', error);
            alert('An error occurred during processing. Please verify that the Flask server is running and python libraries are functional.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="ph ph-magic-wand"></i> Analyze Text';
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadArea.classList.add('hidden');
            previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        
        if (tabUploadBtn.classList.contains('active')) {
            uploadArea.classList.remove('hidden');
        } else {
            drawTabContent.classList.remove('hidden');
            initCanvas();
        }
        
        previewArea.classList.add('hidden');
        imagePreview.src = '';
        resetSteps();
    }

    function resetSteps() {
        Object.values(steps).forEach(step => {
            step.classList.remove('active', 'completed');
        });
        document.getElementById('prepResult').classList.add('hidden');
        document.getElementById('modelProgress').classList.add('hidden');
        document.getElementById('rawTextResult').classList.add('hidden');
        document.getElementById('finalTextResult').classList.add('hidden');
        document.querySelector('.progress-fill').style.width = '0%';
    }

    function activateStep(stepName) {
        steps[stepName].classList.add('active');
        steps[stepName].classList.remove('completed');
    }

    function completeStep(stepName) {
        steps[stepName].classList.remove('active');
        steps[stepName].classList.add('completed');
    }

    async function animateProgress() {
        const fill = document.querySelector('.progress-fill');
        return new Promise(resolve => {
            let width = 0;
            const interval = setInterval(() => {
                width += Math.random() * 20;
                if (width >= 100) {
                    width = 100;
                    fill.style.width = width + '%';
                    clearInterval(interval);
                    setTimeout(resolve, 150);
                } else {
                    fill.style.width = width + '%';
                }
            }, 80);
        });
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // --- Preset Samples Loader ---
    async function loadSamples() {
        const grid = document.getElementById('samplesGrid');
        try {
            const response = await fetch('/api/samples');
            const sampleNames = await response.json();
            
            if (sampleNames.length === 0) {
                grid.innerHTML = '<p class="samples-loading">No sample images found.</p>';
                return;
            }
            
            grid.innerHTML = '';
            for (const name of sampleNames) {
                const card = document.createElement('div');
                card.className = 'sample-card';
                card.title = `Test with preset ${name}`;
                
                // Fetch specific sample image data
                const imgRes = await fetch(`/api/sample/${name}`);
                const imgData = await imgRes.json();
                
                card.innerHTML = `<img src="${imgData.image}" alt="${name}">`;
                card.addEventListener('click', () => {
                    selectSample(imgData.image, name);
                });
                grid.appendChild(card);
            }
        } catch (error) {
            console.error('Error loading samples:', error);
            grid.innerHTML = '<p class="samples-loading">Error loading samples.</p>';
        }
    }

    function selectSample(base64Image, name) {
        const blob = dataURItoBlob(base64Image);
        selectedFile = new File([blob], name, { type: 'image/png' });
        
        imagePreview.src = base64Image;
        uploadArea.classList.add('hidden');
        previewArea.classList.remove('hidden');
        
        // Auto trigger analysis
        analyzeBtn.click();
    }

    function dataURItoBlob(dataURI) {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], {type: mimeString});
    }

    // Initialize Canvas on load
    initCanvas();
    // Load presets
    loadSamples();
});
