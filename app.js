// RetinaFace Multi-Face Detection Application
// Uses ONNX Runtime Web for client-side inference

class RetinaFaceDetector {
    constructor() {
        this.session = null;
        this.modelLoaded = false;
        this.isDetecting = false;
        this.videoElement = null;
        this.canvasElement = null;
        this.ctx = null;
        this.animationFrameId = null;

        // Detection parameters
        this.confidenceThreshold = 0.50;
        this.nmsThreshold = 0.40;

        // Performance tracking
        this.lastFrameTime = 0;
        this.fps = 0;
        this.inferenceTime = 0;

        // Model input resolution (RetinaFace export is square 640x640)
        this.inputWidth = 640;
        this.inputHeight = 640;
        this.variances = [0.1, 0.2];
        this.priors = null;
        this.priorInputKey = '';
        this.retinaCfg = {
            minSizes: [[16, 32], [64, 128], [256, 512]],
            steps: [8, 16, 32],
            clip: false
        };
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupONNXRuntime();
    }

    setupONNXRuntime() {
        // Configure ONNX Runtime for optimal performance
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        ort.env.wasm.simd = true;
        ort.env.logLevel = 'warning';

        console.log('ONNX Runtime initialized');
        console.log('Available execution providers:', ort.env.wasm);
    }

    setupEventListeners() {
        // Upload area
        const uploadArea = document.getElementById('uploadArea');
        const videoInput = document.getElementById('videoInput');

        uploadArea.addEventListener('click', () => videoInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                this.loadVideo(file);
            }
        });

        videoInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.loadVideo(file);
            }
        });

        // Model loading
        document.getElementById('loadModelBtn').addEventListener('click', () => {
            this.loadModel();
        });

        // Video controls
        document.getElementById('playBtn').addEventListener('click', () => {
            if (this.videoElement) this.videoElement.play();
        });

        document.getElementById('pauseBtn').addEventListener('click', () => {
            if (this.videoElement) this.videoElement.pause();
        });

        document.getElementById('toggleDetectionBtn').addEventListener('click', () => {
            this.toggleDetection();
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.reset();
        });

        // Sliders
        const confidenceSlider = document.getElementById('confidenceSlider');
        const nmsSlider = document.getElementById('nmsSlider');

        confidenceSlider.addEventListener('input', (e) => {
            this.confidenceThreshold = parseFloat(e.target.value);
            document.getElementById('confidenceValue').textContent = this.confidenceThreshold.toFixed(2);
        });

        nmsSlider.addEventListener('input', (e) => {
            this.nmsThreshold = parseFloat(e.target.value);
            document.getElementById('nmsValue').textContent = this.nmsThreshold.toFixed(2);
        });
    }

    async loadModel() {
        try {
            this.showLoading('Loading RetinaFace model...');
            this.updateModelStatus('loading', 'Loading model...');

            // Load the RetinaFace ONNX model
            const modelPath = 'retinaface.onnx'; // User needs to provide this file

            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.modelLoaded = true;
            this.updateModelStatus('loaded', 'Model loaded successfully');
            this.hideLoading();

            // Show detection controls
            document.getElementById('detectionControls').style.display = 'block';

            console.log('RetinaFace model loaded successfully');
            console.log('Model inputs:', this.session.inputNames);
            console.log('Model outputs:', this.session.outputNames);

        } catch (error) {
            console.error('Error loading model:', error);
            this.updateModelStatus('error', 'Failed to load model');
            this.hideLoading();
            alert('Error loading model. Please ensure retinaface.onnx is in the same directory as this HTML file.\n\nError: ' + error.message);
        }
    }

    loadVideo(file) {
        const videoPlayer = document.getElementById('videoPlayer');
        const videoSection = document.getElementById('videoSection');

        const url = URL.createObjectURL(file);
        videoPlayer.src = url;
        videoSection.style.display = 'block';

        this.videoElement = videoPlayer;
        this.canvasElement = document.getElementById('overlayCanvas');
        this.ctx = this.canvasElement.getContext('2d');

        videoPlayer.addEventListener('loadedmetadata', () => {
            // Set canvas size to match video
            this.canvasElement.width = videoPlayer.videoWidth;
            this.canvasElement.height = videoPlayer.videoHeight;
            console.log(`Video loaded: ${videoPlayer.videoWidth}x${videoPlayer.videoHeight}`);
        });
    }

    toggleDetection() {
        if (!this.modelLoaded) {
            alert('Please load the RetinaFace model first.');
            return;
        }

        if (!this.videoElement) {
            alert('Please upload a video first.');
            return;
        }

        this.isDetecting = !this.isDetecting;
        const btn = document.getElementById('toggleDetectionBtn');

        if (this.isDetecting) {
            btn.textContent = 'Stop Detection';
            btn.classList.add('active');
            this.startDetection();
        } else {
            btn.textContent = 'Start Detection';
            btn.classList.remove('active');
            this.stopDetection();
        }
    }

    startDetection() {
        const detectFrame = async () => {
            if (!this.isDetecting) return;

            const startTime = performance.now();

            // Extract current frame
            const frameData = this.extractFrame();

            if (frameData) {
                // Run inference
                const faces = await this.detectFaces(frameData);

                // Draw bounding boxes
                this.drawBoundingBoxes(faces);

                // Update stats
                this.inferenceTime = performance.now() - startTime;
                this.updateStats(faces.length);
            }

            // Continue detection loop
            this.animationFrameId = requestAnimationFrame(detectFrame);
        };

        detectFrame();
    }

    stopDetection() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        // Clear canvas
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        }
    }

    extractFrame() {
        if (!this.videoElement || this.videoElement.paused || this.videoElement.ended) {
            return null;
        }

        // Create temporary canvas for frame extraction
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.inputWidth;
        tempCanvas.height = this.inputHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw video frame to canvas (resized to model input size)
        tempCtx.drawImage(this.videoElement, 0, 0, this.inputWidth, this.inputHeight);

        // Get image data
        const imageData = tempCtx.getImageData(0, 0, this.inputWidth, this.inputHeight);
        return imageData;
    }

    async detectFaces(imageData) {
        try {
            // Preprocess image data for RetinaFace
            const inputTensor = this.preprocessImage(imageData);

            // Run inference
            const feeds = {};
            feeds[this.session.inputNames[0]] = inputTensor;

            const results = await this.session.run(feeds);

            // Post-process results
            const faces = this.postprocessResults(results);

            return faces;
        } catch (error) {
            console.error('Error during inference:', error);
            return [];
        }
    }

    preprocessImage(imageData) {
        // Convert ImageData to tensor format expected by RetinaFace
        // RetinaFace expects: [batch, channels, height, width] with RGB values normalized

        const { data } = imageData;
        const width = this.inputWidth;
        const height = this.inputHeight;

        // Create Float32Array for the tensor (BGR, channel-first, height, width)
        const tensorData = new Float32Array(1 * 3 * height * width);

        // Mean subtraction used by the training pipeline (BGR order)
        const mean = [104, 117, 123];

        // Convert RGBA to BGR (channel-first) and subtract mean
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixelIndex = (y * width + x) * 4;

                // B channel
                tensorData[0 * height * width + y * width + x] =
                    data[pixelIndex + 2] - mean[0];

                // G channel
                tensorData[1 * height * width + y * width + x] =
                    data[pixelIndex + 1] - mean[1];

                // R channel
                tensorData[2 * height * width + y * width + x] =
                    data[pixelIndex] - mean[2];
            }
        }

        // Create ONNX tensor
        const tensor = new ort.Tensor('float32', tensorData, [1, 3, height, width]);
        return tensor;
    }

    postprocessResults(results) {
        // RetinaFace raw outputs: loc + conf (needs decoding with priors)
        // If not found, fall back to the previous generic parsing
        const outputNames = this.session.outputNames || [];
        let locTensor = null;
        let confTensor = null;

        for (const name of outputNames) {
            const tensor = results[name];
            if (!tensor) continue;
            const shape = tensor.dims;
            if (shape.length === 3 && shape[2] === 4 && !locTensor) {
                locTensor = tensor;
            } else if (shape.length === 3 && shape[2] === 2 && !confTensor) {
                confTensor = tensor;
            }
        }

        if (locTensor && confTensor) {
            const faces = this.decodeRetinaFace(locTensor, confTensor);
            if (faces.length === 0) {
                console.log('[RetinaFace] No faces after decode. First conf:', confTensor.data ? Math.max(...confTensor.data) : 'n/a');
            }
            return faces;
        }

        // Fallback: attempt to interpret outputs as already-decoded boxes
        return this.parseGenericOutputs(results);
    }

    applyNMS(faces) {
        // Non-Maximum Suppression to remove overlapping detections
        if (faces.length === 0) return [];

        // Sort by score (descending)
        faces.sort((a, b) => b.score - a.score);

        const keep = [];
        const suppressed = new Set();

        for (let i = 0; i < faces.length; i++) {
            if (suppressed.has(i)) continue;

            keep.push(faces[i]);

            const bbox1 = faces[i].bbox;

            for (let j = i + 1; j < faces.length; j++) {
                if (suppressed.has(j)) continue;

                const bbox2 = faces[j].bbox;
                const iou = this.calculateIOU(bbox1, bbox2);

                if (iou > this.nmsThreshold) {
                    suppressed.add(j);
                }
            }
        }

        return keep;
    }

    calculateIOU(bbox1, bbox2) {
        // Calculate Intersection over Union
        const [x1_1, y1_1, x2_1, y2_1] = bbox1;
        const [x1_2, y1_2, x2_2, y2_2] = bbox2;

        const x1_i = Math.max(x1_1, x1_2);
        const y1_i = Math.max(y1_1, y1_2);
        const x2_i = Math.min(x2_1, x2_2);
        const y2_i = Math.min(y2_1, y2_2);

        const intersection = Math.max(0, x2_i - x1_i) * Math.max(0, y2_i - y1_i);

        const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        const union = area1 + area2 - intersection;

        return intersection / union;
    }

    generatePriors() {
        const { minSizes, steps, clip } = this.retinaCfg;
        const featureMaps = steps.map(step => [
            Math.ceil(this.inputHeight / step),
            Math.ceil(this.inputWidth / step)
        ]);

        const priors = [];
        featureMaps.forEach((fmap, k) => {
            const minSizesForLevel = minSizes[k];
            for (let i = 0; i < fmap[0]; i++) {
                for (let j = 0; j < fmap[1]; j++) {
                    minSizesForLevel.forEach(minSize => {
                        const sKx = minSize / this.inputWidth;
                        const sKy = minSize / this.inputHeight;
                        const cx = (j + 0.5) * steps[k] / this.inputWidth;
                        const cy = (i + 0.5) * steps[k] / this.inputHeight;
                        priors.push(cx, cy, sKx, sKy);
                    });
                }
            }
        });

        if (clip) {
            for (let i = 0; i < priors.length; i++) {
                priors[i] = Math.min(1, Math.max(0, priors[i]));
            }
        }

        this.priorInputKey = `${this.inputWidth}x${this.inputHeight}`;
        this.priors = new Float32Array(priors);
        return this.priors;
    }

    decodeRetinaFace(locTensor, confTensor) {
        // Ensure priors match current input size
        if (!this.priors || this.priorInputKey !== `${this.inputWidth}x${this.inputHeight}`) {
            this.generatePriors();
        }

        const loc = locTensor.data;
        const scores = confTensor.data;
        const numBoxes = locTensor.dims[1];
        const priors = this.priors;
        const faces = [];

        for (let i = 0; i < numBoxes; i++) {
            const score = scores[i * 2 + 1]; // class 1 = face
            if (score < this.confidenceThreshold) continue;

            const priorIndex = i * 4;
            const cx = priors[priorIndex];
            const cy = priors[priorIndex + 1];
            const w = priors[priorIndex + 2];
            const h = priors[priorIndex + 3];

            const dx = loc[i * 4];
            const dy = loc[i * 4 + 1];
            const dw = loc[i * 4 + 2];
            const dh = loc[i * 4 + 3];

            // Decode (mirrors utils/box_utils.py decode)
            const decodedCx = cx + dx * this.variances[0] * w;
            const decodedCy = cy + dy * this.variances[0] * h;
            const decodedW = w * Math.exp(dw * this.variances[1]);
            const decodedH = h * Math.exp(dh * this.variances[1]);

            let x1 = (decodedCx - decodedW / 2) * this.inputWidth;
            let y1 = (decodedCy - decodedH / 2) * this.inputHeight;
            let x2 = (decodedCx + decodedW / 2) * this.inputWidth;
            let y2 = (decodedCy + decodedH / 2) * this.inputHeight;

            // Clamp to image bounds
            x1 = Math.max(0, Math.min(this.inputWidth, x1));
            y1 = Math.max(0, Math.min(this.inputHeight, y1));
            x2 = Math.max(0, Math.min(this.inputWidth, x2));
            y2 = Math.max(0, Math.min(this.inputHeight, y2));

            if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;

            faces.push({
                bbox: [x1, y1, x2, y2],
                score
            });
        }

        return this.applyNMS(faces);
    }

    parseGenericOutputs(results) {
        const faces = [];
        const outputNames = this.session.outputNames || [];
        let bboxes, scores;

        for (const name of outputNames) {
            const tensor = results[name];
            if (!tensor) continue;
            const shape = tensor.dims;
            if (shape.length === 3 && shape[2] === 4) {
                bboxes = tensor;
            } else if (shape.length === 3 && (shape[2] === 2 || shape[2] === 1)) {
                scores = tensor;
            }
        }

        if (!bboxes || !scores) {
            console.warn('Could not identify bbox/score outputs. Available outputs:', outputNames);
            return faces;
        }

        const bboxData = bboxes.data;
        const scoreData = scores.data;
        const numBoxes = bboxes.dims[1];

        for (let i = 0; i < numBoxes; i++) {
            const score = scores.dims[2] === 2 ? scoreData[i * 2 + 1] : scoreData[i];
            if (score >= this.confidenceThreshold) {
                faces.push({
                    bbox: [
                        bboxData[i * 4],
                        bboxData[i * 4 + 1],
                        bboxData[i * 4 + 2],
                        bboxData[i * 4 + 3]
                    ],
                    score
                });
            }
        }

        return this.applyNMS(faces);
    }

    drawBoundingBoxes(faces) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

        if (faces.length === 0) return;

        // Scale factors (from model input size to video display size)
        const scaleX = this.videoElement.videoWidth / this.inputWidth;
        const scaleY = this.videoElement.videoHeight / this.inputHeight;

        // Draw each face bounding box
        faces.forEach((face, index) => {
            const [x1, y1, x2, y2] = face.bbox;

            // Scale coordinates to video size
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;

            const width = scaledX2 - scaledX1;
            const height = scaledY2 - scaledY1;

            // Draw bounding box
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(scaledX1, scaledY1, width, height);

            // Draw label background
            const label = `Face ${index + 1}: ${(face.score * 100).toFixed(1)}%`;
            this.ctx.font = '16px Arial';
            const textMetrics = this.ctx.measureText(label);
            const textHeight = 20;

            this.ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
            this.ctx.fillRect(scaledX1, scaledY1 - textHeight - 4, textMetrics.width + 8, textHeight + 4);

            // Draw label text
            this.ctx.fillStyle = '#000000';
            this.ctx.fillText(label, scaledX1 + 4, scaledY1 - 8);
        });
    }

    updateStats(faceCount) {
        // Update face count
        document.getElementById('faceCount').textContent = faceCount;

        // Update FPS
        const currentTime = performance.now();
        if (this.lastFrameTime > 0) {
            const delta = currentTime - this.lastFrameTime;
            this.fps = Math.round(1000 / delta);
            document.getElementById('fpsValue').textContent = this.fps;
        }
        this.lastFrameTime = currentTime;

        // Update inference time
        document.getElementById('inferenceTime').textContent = `${this.inferenceTime.toFixed(1)}ms`;
    }

    updateModelStatus(status, text) {
        const statusElement = document.getElementById('modelStatus');
        const indicator = statusElement.querySelector('.status-indicator');
        const textElement = statusElement.querySelector('.status-text');

        indicator.className = 'status-indicator';
        indicator.classList.add(status);
        textElement.textContent = text;
    }

    showLoading(text) {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        loadingText.textContent = text;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    reset() {
        this.stopDetection();

        if (this.videoElement) {
            this.videoElement.pause();
            this.videoElement.currentTime = 0;
        }

        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        }

        // Reset stats
        document.getElementById('faceCount').textContent = '0';
        document.getElementById('fpsValue').textContent = '0';
        document.getElementById('inferenceTime').textContent = '0ms';

        const btn = document.getElementById('toggleDetectionBtn');
        btn.textContent = 'Start Detection';
        btn.classList.remove('active');

        this.isDetecting = false;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const detector = new RetinaFaceDetector();
    console.log('RetinaFace Multi-Face Detection App initialized');
});
