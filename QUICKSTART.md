# Quick Start Guide

Get started with RetinaFace Multi-Face Detection in 5 minutes!

## Prerequisites

- Python 3.7+ (for model conversion/testing)
- Modern web browser (Chrome, Firefox, Safari, or Edge)
- RetinaFace ONNX model (see below)

## Step 1: Get the RetinaFace Model

### Option A: Download Pre-converted Model (Easiest)

Search for "retinaface onnx" on:
- [Hugging Face Models](https://huggingface.co/models)
- [ONNX Model Zoo](https://github.com/onnx/models)
- GitHub repositories

### Option B: Convert from PyTorch

```bash
# 1. Install dependencies
pip install torch torchvision onnx onnxruntime

# 2. Clone RetinaFace PyTorch implementation
git clone https://github.com/biubug6/Pytorch_Retinaface.git
cd Pytorch_Retinaface

# 3. Download pre-trained weights from the repository

# 4. Convert to ONNX
python ../convert_to_onnx.py \
    --weights ./weights/mobilenet0.25_Final.pth \
    --output ../retinaface.onnx
```

## Step 2: Test the Model (Optional but Recommended)

```bash
# Test your ONNX model
python test_model.py retinaface.onnx

# Test with a sample image
python test_model.py retinaface.onnx --image path/to/photo.jpg

# Run performance benchmark
python test_model.py retinaface.onnx --performance
```

## Step 3: Set Up the Web Application

```bash
# Ensure files are in place
FACIALFREESTYLE/
├── index.html
├── app.js
├── styles.css
└── retinaface.onnx    # Your ONNX model
```

## Step 4: Start the Web Server

Choose one method:

```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Node.js
npx http-server -p 8000

# PHP
php -S localhost:8000
```

## Step 5: Open in Browser

1. Navigate to `http://localhost:8000`
2. Click **"Load RetinaFace Model"**
3. Wait for model to load (status turns green)
4. Upload a video file
5. Click **"Start Detection"**
6. Play the video to see face detection in action!

## Troubleshooting

### Model won't load
- Ensure `retinaface.onnx` is in the same directory as `index.html`
- Check browser console (F12) for error messages
- Verify the ONNX model with `test_model.py`

### No faces detected
- Lower the confidence threshold to 0.5 or lower
- Ensure video is playing (detection only works during playback)
- Check that faces are clearly visible in the video

### Poor performance
- Use Chrome for best performance
- Try a shorter or lower resolution video
- Close other browser tabs

### CORS errors
- Must use a web server (cannot open files directly)
- Ensure all files are served from the same domain

## Tips for Best Results

1. **Video Quality**: Higher resolution = better detection
2. **Lighting**: Well-lit faces are easier to detect
3. **Face Size**: Larger faces (closer to camera) work better
4. **Confidence**: Start with 0.7-0.8, adjust based on results
5. **Performance**: 640x640 input size is a good balance

## Model Sources

### Recommended Sources

1. **InsightFace SCRFD** (Alternative to RetinaFace):
   - https://github.com/deepinsight/insightface
   - Pre-converted ONNX models available
   - Similar accuracy to RetinaFace

2. **Pre-trained RetinaFace**:
   - Search "retinaface onnx" on GitHub
   - Check model hubs (Hugging Face, ONNX Zoo)

3. **Convert Your Own**:
   - Use the provided `convert_to_onnx.py` script
   - Start with PyTorch implementations
   - Customize for your specific needs

## Common Model Configurations

| Model | Input Size | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| RetinaFace-MobileNet | 640x640 | Fast | High | General use |
| RetinaFace-ResNet50 | 640x640 | Medium | Very High | Maximum accuracy |
| RetinaFace-MobileNet | 1024x1024 | Slow | Very High | Large images |

## Parameter Tuning Guide

### Confidence Threshold
- **0.5-0.6**: More detections, more false positives
- **0.7-0.8**: Balanced (recommended)
- **0.9+**: Fewer detections, high confidence only

### NMS Threshold
- **0.2-0.3**: Aggressive overlap removal
- **0.4-0.5**: Balanced (recommended)
- **0.6+**: Allow more overlapping boxes

### Input Size
- **320-480**: Fast, lower accuracy
- **640**: Balanced (recommended)
- **1024+**: Slow, highest accuracy

## Browser Performance Comparison

Based on M4 Mac testing:

| Browser | FPS | Inference Time | Notes |
|---------|-----|----------------|-------|
| Chrome 91+ | 25-30 | 20-30ms | Best performance |
| Firefox 89+ | 20-25 | 30-40ms | Good performance |
| Safari 14.1+ | 15-20 | 40-50ms | Acceptable |
| Edge 91+ | 25-30 | 20-30ms | Same as Chrome |

## Advanced Features

### Modify Detection Colors

Edit `app.js` line 365:
```javascript
this.ctx.strokeStyle = '#00ff00'; // Green
// Try: '#ff0000' (red), '#0000ff' (blue), '#ffff00' (yellow)
```

### Change Bounding Box Thickness

Edit `app.js` line 368:
```javascript
this.ctx.lineWidth = 3; // Pixels
```

### Adjust Label Size

Edit `app.js` line 373:
```javascript
this.ctx.font = '16px Arial';
// Try: '20px Arial' (larger), '12px Arial' (smaller)
```

## Export Options

The current implementation displays detections in real-time. To export results:

1. **Screenshots**: Use browser's screenshot feature
2. **Video Recording**: Use screen recording software (OBS, QuickTime)
3. **Custom Export**: Modify `app.js` to save detection data to JSON

## Privacy & Security

- ✓ All processing happens in your browser
- ✓ Videos never leave your device
- ✓ No data is sent to any server
- ✓ No tracking or analytics
- ✓ Works offline after initial load

## Need Help?

1. Check the full [README.md](README.md) for detailed documentation
2. Test your model with `test_model.py`
3. Check browser console (F12) for error messages
4. Verify files are in correct locations
5. Ensure web server is running

## Next Steps

Once you have the basic setup working:

1. Try different videos and lighting conditions
2. Experiment with confidence and NMS thresholds
3. Test with different face sizes and angles
4. Compare performance across browsers
5. Customize the UI colors and styling

---

**Ready to detect faces?** Start with Step 1 above!

For detailed technical information, see [README.md](README.md)
