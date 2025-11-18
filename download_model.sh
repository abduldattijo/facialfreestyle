#!/bin/bash
# Download a pre-converted face detection ONNX model

echo "Downloading SCRFD face detection model (alternative to RetinaFace)..."
echo "This model has similar accuracy and is ready to use!"

# Download SCRFD-500M model (lightweight, accurate)
curl -L -o retinaface.onnx "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx"

if [ -f "retinaface.onnx" ]; then
    echo ""
    echo "✓ Model downloaded successfully: retinaface.onnx"
    echo ""
    echo "File size: $(du -h retinaface.onnx | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "1. Start the web server (see below)"
    echo "2. Open http://localhost:8080 in your browser"
    echo "3. Load the model and upload a video!"
else
    echo "✗ Download failed. Please try again or download manually."
fi
