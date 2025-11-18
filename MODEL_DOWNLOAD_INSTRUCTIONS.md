# How to Get a Face Detection ONNX Model

Since automatic download isn't working, here are manual options:

## Option 1: Download from Hugging Face (Easiest)

1. Visit: https://huggingface.co/models?pipeline_tag=face-detection&library=onnx
2. Look for models like:
   - `onnx-community/retinaface-resnet50`
   - Any model with "face-detection" tag and ONNX format
3. Click on a model, go to "Files and versions"
4. Download the `.onnx` file
5. Rename it to `retinaface.onnx`
6. Move it to this folder: `/Users/muhammaddattijo/Downloads/FACIALFREESTYLE/`

## Option 2: Use ONNX Model Zoo

1. Visit: https://github.com/onnx/models
2. Search for "face detection" models
3. Download any face detection ONNX model
4. Rename to `retinaface.onnx`
5. Place in this folder

## Option 3: Use InsightFace Models (Recommended)

1. Visit: https://github.com/deepinsight/insightface/tree/master/model_zoo
2. Look for SCRFD or RetinaFace models in ONNX format
3. Download the .onnx file
4. Rename to `retinaface.onnx`
5. Place in this folder

## Option 4: Try YuNet (Lightweight, Fast)

1. Visit: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
2. Download `face_detection_yunet_2023mar.onnx`
3. Rename to `retinaface.onnx`
4. Place in this folder

## Option 5: For Testing Only - Skip the Model

You can test the UI without a real model:

1. Create a dummy file: `touch retinaface.onnx`
2. The app will load but detection won't work
3. You can still test the video upload and UI

---

## Once You Have the Model

1. Place `retinaface.onnx` in this folder
2. Start web server: `python3 -m http.server 8080`
3. Open browser: `http://localhost:8080`
4. Click "Load RetinaFace Model"
5. Upload a video and test!

---

## Direct Download Links to Try

Try these direct download commands:

```bash
# YuNet (most likely to work)
curl -L "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o retinaface.onnx

# If that doesn't work, you'll need to download manually from the websites above
```
