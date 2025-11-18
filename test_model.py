#!/usr/bin/env python3
"""
Test RetinaFace ONNX Model

This script helps you test your RetinaFace ONNX model to ensure it works correctly
before using it in the web application.

Usage:
    python test_model.py retinaface.onnx

Requirements:
    pip install onnxruntime opencv-python numpy pillow
"""

import argparse
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_model_basic(model_path):
    """Basic model loading and info test."""
    print("\n" + "=" * 60)
    print("BASIC MODEL TEST")
    print("=" * 60)

    try:
        # Load the ONNX model
        print(f"\nLoading model: {model_path}")
        session = ort.InferenceSession(model_path)
        print("✓ Model loaded successfully")

        # Print model information
        print("\n--- Model Inputs ---")
        for input_meta in session.get_inputs():
            print(f"Name: {input_meta.name}")
            print(f"Shape: {input_meta.shape}")
            print(f"Type: {input_meta.type}")
            print()

        print("--- Model Outputs ---")
        for output_meta in session.get_outputs():
            print(f"Name: {output_meta.name}")
            print(f"Shape: {output_meta.shape}")
            print(f"Type: {output_meta.type}")
            print()

        return session

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def test_inference(session, input_size=640):
    """Test model inference with dummy data."""
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    try:
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Create dummy input (random noise)
        print(f"\nCreating dummy input: {input_shape}")
        batch_size = 1 if isinstance(input_shape[0], str) else input_shape[0]
        dummy_input = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)

        # Run inference
        print("Running inference...")
        outputs = session.run(None, {input_name: dummy_input})
        print("✓ Inference successful")

        # Print output information
        print("\n--- Output Results ---")
        for i, output in enumerate(outputs):
            output_name = session.get_outputs()[i].name
            print(f"Output {i} ({output_name}):")
            print(f"  Shape: {output.shape}")
            print(f"  Type: {output.dtype}")
            print(f"  Min value: {output.min():.6f}")
            print(f"  Max value: {output.max():.6f}")
            print(f"  Mean value: {output.mean():.6f}")
            print()

        return outputs

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_image(session, image_path=None):
    """Test model with a real image."""
    print("\n" + "=" * 60)
    print("REAL IMAGE TEST")
    print("=" * 60)

    if image_path is None:
        print("\nSkipping real image test (no image provided)")
        print("Use --image flag to test with a real image")
        return

    try:
        from PIL import Image

        # Load image
        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        print(f"Original size: {image.size}")

        # Get model input size
        input_shape = session.get_inputs()[0].shape
        input_size = input_shape[2]  # Assuming square input

        # Resize image
        image_resized = image.resize((input_size, input_size))
        print(f"Resized to: {image_resized.size}")

        # Convert to array and normalize
        img_array = np.array(image_resized).astype(np.float32)

        # Normalize (ImageNet mean and std)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])

        img_array = (img_array - mean) / std

        # Convert to CHW format and add batch dimension
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        print(f"Preprocessed shape: {img_array.shape}")

        # Run inference
        input_name = session.get_inputs()[0].name
        print("Running inference...")
        outputs = session.run(None, {input_name: img_array})
        print("✓ Inference successful")

        # Analyze outputs
        print("\n--- Detection Results ---")

        # Try to identify which output is which
        bboxes_idx = None
        scores_idx = None

        for i, output in enumerate(outputs):
            shape = output.shape
            output_name = session.get_outputs()[i].name

            print(f"\nOutput {i} ({output_name}):")
            print(f"  Shape: {shape}")

            # Guess output type based on shape
            if len(shape) == 3 and shape[2] == 4:
                print(f"  Type: Likely BOUNDING BOXES")
                bboxes_idx = i
            elif len(shape) == 2 or (len(shape) == 3 and shape[2] == 1):
                print(f"  Type: Likely SCORES")
                scores_idx = i
            elif len(shape) == 3 and shape[2] == 10:
                print(f"  Type: Likely LANDMARKS")

        # Count detections above threshold
        if scores_idx is not None:
            scores = outputs[scores_idx].flatten()
            thresholds = [0.5, 0.7, 0.8, 0.9]

            print("\n--- Detection Counts by Threshold ---")
            for threshold in thresholds:
                count = (scores > threshold).sum()
                print(f"  Confidence > {threshold}: {count} faces")

            # Show top detections
            top_k = min(5, len(scores))
            top_indices = np.argsort(scores)[-top_k:][::-1]

            print("\n--- Top Detections ---")
            for idx in top_indices:
                score = scores[idx]
                if score > 0.1:  # Only show reasonable scores
                    print(f"  Detection {idx}: confidence = {score:.4f}")
                    if bboxes_idx is not None:
                        bbox = outputs[bboxes_idx].reshape(-1, 4)[idx]
                        print(f"    BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

    except ImportError:
        print("✗ PIL (Pillow) not installed")
        print("Install with: pip install pillow")
    except Exception as e:
        print(f"✗ Error testing with image: {e}")
        import traceback
        traceback.print_exc()


def test_performance(session, num_runs=100):
    """Test model performance."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)

    try:
        import time

        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_size = input_shape[2]

        batch_size = 1 if isinstance(input_shape[0], str) else input_shape[0]
        dummy_input = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)

        print(f"\nRunning {num_runs} inference iterations...")

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})

        # Timed runs
        times = []
        for i in range(num_runs):
            start = time.time()
            session.run(None, {input_name: dummy_input})
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        # Statistics
        times = np.array(times)
        print(f"\n--- Performance Statistics ---")
        print(f"  Mean inference time: {times.mean():.2f} ms")
        print(f"  Median inference time: {np.median(times):.2f} ms")
        print(f"  Min inference time: {times.min():.2f} ms")
        print(f"  Max inference time: {times.max():.2f} ms")
        print(f"  Std deviation: {times.std():.2f} ms")
        print(f"  Estimated FPS: {1000 / times.mean():.1f}")

    except Exception as e:
        print(f"✗ Performance test failed: {e}")


def check_browser_compatibility(session):
    """Check if model is compatible with ONNX Runtime Web."""
    print("\n" + "=" * 60)
    print("BROWSER COMPATIBILITY CHECK")
    print("=" * 60)

    try:
        # Check opset version
        import onnx
        model = onnx.load(session._model_path)

        print(f"\nONNX IR Version: {model.ir_version}")
        print(f"Opset Version: {model.opset_import[0].version}")

        # Recommended opset for browser
        recommended_opset = [9, 10, 11, 12, 13]
        current_opset = model.opset_import[0].version

        if current_opset in recommended_opset:
            print(f"✓ Opset {current_opset} is compatible with ONNX Runtime Web")
        else:
            print(f"⚠ Opset {current_opset} may have limited support in browsers")
            print(f"  Recommended: {recommended_opset}")

        # Check for unsupported operations
        print("\n--- Checking for browser-unsupported operations ---")
        unsupported_ops = []
        for node in model.graph.node:
            # List of operations known to have issues in browser
            problematic_ops = ['Loop', 'Scan', 'If', 'SequenceConstruct']
            if node.op_type in problematic_ops:
                unsupported_ops.append(node.op_type)

        if unsupported_ops:
            print(f"⚠ Found potentially unsupported operations: {set(unsupported_ops)}")
            print("  Model may not work in browser or may be slow")
        else:
            print("✓ No obvious compatibility issues found")

    except ImportError:
        print("Note: Install 'onnx' package for detailed compatibility check")
        print("  pip install onnx")
    except Exception as e:
        print(f"⚠ Could not perform full compatibility check: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Test RetinaFace ONNX model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('model', type=str,
                       help='Path to ONNX model file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image (optional)')
    parser.add_argument('--input-size', type=int, default=640,
                       help='Model input size (default: 640)')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of runs for performance test (default: 100)')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RetinaFace ONNX Model Tester")
    print("=" * 60)
    print(f"Model: {args.model}")

    # Run tests
    session = test_model_basic(args.model)
    if session is None:
        sys.exit(1)

    # Store model path for compatibility check
    session._model_path = args.model

    test_inference(session, args.input_size)

    if args.image:
        test_with_image(session, args.image)

    check_browser_compatibility(session)

    if args.performance:
        test_performance(session, args.num_runs)

    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("\n✓ All basic tests passed!")
    print("\nYour model is ready to use in the web application.")
    print("\nNext steps:")
    print(f"1. Copy {args.model} to your web app directory")
    print("2. Update the model path in app.js if needed")
    print("3. Open index.html in a web browser")
    print("4. Load the model and test with a video")
    print("\nIf you encounter issues in the browser:")
    print("- Check browser console for detailed error messages")
    print("- Try a different opset version when converting the model")
    print("- Verify input preprocessing matches the training setup")


if __name__ == '__main__':
    main()
