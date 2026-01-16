# Face Alignment & Cropping Tool

This project provides a robust solution for detecting faces, extracting facial landmarks, and generating aligned face crops suitable for Face Recognition (FR) model training. It utilizes ONNX-based models for high-performance inference.

## Features

-   **ONNX-Based Inference**: Precise face detection and landmark estimation using `onnxruntime`.
-   **Recursive Processing**: Automatically processes images in nested directory structures.
-   **Robust Alignment**: Algorithms are tuned to use similarity transforms where appropriate, aligning faces without introducing unnatural skew to facial geometry.
-   **Three Optimized Cropping Methodologies**:

    1.  **Buffalo Style** (`buffaloStyle_crop_and_align.py`)
        -   **Characteristics**: Produces a "loose" crop where the face occupies approximately 65% of the image.
        -   **Parameters**: Scale `0.65`, Y-Shift `0.18` (shifts face down).
        -   **Alignment**: Uses a similarity transform (Rotation + Scale + Translation) to strictly preserve head shape, preventing skewing.
        -   **Use Case**: Ideal when more background context is needed.

    2.  **Production Style - Original** (`productionStyle_cropping_and_alignment_original.py`)
        -   **Characteristics**: Standard tight crop fitting the reference template.
        -   **Parameters**: No additional scale or shift parameters.
        -   **Alignment**: Uses a standard affine transform (`cv2.getAffineTransform`).
        -   **Use Case**: Legacy production pipeline requiring a tight fit. *Note: Can introduce slight skewing to force-fit landmarks.*

    3.  **Production Style - Scale & Shift** (`productionStyle_cropping_and_alignment_with_scale_and_shift.py`)
        -   **Characteristics**: An enhanced version of the production crop for better robustness.
        -   **Parameters**: Scale `0.85`, Y-Shift `-0.085` (shifts face up).
        -   **Alignment**: Specifically employs `cv2.estimateAffinePartial2D` to ensure a similarity transform.
        -   **Use Case**: Best balance of tightness and geometric correctness for modern training pipelines.

## Installation

### Prerequisites
-   Python 3.x
-   CUDA-enabled GPU (optional, but recommended for performance).

### Install Dependencies
Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
-   `numpy`
-   `onnxruntime` / `onnxruntime-gpu`
-   `opencv-python`
-   `Pillow`
-   `torchvision`

## Usage

Each script is designed to recursively process a folder of images and save the aligned results to a specified output directory.

### Common Arguments
-   `--image_folder`: Path to the input directory containing images (default: `test_data`).
-   `--output_folder`: Path where cropped images will be saved.
-   `--detector`: Path to the face detector ONNX model (default: `onnx_models/RFB_finetuned_with_postprocessing.onnx`).
-   `--landmark`: Path to the landmark detector ONNX model (default: `onnx_models/landmark_model.onnx`).
-   `--threshold`: Confidence threshold for face detection (model dependent, typically `0.3` - `0.8`).

### 1. Run Buffalo Style Crop
```bash
python buffaloStyle_crop_and_align.py --image_folder "path/to/input_images" --output_folder "output/buffalo_crops"
```

### 2. Run Production Style (Original)
```bash
python productionStyle_cropping_and_alignment_original.py --image_folder "path/to/input_images" --output_folder "output/prod_original_crops"
```

### 3. Run Production Style (Scale & Shift)
```bash
python productionStyle_cropping_and_alignment_with_scale_and_shift.py --image_folder "path/to/input_images" --output_folder "output/prod_scale_shift_crops"
```

## Project Structure

```text
├── buffaloStyle_crop_and_align.py                          # Script for loose, Buffalo-style cropping
├── productionStyle_cropping_and_alignment_original.py      # Original production cropping script
├── productionStyle_cropping_and_alignment_with_scale_and_shift.py # Enhanced production script with scale/shift
├── requirements.txt                                        # Project dependencies
├── onnx_models/                                            # Directory for ONNX models
│   ├── RFB-Epoch-155-Loss-4.onnx
│   ├── RFB_finetuned_with_postprocessing.onnx
│   └── landmark_model.onnx
└── README.md
```
