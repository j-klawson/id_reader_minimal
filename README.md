# ID Card Reader

A minimal C++ application that detects and extracts ID cards from images using OpenCV.

## Features

- Detects ID-1 standard cards (credit card size) on solid backgrounds
- Extracts card region using perspective transformation
- Attempts portrait detection within the card region
- Outputs debug images and logs for troubleshooting

## Requirements

- OpenCV 4.x
- C++17 compiler
- `xxd` (usually part of `vim-common` or `vim-runtime` packages on Linux, or available by default on macOS)

## Build & Run

The Haar Cascade XML file (`haarcascade_frontalface_default.xml`) is converted into a C++ array during the build process and embedded directly into the executable. This eliminates the need to distribute the XML file separately.

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
cd ..
./bin/detect_id_card <image_path>
```

**Running Tests:**
```bash
cd build
ctest
cd ..
```

## Output Files

When the `--debug` flag is used, the following files are generated.

- `original_image.jpg` - Input image
- `canny_edges.jpg` - Edge detection visualization
- `detected_rectangles.jpg` - Contour detection debug
- `detected_card.jpg` - Extracted card region
- `portrait_detection_debug.jpg` - Face detection debug (if portrait found)
- `debug.log` - Detailed processing log

## Algorithm

1. Convert to grayscale and apply Gaussian blur
2. **Edge detection**: Apply Canny edge detection with morphological operations
3. Find contours and select best rectangular candidate by size and aspect ratio (typically 1.25-1.90 for ID cards)
4. Apply perspective transformation to extract card region
5. Attempt face detection on extracted region using Haar cascades
