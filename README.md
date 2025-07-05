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
- pkg-config

## Build & Run

```bash
make
./detect_id_card <image_path> [face_cascade_path]
```

Or use the test script:
```bash
./runtest.sh
```

## Output Files

- `original_image.jpg` - Input image
- `canny_edges.jpg` - Edge detection visualization
- `detected_rectangles.jpg` - Contour detection debug
- `detected_card.jpg` - Extracted card region
- `portrait_detection_debug.jpg` - Face detection debug (if portrait found)
- `debug.log` - Detailed processing log

## Algorithm

1. Convert to grayscale and apply Gaussian blur
2. Canny edge detection with morphological operations
3. Find contours and select best rectangular candidate by size and aspect ratio
4. Apply perspective transformation to extract card
5. Attempt face detection on extracted region