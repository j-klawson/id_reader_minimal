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
- `color_mask.jpg` - Color-based segmentation result
- `canny_edges.jpg` - Edge detection visualization
- `combined_mask.jpg` - Combined color and edge detection
- `detected_rectangles.jpg` - Contour detection debug
- `detected_card.jpg` - Extracted card region
- `portrait_detection_debug.jpg` - Face detection debug (if portrait found)
- `debug.log` - Detailed processing log

## Algorithm

1. Convert to grayscale and apply Gaussian blur
2. **Color-based segmentation**: Use k-means clustering to separate card from background based on distinct colors
3. **Edge detection**: Apply Canny edge detection with morphological operations
4. **Combined approach**: Merge color mask with edge detection for cleaner contours
5. Find contours and select best rectangular candidate by size and aspect ratio (typically 1.3-1.8 for ID cards)
6. Apply perspective transformation to extract card region
7. Attempt face detection on extracted region using Haar cascades