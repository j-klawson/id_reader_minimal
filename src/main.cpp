// main.cpp
#include "detect_id_card.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;

void printUsage() {
    std::cout << "Usage: ./detect_id_card <image_path> [OPTIONS]" << std::endl;
    std::cout << "Detects and extracts ID cards from images using OpenCV." << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  <image_path>          Path to the input image file." << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                Display this help message and exit." << std::endl;
    std::cout << "  --debug               Enable debug output, including intermediate image files and debug.log." << std::endl;
    std::cout << "  --output-prefix <path>  Prefix for output debug images (e.g., 'test_')." << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    std::string imagePath;
    bool debugMode = false;
    std::string outputPrefix;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage();
            return 0;
        } else if (arg == "--debug") {
            debugMode = true;
        } else if (arg == "--output-prefix") {
            if (i + 1 < argc) {
                outputPrefix = argv[++i];
            } else {
                std::cerr << "Error: --output-prefix requires a path." << std::endl;
                printUsage();
                return -1;
            }
        } else if (imagePath.empty()) {
            imagePath = arg;
        } else {
            std::cerr << "Error: Unknown argument or too many image paths: " << arg << std::endl;
            printUsage();
            return -1;
        }
    }

    if (imagePath.empty()) {
        std::cerr << "Error: Missing image_path argument." << std::endl;
        printUsage();
        return -1;
    }

    // Setup debug output stream
    std::ofstream debugFile;
    std::ostream* debugStream = &std::cout;
    if (debugMode) {
        debugFile.open("debug.log");
        if (debugFile.is_open()) {
            debugStream = &debugFile;
        } else {
            std::cerr << "Warning: Could not open debug.log for writing. Debug output will go to console." << std::endl;
        }
    }

    Mat image = imread(imagePath);
    if (image.empty()) {
        std::cerr << "Could not open image: " << imagePath << std::endl;
        return -1;
    }

    // Resize image for consistent processing and noise reduction
    double max_width = 800.0;
    if (image.cols > max_width) {
        double scale = max_width / image.cols;
        resize(image, image, Size(max_width, image.rows * scale));
    }

    if (debugMode) {
        imwrite(outputPrefix + "original_image.jpg", image);
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(9, 9), 0);

    RotatedRect card = findCardContour(image, gray, *debugStream, debugMode, outputPrefix);

    if (card.size.area() <= 0) {
        std::cout << "No card-like rectangles found." << std::endl;
        std::cout << "Processed image size: " << image.cols << " x " << image.rows << std::endl;
        return 0;
    }

    Point2f vertices[4];
    card.points(vertices);

    // Determine the correct width and height for the warped image
    float warpedWidth = card.size.width;
    float warpedHeight = card.size.height;

    // Ensure width is always the longer dimension for a landscape ID card
    if (warpedWidth < warpedHeight) {
        std::swap(warpedWidth, warpedHeight);
    }

    Point2f dstPts[4] = {
        Point2f(0, 0),
        Point2f(warpedWidth, 0),
        Point2f(warpedWidth, warpedHeight),
        Point2f(0, warpedHeight)
    };

    Mat M = getPerspectiveTransform(vertices, dstPts);
    Mat warped;
    warpPerspective(image, warped, M, Size(warpedWidth, warpedHeight));

    if (debugMode) {
        imwrite(outputPrefix + "detected_card.jpg", warped);
    }

    Mat croppedPortrait;
    if (detectPortrait(warped, croppedPortrait, *debugStream, debugMode, outputPrefix)) {
        imwrite(outputPrefix + "cropped_card.jpg", warped);
        imwrite(outputPrefix + "cropped_portrait.jpg", croppedPortrait);
        std::cout << "Card with portrait detected." << std::endl;
    } else {
        std::cout << "No portrait detected." << std::endl;
    }

    return 0;
}

