// main.cpp
#include "detect_id_card.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./detect_id_card <image_path> [face_cascade_path]" << std::endl;
        return -1;
    }

    std::string faceCascadePath = "./assets/haarcascade_frontalface_default.xml";
    if (argc >= 3) {
        faceCascadePath = argv[2];
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Could not open image: " << argv[1] << std::endl;
        return -1;
    }

    imwrite("original_image.jpg", image);

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);

    RotatedRect card = findCardContour(image, gray);

    if (card.size.area() <= 0) {
        std::cout << "No card-like rectangles found." << std::endl;
        std::cout << "Original image size: " << image.cols << " x " << image.rows << std::endl;
        return 0;
    }

    Point2f vertices[4];
    card.points(vertices);

    Point2f dstPts[4] = {
        Point2f(0, 0),
        Point2f(card.size.width, 0),
        Point2f(card.size.width, card.size.height),
        Point2f(0, card.size.height)
    };

    Mat M = getPerspectiveTransform(vertices, dstPts);
    Mat warped;
    warpPerspective(image, warped, M, Size(card.size.width, card.size.height));

    imwrite("detected_card.jpg", warped);

    if (detectPortrait(warped, faceCascadePath)) {
        imwrite("cropped_card.jpg", warped);
        std::cout << "Card with portrait detected." << std::endl;
    } else {
        std::cout << "No portrait detected." << std::endl;
    }

    return 0;
}

