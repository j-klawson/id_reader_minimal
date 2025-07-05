// detect_id_card.cpp
#include "detect_id_card.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace cv;

static std::ofstream debugLog("debug.log");

bool detectPortrait(const Mat &cardROI, const std::string &faceCascadePath) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load(faceCascadePath)) {
        debugLog << "Error loading Haar Cascade face detector from: " << faceCascadePath << std::endl;
        return false;
    }

    std::vector<Rect> faces;
    Mat gray;
    cvtColor(cardROI, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(cardROI.cols * 0.2, cardROI.rows * 0.2));

    for (const auto &face : faces) {
        rectangle(cardROI, face, Scalar(255, 0, 0), 2);
        if (face.x < cardROI.cols * 0.4) {
            debugLog << "Face detected in portrait region." << std::endl;
            return true;
        }
    }

    imwrite("portrait_detection_debug.jpg", cardROI);

    return false;
}

RotatedRect findCardContour(const Mat &image, const Mat &gray) {
    Mat edges;
    Canny(gray, edges, 50, 150);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    imwrite("canny_edges.jpg", edges);

    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    debugLog << "Total contours found: " << contours.size() << std::endl;

    if (contours.empty()) {
        debugLog << "No contours found." << std::endl;
        return RotatedRect();
    }

    double imageArea = image.cols * image.rows;
    Mat debugImage = image.clone();

    RotatedRect bestRect;
    double bestRectArea = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

        drawContours(debugImage, std::vector<std::vector<Point>>{approx}, -1, Scalar(0, 255, 0), 2);

        RotatedRect rect = minAreaRect(approx);
        double w = rect.size.width;
        double h = rect.size.height;
        if (w < h) std::swap(w, h);

        double aspectRatio = w / h;
        double rectArea = w * h;

        debugLog << "Contour #" << i << ": area=" << rectArea
                 << ", aspectRatio=" << aspectRatio
                 << ", vertices=" << approx.size()
                 << ", imageArea=" << imageArea << std::endl;

        // Loosened thresholds:
        if (aspectRatio > 1.2 && aspectRatio < 1.8 && rectArea > imageArea * 0.1) {
            debugLog << "--> Candidate contour accepted." << std::endl;
            if (rectArea > bestRectArea) {
                bestRect = rect;
                bestRectArea = rectArea;
            }
        } else {
            debugLog << "--> Rejected due to aspect ratio/area." << std::endl;
        }
    }

    imwrite("detected_rectangles.jpg", debugImage);

    if (bestRectArea == 0) {
        debugLog << "No rectangle with correct aspect ratio and area found." << std::endl;
        debugLog << "Image size: " << image.cols << " x " << image.rows << std::endl;

        std::vector<std::pair<double, RotatedRect>> areas;
        for (const auto &contour : contours) {
            RotatedRect rect = minAreaRect(contour);
            double area = rect.size.width * rect.size.height;
            areas.emplace_back(area, rect);
        }
        std::sort(areas.begin(), areas.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        for (size_t i = 0; i < std::min(size_t(5), areas.size()); ++i) {
            double w = areas[i].second.size.width;
            double h = areas[i].second.size.height;
            if (w < h) std::swap(w, h);
            double ratio = w / h;
            debugLog << "Top contour " << i << ": area=" << areas[i].first
                     << ", size=(" << w << "x" << h << ")"
                     << ", aspectRatio=" << ratio << std::endl;
        }

        return RotatedRect();
    }

    debugLog << "Selected rect area: " << bestRectArea << " | Aspect Ratio: "
             << bestRect.size.width / bestRect.size.height << std::endl;

    return bestRect;
}
