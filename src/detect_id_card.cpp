// detect_id_card.cpp
#include "detect_id_card.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "haarcascade_data.h"

using namespace cv;

bool detectPortrait(const Mat &cardROI, Mat& portraitROI, std::ostream& debugStream, bool debugMode) {
    CascadeClassifier face_cascade;
    
    // Load cascade from embedded data using FileStorage
    FileStorage fs(std::string((char*)gHaarCascadeData, gHaarCascadeData_len), FileStorage::MEMORY | FileStorage::READ);
    if (!fs.isOpened()) {
        if (debugMode) debugStream << "Failed to open cascade from memory." << std::endl;
        return false;
    }
    if (!face_cascade.read(fs.getFirstTopLevelNode())) {
        if (debugMode) debugStream << "Failed to read cascade from FileStorage." << std::endl;
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
            if (debugMode) debugStream << "Face detected in portrait region." << std::endl;
            portraitROI = cardROI(face);
            return true;
        }
    }

    if (debugMode) imwrite("portrait_detection_debug.jpg", cardROI);

    return false;
}

RotatedRect findCardContour(const Mat &image, const Mat &gray, std::ostream& debugStream, bool debugMode) {
    // Edge detection on the original image
    Mat edges;
    Canny(gray, edges, 50, 150, 3);

    // Morphological operations to connect edges
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    if (debugMode) imwrite("canny_edges.jpg", edges);

    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (debugMode) debugStream << "Total contours found: " << contours.size() << std::endl;

    if (contours.empty()) {
        if (debugMode) debugStream << "No contours found." << std::endl;
        return RotatedRect();
    }

    Mat debugImage = image.clone();

    // Sort contours by area to find the second largest
    std::vector<std::pair<double, RotatedRect>> contoursByArea;
    
    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

        drawContours(debugImage, std::vector<std::vector<Point>>{approx}, -1, Scalar(0, 255, 0), 2);

        RotatedRect rect = minAreaRect(approx);
        double rectArea = rect.size.width * rect.size.height;

        if (debugMode) debugStream << "Contour #" << i << ": area=" << rectArea
                 << ", vertices=" << approx.size() << std::endl;

        // Only consider contours with reasonable vertex count for rectangles and plausible area
        double imageArea = image.cols * image.rows;
        if (approx.size() >= 4 && approx.size() <= 15 && 
            rectArea > imageArea * 0.1 && rectArea < imageArea * 0.8) {
            contoursByArea.emplace_back(rectArea, rect);
        }
    }
    
    // Sort by area (largest first)
    std::sort(contoursByArea.begin(), contoursByArea.end(), 
              [](const auto &a, const auto &b) { return a.first > b.first; });

    if (debugMode) debugStream << "Found " << contoursByArea.size() << " rectangular contours" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), contoursByArea.size()); ++i) {
        double w = contoursByArea[i].second.size.width;
        double h = contoursByArea[i].second.size.height;
        if (w < h) std::swap(w, h);
        double aspectRatio = w / h;
        if (debugMode) debugStream << "Rank " << i << ": area=" << contoursByArea[i].first
                 << ", aspectRatio=" << aspectRatio << std::endl;
    }

    RotatedRect bestRect;
    double bestRectArea = 0;

    // Find the best contour: largest one with reasonable aspect ratio for an ID card
    for (size_t i = 0; i < contoursByArea.size() && i < 5; ++i) {
        double w = contoursByArea[i].second.size.width;
        double h = contoursByArea[i].second.size.height;
        if (w < h) std::swap(w, h);
        double aspectRatio = w / h;
        
        // ISO/IEC 7810 ID-1 aspect ratio is ~1.586. Allow for some perspective distortion.
        if (aspectRatio >= 1.25 && aspectRatio <= 1.90) {
            bestRect = contoursByArea[i].second;
            bestRectArea = contoursByArea[i].first;
            if (debugMode) debugStream << "Selected rank " << i << " contour with good aspect ratio" << std::endl;
            break;
        }
    }
    
    // Fallback: if no good aspect ratio found, use the largest contour
    if (bestRectArea == 0 && !contoursByArea.empty()) {
        bestRect = contoursByArea[0].second;
        bestRectArea = contoursByArea[0].first;
        if (debugMode) debugStream << "Fallback: selected largest contour" << std::endl;
    }

    if (debugMode) imwrite("detected_rectangles.jpg", debugImage);

    if (bestRectArea == 0) {
        if (debugMode) {
            debugStream << "No rectangle with correct aspect ratio and area found." << std::endl;
            debugStream << "Image size: " << image.cols << " x " << image.rows << std::endl;
        }

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
            if (debugMode) debugStream << "Top contour " << i << ": area=" << areas[i].first
                     << ", size=(" << w << "x" << h << ")"
                     << ", aspectRatio=" << ratio << std::endl;
        }

        return RotatedRect();
    }

    double finalW = bestRect.size.width;
    double finalH = bestRect.size.height;
    if (finalW < finalH) std::swap(finalW, finalH);
    if (debugMode) debugStream << "Selected rect area: " << bestRectArea << " | Aspect Ratio: "
             << finalW / finalH << std::endl;

    return bestRect;
}