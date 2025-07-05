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
    // Color-based detection: use k-means clustering to separate distinct color regions
    Mat colorMask;
    
    // Reshape image to 1D array of pixels for k-means
    Mat data;
    image.reshape(1, image.rows * image.cols).convertTo(data, CV_32F);
    
    // K-means clustering to find dominant colors (background vs card)
    int k = 3; // Try 3 clusters to separate background, card, and any other elements
    Mat labels, centers;
    kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 1.0),
           3, KMEANS_PP_CENTERS, centers);
    
    // Reshape labels back to image dimensions
    labels = labels.reshape(1, image.rows);
    
    // Find the cluster with the largest area (likely background)
    std::vector<int> clusterCounts(k, 0);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            clusterCounts[labels.at<int>(i, j)]++;
        }
    }
    
    // Find background cluster (largest) and card cluster (second largest or most central)
    int backgroundCluster = std::max_element(clusterCounts.begin(), clusterCounts.end()) - clusterCounts.begin();
    
    // Create mask excluding the background cluster
    colorMask = Mat::zeros(image.size(), CV_8UC1);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            if (labels.at<int>(i, j) != backgroundCluster) {
                colorMask.at<uchar>(i, j) = 255;
            }
        }
    }
    
    // Clean up the mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(colorMask, colorMask, MORPH_CLOSE, kernel);
    morphologyEx(colorMask, colorMask, MORPH_OPEN, kernel);
    
    imwrite("color_mask.jpg", colorMask);
    
    // Edge detection on the original image
    Mat edges;
    // Stronger edge detection for better card boundary detection
    Canny(gray, edges, 100, 200, 3);

    // Larger morphological operations to better connect card edges
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel2);
    
    // Additional dilation to strengthen edges
    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(2, 2));
    dilate(edges, edges, kernel3);

    // Combine color mask with edge detection for better contour detection
    Mat combinedMask;
    bitwise_and(edges, colorMask, combinedMask);
    
    imwrite("canny_edges.jpg", edges);
    imwrite("combined_mask.jpg", combinedMask);

    std::vector<std::vector<Point>> contours;
    // Use combined mask for better contour detection
    findContours(combinedMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    debugLog << "Total contours found: " << contours.size() << std::endl;

    if (contours.empty()) {
        debugLog << "No contours found." << std::endl;
        return RotatedRect();
    }

    double imageArea = image.cols * image.rows;
    Mat debugImage = image.clone();

    // Sort contours by area to find the second largest
    std::vector<std::pair<double, RotatedRect>> contoursByArea;
    
    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

        drawContours(debugImage, std::vector<std::vector<Point>>{approx}, -1, Scalar(0, 255, 0), 2);

        RotatedRect rect = minAreaRect(approx);
        double rectArea = rect.size.width * rect.size.height;

        debugLog << "Contour #" << i << ": area=" << rectArea
                 << ", vertices=" << approx.size() << std::endl;

        // Only consider contours with reasonable vertex count for rectangles
        if (approx.size() >= 4 && approx.size() <= 15) {
            contoursByArea.emplace_back(rectArea, rect);
        }
    }
    
    // Sort by area (largest first)
    std::sort(contoursByArea.begin(), contoursByArea.end(), 
              [](const auto &a, const auto &b) { return a.first > b.first; });

    debugLog << "Found " << contoursByArea.size() << " rectangular contours" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), contoursByArea.size()); ++i) {
        double w = contoursByArea[i].second.size.width;
        double h = contoursByArea[i].second.size.height;
        if (w < h) std::swap(w, h);
        double aspectRatio = w / h;
        debugLog << "Rank " << i << ": area=" << contoursByArea[i].first
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
        
        // ID cards typically have aspect ratio between 1.3 and 1.8
        if (aspectRatio >= 1.3 && aspectRatio <= 1.8) {
            bestRect = contoursByArea[i].second;
            bestRectArea = contoursByArea[i].first;
            debugLog << "Selected rank " << i << " contour with good aspect ratio" << std::endl;
            break;
        }
    }
    
    // Fallback: if no good aspect ratio found, use the second largest
    if (bestRectArea == 0 && contoursByArea.size() >= 2) {
        bestRect = contoursByArea[1].second;
        bestRectArea = contoursByArea[1].first;
        debugLog << "Fallback: selected second largest contour" << std::endl;
    } else if (bestRectArea == 0 && contoursByArea.size() == 1) {
        bestRect = contoursByArea[0].second;
        bestRectArea = contoursByArea[0].first;
        debugLog << "Fallback: only one contour found, using it" << std::endl;
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

    double finalW = bestRect.size.width;
    double finalH = bestRect.size.height;
    if (finalW < finalH) std::swap(finalW, finalH);
    debugLog << "Selected rect area: " << bestRectArea << " | Aspect Ratio: "
             << finalW / finalH << std::endl;

    return bestRect;
}
