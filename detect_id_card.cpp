#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;

// Helper to sort contours by area
bool sortByArea(const RotatedRect& a, const RotatedRect& b) {
    return a.size.area() > b.size.area();
}

// Detect portrait-like rectangle inside a cropped card
bool detectPortraitRegion(const Mat& card) {
    Mat gray, edges;
    cvtColor(card, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    Canny(gray, edges, 50, 150);

    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        RotatedRect rect = minAreaRect(contour);
        float w = rect.size.width;
        float h = rect.size.height;
        if (w < h) std::swap(w, h);
        float aspectRatio = h / w;  // Portrait ratio
        float area = w * h;

        if (aspectRatio > 0.6 && aspectRatio < 0.9 && area > (card.cols * card.rows * 0.05)) {
            Point2f center = rect.center;
            // Ensure it's on the left third of the card
            if (center.x < card.cols * 0.4) {
                std::cout << "Detected possible portrait region (aspect ratio: " << aspectRatio << ")" << std::endl;
                return true;
            }
        }
    }

    return false;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./detect_id_card_with_portrait <image_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    Mat image = imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return -1;
    }

    imwrite("original_image.jpg", image);

    Mat gray, edges;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    Canny(gray, edges, 50, 150);

    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    const double imageArea = image.cols * image.rows;
    std::vector<RotatedRect> cardCandidates;

    for (const auto& contour : contours) {
        RotatedRect rect = minAreaRect(contour);
        float w = rect.size.width;
        float h = rect.size.height;
        if (w < h) std::swap(w, h);
        float aspectRatio = w / h;
        float area = w * h;

        if (aspectRatio > 1.4 && aspectRatio < 1.65 &&
            area > imageArea * 0.15 && area < imageArea * 0.95) {
            cardCandidates.push_back(rect);
        }
    }

    if (cardCandidates.empty()) {
        std::cout << "No card-shaped rectangles found." << std::endl;
        return 0;
    }

    std::sort(cardCandidates.begin(), cardCandidates.end(), sortByArea);
    RotatedRect cardRect = cardCandidates.front();

    // Warp the card to a flat rectangle
    Point2f srcPoints[4];
    cardRect.points(srcPoints);
    Point2f dstPoints[4] = {
        Point2f(0, 0),
        Point2f(cardRect.size.width, 0),
        Point2f(cardRect.size.width, cardRect.size.height),
        Point2f(0, cardRect.size.height)
    };

    Mat warpMatrix = getPerspectiveTransform(srcPoints, dstPoints);
    Mat warpedCard;
    warpPerspective(image, warpedCard, warpMatrix, cardRect.size);

    imwrite("detected_card.jpg", warpedCard);

    // Detect portrait region in warped card
    if (detectPortraitRegion(warpedCard)) {
        imwrite("cropped_card.jpg", warpedCard);
        std::cout << "✅ ID-1 card with portrait region detected." << std::endl;
    } else {
        std::cout << "⚠️ No portrait region found. Skipping save." << std::endl;
    }

    return 0;
}
