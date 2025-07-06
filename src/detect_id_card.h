// detect_id_card.h
#ifndef DETECT_ID_CARD_H
#define DETECT_ID_CARD_H

#include <opencv2/opencv.hpp>
#include <string>

#include <ostream>

cv::RotatedRect findCardContour(const cv::Mat &image, const cv::Mat &gray, std::ostream& debugStream, bool debugMode, const std::string& outputPrefix);
bool detectPortrait(const cv::Mat &cardROI, const std::string &faceCascadePath, cv::Mat& portraitROI, std::ostream& debugStream, bool debugMode, const std::string& outputPrefix);

#endif // DETECT_ID_CARD_H

