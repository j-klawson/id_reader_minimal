// detect_id_card.h
#ifndef DETECT_ID_CARD_H
#define DETECT_ID_CARD_H

#include <opencv2/opencv.hpp>
#include <string>

cv::RotatedRect findCardContour(const cv::Mat &image, const cv::Mat &gray);
bool detectPortrait(const cv::Mat &cardROI, const std::string &faceCascadePath);

#endif // DETECT_ID_CARD_H

