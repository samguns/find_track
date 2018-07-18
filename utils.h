//
// Created by Gang Wang on 7/18/2018.
//

#ifndef FIND_TRACK_UTILS_H
#define FIND_TRACK_UTILS_H

#include <opencv2/opencv.hpp>

void warp_image(cv::Mat& image, cv::Mat& mtx, cv::Mat& distCoeffs,
                cv::Mat& warpedImage, cv::Mat& mtxInverse);

#endif //FIND_TRACK_UTILS_H
