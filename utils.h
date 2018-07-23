//
// Created by Gang Wang on 7/18/2018.
//

#ifndef FIND_TRACK_UTILS_H
#define FIND_TRACK_UTILS_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

void warp_image(cv::Mat& image, cv::Mat& mtx, cv::Mat& distCoeffs,
                cv::Mat& warpedImage, cv::Mat& mtxInverse);

Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order);
double polyeval(Eigen::VectorXd coeffs, double x);

#endif //FIND_TRACK_UTILS_H
