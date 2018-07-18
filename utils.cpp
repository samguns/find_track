//
// Created by Gang Wang on 7/18/2018.
//

#include <iostream>
#include "utils.h"

using namespace cv;
using namespace std;


static void hsv_select(Mat& img, int lowerbound, int upperbound,
                       Mat& thresholded);
static void red_select(Mat& img, int lowerbound, int upperbound,
                       Mat& thresholded);
static void warp(Mat& binary, double bottom_width, double mid_width,
                 double height_pct, Mat& warped, Mat& Minv);


void warp_image(Mat& image, Mat& mtx, Mat& distCoeffs,
                Mat& warpedImage, Mat& mtxInverse) {
  if (image.empty() || mtx.empty() || distCoeffs.empty()) {
    std::cout << "Invalid input" << std::endl;
    return;
  }

  Mat undist;
  undistort(image, undist, mtx, distCoeffs);

  Mat v_channel;
  hsv_select(undist, 180, 255, v_channel);
  Mat r_channel;
  red_select(undist, 170, 255, r_channel);

  Mat binary = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);
  bitwise_and(r_channel, v_channel, binary);

  float bottom_width = 0.76;
  float mid_width = 0.2;
  float height_pct = 0.68;

  warp(binary, bottom_width, mid_width,
       height_pct, warpedImage, mtxInverse);
}


static void hsv_select(Mat& img, int lowerbound, int upperbound,
                       Mat& thresholded) {
  Mat hsv;
  cvtColor(img, hsv, COLOR_BGR2HSV);
  vector<Mat> hsvChannels(3);
  split(img, hsvChannels);

  thresholded = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);
  inRange(hsvChannels[2], Scalar(lowerbound), Scalar(upperbound), thresholded);
}

static void red_select(Mat& img, int lowerbound, int upperbound,
                       Mat& thresholded) {
  vector<Mat> bgrChannels(3);
  split(img, bgrChannels);

  thresholded = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);
  inRange(bgrChannels[2], Scalar(lowerbound), Scalar(upperbound), thresholded);
}

static void warp(Mat& img, double bottom_width, double mid_width,
                 double height_pct, Mat& warped, Mat& Minv) {
  Point2f src[4];
  Point2f dst[4];
  int width = img.cols;
  int height = img.rows;
  double bottom_trim = 0.935;
  double offset = (double)width * 0.25;

  src[0] = Point2f(width * (0.5-mid_width/2), height * height_pct);
  src[1] = Point2f(width * (0.5+mid_width/2), height * height_pct);
  src[2] = Point2f(width * (0.5+bottom_width/2), height * bottom_trim);
  src[3] = Point2f(width * (0.5-bottom_width/2), height * bottom_trim);

  dst[0] = Point2f(offset, 0);
  dst[1] = Point2f(width - offset, 0);
  dst[2] = Point2f(width - offset, height);
  dst[3] = Point2f(offset, height);

  Mat M = getPerspectiveTransform(src, dst);
  Minv = getPerspectiveTransform(dst, src);

  warpPerspective(img, warped, M, img.size());
}
