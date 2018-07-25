//
// Created by Sam on 2018/7/19.
//
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <algorithm>
#include "tracker.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;


tracker::tracker(int win_width, int win_height, int margin,
                 double xmpp, double ympp, double smooth_factor)
    : mWindowWidth(win_width),
      mWindowHeight(win_height),
      mMargin(margin),
      mXmPerPixel(xmpp),
      mYmPerPixel(ympp),
      mSmoothFactor(smooth_factor)
{
  mLeftFit.setZero();
  mRightFit.setZero();
}


bool tracker::find_window_centroids(cv::Mat &img) {
  int image_width = img.cols;
  int image_height = img.rows;

  int offset = mWindowWidth / 2;

  /* Get bottom 3/4 quarter of the img */
  int w = image_width / 2;
  int h = image_height / 4;
  Rect rectLeft(0, 3 * h, w, h);
  Rect rectRight(w, 3 * h, w, h);
  Mat leftQ, rightQ;
  img(rectLeft).copyTo(leftQ);
  img(rectRight).copyTo(rightQ);

  Matrix<int, Dynamic, Dynamic> window(1, mWindowWidth);
  window.setOnes();

  Matrix<int, Dynamic, Dynamic> eigenLeftQ;
  cv2eigen(leftQ, eigenLeftQ);
  Matrix<int, Dynamic, Dynamic> l_sum = eigenLeftQ.colwise().sum();
  Matrix<int, Dynamic, Dynamic> l_conv = convolve(l_sum, window);
  Matrix<int, Dynamic, Dynamic>::Index maxRow, leftCenter, rightCenter;
  l_conv.maxCoeff(&maxRow, &leftCenter);

  Matrix<int, Dynamic, Dynamic> eigenRightQ;
  cv2eigen(rightQ, eigenRightQ);
  Matrix<int, Dynamic, Dynamic> r_sum = eigenRightQ.colwise().sum();
  Matrix<int, Dynamic, Dynamic> r_conv = convolve(r_sum, window);
  r_conv.maxCoeff(&maxRow, &rightCenter);

  leftCenter -= offset;
  rightCenter -= offset;
  rightCenter += image_width / 2;
  int y = image_height - (mWindowHeight/2);
  reject_anomaly(leftCenter, rightCenter, y);
  mRecentCentroids.push_back(Point(leftCenter, rightCenter));

  for (int level = 1; level < image_height/mWindowHeight; level++) {
    y = image_height - (level+1) * mWindowHeight;
    Rect rectLayer(0, y, image_width, mWindowHeight);
    Mat subImage;
    img(rectLayer).copyTo(subImage);

    Matrix<int, Dynamic, Dynamic> eigenImageLayer;
    cv2eigen(subImage, eigenImageLayer);
    Matrix<int, Dynamic, Dynamic> image_layer = eigenImageLayer.colwise().sum();
    Matrix<int, Dynamic, Dynamic> conv_signal = convolve(image_layer, window);
    int min_index = max(int(leftCenter + offset - mMargin), 0);
    int max_index = min(int(leftCenter + offset + mMargin), image_width);
    int cols = max_index - min_index;
    Matrix<int, Dynamic, Dynamic> leftConv(1, cols);
    leftConv = conv_signal.block(0, min_index, 1, cols);

    leftConv.maxCoeff(&maxRow, &leftCenter);
    leftCenter = leftCenter + min_index - offset;

    min_index = max(int(rightCenter + offset - mMargin), 0);
    max_index = min(int(rightCenter + offset + mMargin), image_width);
    cols = max_index - min_index;
    Matrix<int, Dynamic, Dynamic> rightConv(1, cols);
    rightConv = conv_signal.block(0, min_index, 1, cols);
    rightConv.maxCoeff(&maxRow, &rightCenter);
    rightCenter = rightCenter + min_index - offset;

    reject_anomaly(leftCenter, rightCenter, y + mWindowHeight/2);
    mRecentCentroids.push_back(Point(leftCenter, rightCenter));
  }

  return true;
}

void tracker::unitTest() {
//  vector<Point> a, b;
//  a.emplace_back(1, 2);
//  a.emplace_back(3, 4);
//
//  b.emplace_back(5, 6);
//  b.emplace_back(7, 8);
//
//  cout << "a total " << accumulate(a.begin(), a.end(), Point(0, 0)) << endl;
//  cout << "b total " << accumulate(b.begin(), b.end(), Point(0, 0)) << endl;

//  Matrix<int, Dynamic, Dynamic> window(1, 2);
//  window.setOnes();
//
//  Matrix<int, Dynamic, Dynamic> test(1, 3);
//  test << 9, 12, 15;
//  cout << window << endl;
//  cout << test << endl;
//
//  Matrix<int, Dynamic, Dynamic> l_conv = convolve(test, window);
//  cout << l_conv << endl;

  Mat img = imread("python/warped_example.jpg");
  Mat gray;
  cvtColor(img, gray, CV_BGR2GRAY);
  find_window_centroids(gray);

  VectorXd res_yval = VectorXd::LinSpaced(9, 40, 680);
  res_yval.reverseInPlace();

  VectorXd leftx(9);
  VectorXd rightx(9);
  for (int level = 0; level < mRecentCentroids.size(); level++) {
    leftx[level] = mRecentCentroids[level].x;
    rightx[level] = mRecentCentroids[level].y;
  }

  VectorXd left_fit = polyfit(res_yval, leftx, 2);
  cout << left_fit.transpose() << endl;
}


Matrix<int, Dynamic, Dynamic> tracker::convolve(Matrix<int, Dynamic, Dynamic> &input,
                                                Matrix<int, Dynamic, Dynamic> &kernel) {
  int const nInput = input.cols();
  int const nKernel = kernel.cols();
  int const n = nInput + nKernel - 1;

  Matrix<int, Dynamic, Dynamic> out(1, n);
  out.setZero();

  for (int i = 0; i < n; i++) {
    int const startk = i >= nInput ? i - nInput + 1 : 0;
    int const endk = i < nKernel ? i : nKernel - 1;

    for (int k = startk; k <= endk; k++) {
      out(0, i) += (input(0, i-k) * kernel(0, k));
    }
  }

  return out;
}

void tracker::reject_anomaly(Matrix<int, Dynamic, Dynamic>::Index &l_center,
                             Matrix<int, Dynamic, Dynamic>::Index &r_center,
                             int y) {
  if (!mLeftFit.isZero(0)) {
    double ref_l_center = polyeval(mLeftFit, (double)y);

    if ((l_center < (ref_l_center - mMargin)) ||
        (l_center > (ref_l_center + mMargin))) {
      l_center = ref_l_center;
    }
  }

  if (!mRightFit.isZero(0)) {
    double ref_r_center = polyeval(mRightFit, (double)y);

    if ((r_center < (ref_r_center - mMargin)) ||
        (r_center > (ref_r_center + mMargin))) {
      r_center = ref_r_center;
    }
  }
}