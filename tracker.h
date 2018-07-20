//
// Created by Sam on 2018/7/19.
//

#ifndef FIND_TRACK_TRACKER_H
#define FIND_TRACK_TRACKER_H

#include "Eigen/Dense"
#include <opencv2/opencv.hpp>

class tracker {
 public:
  tracker(int win_width, int win_height, int margin,
          double xmpp, double ympp, double smooth_factor);
  ~tracker() = default;

  bool find_window_centroids(cv::Mat& img);
  void unitTest();

 private:
  int mWindowWidth;
  int mWindowHeight;
  int mMargin;
  double mXmPerPixel;
  double mYmPerPixel;
  double mSmoothFactor;

  cv::Mat convolve(cv::Mat& f_1d, cv::Mat& g_1d);
};

#endif //FIND_TRACK_TRACKER_H
