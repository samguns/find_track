//
// Created by Sam on 2018/7/19.
//

#ifndef FIND_TRACK_TRACKER_H
#define FIND_TRACK_TRACKER_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <list>

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
  Eigen::VectorXd mLeftFit;
  Eigen::VectorXd mRightFit;

  std::vector<std::list<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Index>> mRecentLeftCentroids;
  std::vector<std::list<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Index>> mRecentRightCentroids;

  std::vector<cv::Point> mRecentCentroids;

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> convolve(
      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>& input,
      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>& kernel
      );

  void reject_anomaly(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Index& l_center,
                      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Index& r_center,
                      int y);
};

#endif //FIND_TRACK_TRACKER_H
