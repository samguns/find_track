//
// Created by Sam on 2018/7/19.
//

#include "tracker.h"

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
{}


bool tracker::find_window_centroids(cv::Mat &img) {
  return true;
}

void tracker::unitTest() {
  Mat img = imread("python/warped_example.jpg");
  find_window_centroids(img);
}