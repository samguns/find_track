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
  int image_width = img.cols;
  int image_height = img.rows;

  Mat window = Mat::ones(1, mWindowWidth, CV_32F);

  /* Get bottom 3/4 quarter of the img */
  int w = image_width / 2;
  int h = image_height / 4;
  Rect rectLeft(0, 3 * h, w, h);
  Rect rectRight(w, 3 * h, w, h);
  Mat leftQ, rightQ;
  img(rectLeft).copyTo(leftQ);
  img(rectRight).copyTo(rightQ);

  /* Squash 2-D array into 1-D signals for convolution */
  Mat l_sum, r_sum;
  reduce(leftQ, l_sum, 0, REDUCE_SUM, CV_32F);
  reduce(rightQ, r_sum, 0, REDUCE_SUM, CV_32F);

  Mat l_conv = convolve(window, l_sum);
  Mat r_conv = convolve(window, r_sum);

  return true;
}

void tracker::unitTest() {
  Mat test(1, 3, CV_32FC1);
  test.at<double>(0, 0) = 9;
  test.at<double>(0, 1) = 12;
  test.at<double>(0, 2) = 15;
  Mat conv = Mat::ones(1, 2, CV_32FC1);
  cout << "test " << test << endl;
  cout << "conv " << conv << endl;
  convolve(test, conv);
  Mat img = imread("python/warped_example.jpg");
  find_window_centroids(img);
}


Mat tracker::convolve(Mat &f_1d, Mat &g_1d) {
  int const nf = f_1d.cols;
  int const ng = g_1d.cols;
  int const n = nf + ng -1;

  cout << "f_1d " << f_1d << endl;
  cout << "g_1d " << g_1d << endl;

  Mat out = Mat::zeros(1, n, CV_32FC1);
  for (int i = 0; i < n; i++) {
    int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
    int const jmx = (i < ng - 1) ? i : nf - 1;

    for (int j = jmn; j <= jmx; j++) {
      cout << "i " << i << " j " << j << endl;
      cout << "out " << out << endl;
      cout << "f_1d.at<double>(0, j) " << f_1d.at<double>(0, j) << endl;
      cout << "g_1d.at<double>(0, i-j) " << g_1d.at<double>(0, i-j) << endl;
      //out.ptr<double>(0)[i] += (f_1d.ptr<double>(0)[j] * g_1d.ptr<double>(0)[i-j]);
      out.at<double>(0, i) += (f_1d.at<double>(0, j) * g_1d.at<double>(0, i-j));
    }
  }

  cout << "out: " << endl << out << endl;

  return out;
}