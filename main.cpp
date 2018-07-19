#include <iostream>

#include "utils.h"
#include "tracker.h"

using namespace cv;
using namespace std;


int main() {
  FileStorage fs("cal_params.yml", FileStorage::READ);
  Mat mtx;
  Mat distCoeffs;
  fs["mtx"] >> mtx;
  fs["dist"] >> distCoeffs;

  Mat testImg = imread("test_images/test3.jpg");
  if (testImg.empty()) {
    cout << "Couldn't open test image." << endl;
    return -1;
  }

  Mat warped, mInv;
  warp_image(testImg, mtx, distCoeffs, warped, mInv);

  tracker line_tracker(50, 80, 100, 1, 1, 15);
  line_tracker.unitTest();

//  namedWindow("warp");
//  imshow("warp", warped);
//  waitKey(0);

  return 0;
}