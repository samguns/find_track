#include <iostream>

#include "utils.h"
#include "tracker.h"

using namespace cv;
using namespace std;

static void extract() {
  VideoCapture cap("project_video.mp4");
  if (!cap.isOpened()) {
    cout << "Error opening video file" << endl;
    return;
  }

  int currentFrame(0);
  while (true) {
    Mat frame;

    cap >> frame;

    if (frame.empty()) {
      break;
    }

    string name = "./data/frame" +
        to_string(static_cast<int>(currentFrame)) + ".jpg";
    imwrite(name, frame);

    currentFrame++;
  }

  cap.release();
}

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
//  line_tracker.unitTest();

//  namedWindow("warp");
//  imshow("warp", warped);
//  waitKey(0);

  extract();

  return 0;
}