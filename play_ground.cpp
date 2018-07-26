//
// Created by Gang Wang on 7/25/2018.
//
#include <iostream>
#include <list>
#include <numeric>
#include <Eigen/Dense>

#include "utils.h"
#include "tracker.h"

using namespace cv;
using namespace std;
using namespace Eigen;

int main() {
  vector<list<int>> recent(8);

  recent[0].push_back(1);
  recent[0].push_back(2);
  recent[0].push_back(3);
  recent[0].push_back(4);

  recent[1].push_back(5);
  recent[1].push_back(6);
  recent[1].push_back(7);
  recent[1].push_back(8);
  recent[1].push_back(9);
  recent[1].pop_front();

  double first = std::accumulate( recent[0].begin(), recent[0].end(), 0.0)/recent[0].size();
  double second = std::accumulate( recent[1].begin(), recent[1].end(), 0.0)/recent[1].size();

  cout << first << endl;
  cout << second << endl;

  VectorXd v(recent.size());

  int i = 0;
  for (const auto& r : recent) {
    double avg = std::accumulate(r.begin(), r.end(), 0.0)/r.size();
    v[i++] = avg;
  }
  cout << "Hello Playground" << endl;
  cout << "v" << endl << v.transpose() << endl;
}