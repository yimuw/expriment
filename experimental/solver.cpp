#include <iostream>
#include <fstream>

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


std::vector<std::vector<double>> readSensorData(std::string path) {
  std::vector<std::vector<double>> ret;

  std::ifstream csvFile;
  csvFile.open(path);

  std::string line;
  std::vector<double> row;
  while(std::getline(csvFile, line)) {
    std::cout << "line:" << line << std::endl;
    std::istringstream s(line);
    std::string field;
    while (std::getline(s, field,',')) {
      std::cout << "field: " << field << std::endl;
      row.push_back(std::stod(field));
    }
    ret.push_back(row);
  }

  return ret;
}

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "missing arg for the csv file" << std::endl;
  }

  std::string path = argv[1];
  std::vector<std::vector<double>> data = readSensorData(path);
  return 0;
}