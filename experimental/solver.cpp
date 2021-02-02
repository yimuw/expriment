#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

const double DT = 1.0;
const Eigen::Vector3d GRAVITY{0,0,0};

struct State {
  Eigen::Vector3d pos = Eigen::Vector3d::Random(); // position 
  Eigen::Vector3d vel = Eigen::Vector3d::Random(); // velocity
  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom(); // pose Qwr

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Measurement{
  Eigen::Matrix3d Rwr;
  Eigen::Vector3d twr;
  Eigen::Vector3d acc;
  Eigen::Vector3d omega; 

  Measurement(Eigen::Matrix3d Rwr,
              Eigen::Vector3d twr,
              Eigen::Vector3d acc,
              Eigen::Vector3d omega)
    : Rwr(Rwr), twr(twr), acc(acc), omega(omega) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PosError {
  PosError(const Eigen::Vector3d& pos_measured)
    : pos_measured_(pos_measured) {}

  template <typename T>
  bool operator()(const T* const pos_hat_ptr, T* residuals_ptr) const {      
    Eigen::Matrix<T, 3, 1> residuals(residuals_ptr);
    Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
    residuals = pos_hat - pos_measured_;
    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& pos_measured) {
    return new AutoDiffCostFunction<PosError, 3, 3>(
      new PosError(pos_measured));
  }

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const Eigen::Vector3d pos_measured_;
};


struct PredictionError{
  PredictionError(const Eigen::Vector3d& acc_measured,
                  const Eigen::Vector3d& omega_measured)
    : acc_measured_(acc_measured), omega_measured_(omega_measured) {}

  template <typename T>
  bool operator()(const T* const pos_b_ptr,
                  const T* const vel_b_ptr,
                  const T* const q_b_ptr,
                  const T* const pos_e_ptr,
                  const T* const vel_e_ptr,
                  const T* const q_e_ptr,
                  T* residuals_ptr) const {
    Eigen::Matrix<T, 9, 1> residuals(residuals_ptr);

    // quat error
    const Eigen::Quaternion<T> q_b(q_b_ptr);
    const Eigen::Quaternion<T> q_e(q_e_ptr);
    // https://gamedev.stackexchange.com/questions/108920/applying-angular-velocity-to-quaternion 
    Eigen::Quaternion<T> q_new;
    Eigen::Quaterniond q_omega;
    Eigen::Quaternion<T> q_add; 
    q_omega.w() = 0;
    q_omega.vec() = omega_measured_ * DT * 0.5;
    q_add = q_omega.template cast<T>() * q_b;
    q_new.w() = q_b.w() + q_add.w();
    q_new.vec() = q_b.vec() + q_add.vec();

    Eigen::Quaternion<T> q_delta = q_e.conjugate() * q_new;
    residuals.block(6, 0, 3, 1) = q_delta.vec();

    // vel error
    const Eigen::Matrix<T, 3, 1> vel_b(vel_b_ptr);
    const Eigen::Matrix<T, 3, 1> vel_e(vel_e_ptr);
    residuals.block(3, 0, 3, 1) = vel_b + (q_b * acc_measured_.template cast<T>() - GRAVITY) * DT - vel_e;

    // pos error
    const Eigen::Matrix<T, 3, 1> pos_b(pos_b_ptr);
    const Eigen::Matrix<T, 3, 1> pos_e(pos_e_ptr);
    residuals.block(0, 0, 3, 1) = pos_b + vel_b * DT - pos_e;

    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& acc_measured,
                              const Eigen::Vector3d& omega_measured) {
    return new AutoDiffCostFunction<PredictionError, 9, 3, 3, 4, 3, 3, 4>(
      new PredictionError(acc_measured, omega_measured));
  }

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const Eigen::Vector3d acc_measured_;
  const Eigen::Vector3d omega_measured_;
};

std::vector<Measurement, Eigen::aligned_allocator<Measurement>> readSensorData(std::string path) {
  std::vector<Measurement, Eigen::aligned_allocator<Measurement>> ret;

  std::ifstream csvFile;
  csvFile.open(path);

  std::string line;
  while(std::getline(csvFile, line)) {
    std::vector<double> row;
    std::cout << "line:" << line << std::endl;
    std::istringstream s(line);
    std::string field;
    while (std::getline(s, field,',')) {
      std::cout << "field: " << field << std::endl;
      row.push_back(std::stod(field));
    }

    Eigen::Matrix3d Rwr;
    Eigen::Vector3d twr;
    Eigen::Vector3d acc;
    Eigen::Vector3d omega; 
    Rwr << row[0], row[1], row[2],
           row[4], row[5], row[6],
           row[8], row[9], row[10];
    twr << row[3], row[7], row[11];
    acc << row[16], row[17], row[18];
    omega << row[19], row[20], row[21];
    std::cout << "Rwr: " << Rwr << std::endl;
    std::cout << "twr: " << twr << std::endl;
    std::cout << "acc: " << acc << std::endl;
    std::cout << "omega: " << omega << std::endl;

    ret.push_back(Measurement(Rwr, twr, acc, omega));
  }

  return ret;
}

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "missing arg for the csv file" << std::endl;
  }

  std::string path = argv[1];
  std::vector<Measurement, Eigen::aligned_allocator<Measurement>> data = readSensorData(path);    
  // int cnt = data.size();
  int cnt = 2;

  std::vector<State, Eigen::aligned_allocator<State>> states(cnt);
  std::cout << "states size: " << states.size() << std::endl;
  Problem problem;
  
  ceres::LossFunction* loss_function = nullptr;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  cnt = 0;
  for (auto& measure : data) {
    ceres::CostFunction* pos_cost_function = PosError::Create(measure.twr);
    problem.AddResidualBlock(pos_cost_function,
                             loss_function,
                             states[cnt].pos.data());
    if (cnt > 0) {
      ceres::CostFunction* pred_cost_function = PredictionError::Create(measure.acc, measure.omega);
      problem.AddResidualBlock(pred_cost_function,
                               loss_function,
                               states[cnt - 1].pos.data(),
                               states[cnt - 1].vel.data(),
                               states[cnt - 1].q.coeffs().data(),
                               states[cnt].pos.data(),
                               states[cnt].vel.data(),
                               states[cnt].q.coeffs().data());
      problem.SetParameterization(states[cnt - 1].q.coeffs().data(),
                                  quaternion_local_parameterization);      
      problem.SetParameterization(states[cnt].q.coeffs().data(),
                                  quaternion_local_parameterization);      

      if (cnt == 1) {
        // fix first state's velocity and pose constant
        // states[cnt - 1].vel << 0, 0, 0;
        // states[cnt - 1].q.w() = 0;
        // states[cnt - 1].q.vec() << 0, 0, 0;
        // problem.SetParameterization(states[cnt - 1].vel.data(), 
        //                             new ceres::SubsetParameterization(3, {0, 0, 0}));
        problem.SetParameterization(states[cnt - 1].q.coeffs().data(), 
                                    new ceres::SubsetParameterization(4, {1, 0, 0, 0}));
      } 
    }

    cnt++;
    if (cnt >= states.size()) {
      break;
    }
  } 

  // TODO Solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}