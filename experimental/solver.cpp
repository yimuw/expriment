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
  Eigen::Quaterniond qwr;
  Eigen::Vector3d twr;
  Eigen::Vector3d acc;
  Eigen::Vector3d omega; 

  Measurement(Eigen::Matrix3d Rwr,                
              Eigen::Quaterniond qwr,
              Eigen::Vector3d twr,
              Eigen::Vector3d acc,
              Eigen::Vector3d omega)
    : Rwr(Rwr), qwr(qwr), twr(twr), acc(acc), omega(omega) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseError {
  PoseError(const Eigen::Vector3d& pos_measured,
            const Eigen::Quaterniond& q_measured)
    : pos_measured_(pos_measured), q_measured_(q_measured) {}

  template <typename T>
  bool operator()(const T* const pos_hat_ptr,
                  const T* const q_hat_prt,
                  T* residuals_ptr) const {      
    Eigen::Matrix<T, 6, 1> residuals(residuals_ptr);
    Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
    residuals.block(0, 0, 3, 1) = pos_hat - pos_measured_.template cast<T>();

    Eigen::Quaternion<T> q_hat(q_hat_prt);
    Eigen::Quaternion<T> q_delta = q_measured_.conjugate().template cast<T>() * q_hat;
    residuals.block(3, 0, 3, 1) = q_delta.vec();
    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& pos_measured,
                              const Eigen::Quaterniond& q_measured) {
    return new AutoDiffCostFunction<PoseError, 6, 3, 4>(
      new PoseError(pos_measured, q_measured));
  }

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const Eigen::Vector3d pos_measured_;
  const Eigen::Quaterniond q_measured_;
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
                  const T* const bias_ptr,
                  const T* const q_e_ptr,
                  T* residuals_ptr) const {
    Eigen::Matrix<T, 9, 1> residuals(residuals_ptr);

    // quat error
    const Eigen::Quaternion<T> q_b(q_b_ptr);
    const Eigen::Quaternion<T> q_e(q_e_ptr);
    
    Eigen::Quaternion<T> q_new;
    Eigen::Quaternion<T> q_add; 
    
    // // https://gamedev.stackexchange.com/questions/108920/applying-angular-velocity-to-quaternion 
    // Eigen::Quaternion<T> q_omega;
    // // q_omega.w() = 0;
    // q_omega.vec() = omega_measured_.template cast<T>() * DT * 0.5;
    // q_add = q_omega * q_b;
    // q_new.w() = q_b.w() + q_add.w();
    // q_new.vec() = q_b.vec() + q_add.vec();

    Eigen::Vector3d rotated = omega_measured_ * DT;
    double angle = rotated.norm();
    Eigen::Vector3d axis = rotated.normalized();
    q_add = Eigen::AngleAxisd(angle, axis).template cast<T>();
    q_new = q_b * q_add;
  
    Eigen::Quaternion<T> q_delta = q_e.conjugate() * q_new;
    residuals.block(6, 0, 3, 1) = q_delta.vec();

    // vel error
    const Eigen::Matrix<T, 3, 1> bias(bias_ptr);
    const Eigen::Matrix<T, 3, 1> vel_b(vel_b_ptr);
    const Eigen::Matrix<T, 3, 1> vel_e(vel_e_ptr);
    residuals.block(3, 0, 3, 1) = vel_b + (q_b * (acc_measured_.template cast<T>() - bias) - GRAVITY) * DT - vel_e;

    // pos error
    const Eigen::Matrix<T, 3, 1> pos_b(pos_b_ptr);
    const Eigen::Matrix<T, 3, 1> pos_e(pos_e_ptr);
    residuals.block(0, 0, 3, 1) = pos_b + vel_b * DT - pos_e;

    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& acc_measured,
                              const Eigen::Vector3d& omega_measured) {
    return new AutoDiffCostFunction<PredictionError, 9, 3, 3, 4, 3, 3, 4, 3>(
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
    
      Eigen::Matrix3d Rwr;
      Eigen::Quaterniond qwr;
      Eigen::Vector3d twr;
      Eigen::Vector3d acc;
      Eigen::Vector3d omega; 
      Rwr << row[0], row[1], row[2],
            row[4], row[5], row[6],
            row[8], row[9], row[10];
      qwr = Rwr;
      twr << row[3], row[7], row[11];
      acc << row[16], row[17], row[18];
      omega << row[19], row[20], row[21];
      std::cout << "Rwr: " << Rwr << std::endl;
      std::cout << "qwr: " << qwr.w() << " " << qwr.vec() << std::endl; 
      std::cout << "twr: " << twr << std::endl;
      std::cout << "acc: " << acc << std::endl;
      std::cout << "omega: " << omega << std::endl;
      
      ret.push_back(Measurement(Rwr, qwr, twr, acc, omega));
    }
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

  Eigen::Vector3d bias = Eigen::Vector3d::Random();
  std::vector<State, Eigen::aligned_allocator<State>> states(cnt);
  std::cout << "states size: " << states.size() << std::endl;
  Problem problem;
  
  ceres::LossFunction* loss_function = nullptr;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  cnt = 0;
  for (auto& measure : data) {
    ceres::CostFunction* pos_cost_function = PoseError::Create(measure.twr, measure.qwr);
    problem.AddResidualBlock(pos_cost_function,
                             loss_function,
                             states[cnt].pos.data(),
                             states[cnt].q.coeffs().data());
    problem.SetParameterization(states[cnt].q.coeffs().data(),
                                quaternion_local_parameterization);      
    if (cnt > 0) {
      ceres::CostFunction* pred_cost_function = PredictionError::Create(measure.acc, measure.omega);
      problem.AddResidualBlock(pred_cost_function,
                               loss_function,
                               states[cnt - 1].pos.data(),
                               states[cnt - 1].vel.data(),
                               states[cnt - 1].q.coeffs().data(),
                               states[cnt].pos.data(),
                               states[cnt].vel.data(),
                               states[cnt].q.coeffs().data(),
                               bias.data());
      problem.SetParameterization(states[cnt - 1].q.coeffs().data(),
                                  quaternion_local_parameterization);      
      problem.SetParameterization(states[cnt].q.coeffs().data(),
                                  quaternion_local_parameterization);      
    }

    cnt++;
    if (cnt >= states.size()) {
      break;
    }
  } 

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}