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

const double DT = 1.0 / 18;
// const Eigen::Vector3d GRAVITY{0, 0, 0};
const Eigen::Vector3d GRAVITY{0, 0, -9.8};

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

struct PositionError {
	PositionError(const Eigen::Vector3d& pos_measured) 
		: pos_measured_(pos_measured) {}

	template <typename T>
	bool operator()(const T* const pos_hat_ptr,
									T* residuals_ptr) const {
		Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
		Eigen::Matrix<T, 3, 1> pos_delta = pos_hat - pos_measured_.template cast<T>();	

    for (int i = 0; i < 3; i++) {
      residuals_ptr[i] = pos_delta[i];
    }
		return true;
	}

	static CostFunction* Create(const Eigen::Vector3d& pos_measured) {
		return new AutoDiffCostFunction<PositionError, 3, 3>(
			new PositionError(pos_measured)
		);
	}

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
	const Eigen::Vector3d pos_measured_;
};

struct PoseError {
  PoseError(const Eigen::Vector3d& pos_measured,
            const Eigen::Quaterniond& q_measured)
    : pos_measured_(pos_measured), q_measured_(q_measured) {}

  template <typename T>
  bool operator()(const T* const pos_hat_ptr,
                  const T* const q_hat_ptr,
                  T* residuals_ptr) const {   
    Eigen::Matrix<T, 6, 1> residuals;
    
    Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
    Eigen::Matrix<T, 3, 1> pos_delta;
    residuals.template block<3, 1>(0, 0) = pos_hat - pos_measured_.template cast<T>();

    Eigen::Quaternion<T> q_hat(q_hat_ptr);
    Eigen::Quaternion<T> q_delta = q_measured_.conjugate().template cast<T>() * q_hat;
    residuals.template block<3, 1>(3, 0) = q_delta.vec();
    for (int i = 0; i < 6; i++) {
      residuals_ptr[i] = residuals[i];
    }
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
                  const T* const q_e_ptr,
                  const T* const bias_ptr,
                  T* residuals_ptr) const {
    Eigen::Matrix<T, 9, 1> residuals;

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
    residuals.template block<3, 1>(6, 0) = q_delta.vec();

    // vel error
    const Eigen::Matrix<T, 3, 1> bias(bias_ptr);
    const Eigen::Matrix<T, 3, 1> vel_b(vel_b_ptr);
    const Eigen::Matrix<T, 3, 1> vel_e(vel_e_ptr);
    residuals.template block<3, 1>(3, 0) = vel_b + (q_b * (acc_measured_.template cast<T>() - bias) - GRAVITY) * DT - vel_e;

    // pos error
    const Eigen::Matrix<T, 3, 1> pos_b(pos_b_ptr);
    const Eigen::Matrix<T, 3, 1> pos_e(pos_e_ptr);
    residuals.template block<3, 1>(0, 0) = pos_b + vel_b * DT - pos_e;

    for (int i = 0; i < 9; i++) {
      residuals_ptr[i] = residuals[i];
    }
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

struct OriPredictionError{
  OriPredictionError(const Eigen::Vector3d& acc_measured,
                  const Eigen::Vector3d& omega_measured)
    : acc_measured_(acc_measured), omega_measured_(omega_measured) {}

  template <typename T>
  bool operator()(const T* const q_b_ptr,
                  const T* const q_e_ptr,
                  T* residuals_ptr) const {
    Eigen::Matrix<T, 3, 1> residuals;

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
    residuals.template block<3, 1>(0, 0) = T(2.0) * q_delta.vec();

    for (int i = 0; i < 3; i++) {
      residuals_ptr[i] = residuals[i];
    }
    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& acc_measured,
                              const Eigen::Vector3d& omega_measured) {
    return new AutoDiffCostFunction<OriPredictionError, 3, 4, 4>(
      new OriPredictionError(acc_measured, omega_measured));
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
      // std::cout << "field: " << field << std::endl;
      row.push_back(std::stod(field));
    }  
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

  return ret;
}

void output_pose(const Eigen::Vector3d& pos, 
								 const Eigen::Quaterniond& q) {
	Eigen::AngleAxisd ori(q);

	std::cout << "Location: " << pos << std::endl;
	std::cout << "Orientation: " << ori.angle() << " * " << std::endl << ori.axis() << std::endl;
}

void output_measurement(const Measurement& data) {
  std::cout << "\nData State: \n" << "R: \n" << data.Rwr << "\nt: \n" << data.twr \
            << "\nacc: \n" << data.acc << "\nomega: \n" << data.omega << std::endl;
} 

void save_states(const std::string& filename, 
                 std::vector<State, Eigen::aligned_allocator<State>>& states) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);

  for (auto& state : states) {
    Eigen::Matrix3d rot = state.q.matrix();
    outfile << rot << "\n" << state.pos.transpose() << "\n" << state.vel.transpose() << "\n\n";
  }
} 


double abs_pos_error(const std::vector<State, Eigen::aligned_allocator<State>>& states,
                     const std::vector<State, Eigen::aligned_allocator<State>>& gt_states) {
  double err = 0.0;
  std::cout << "abs_pos_err by step: ";
  for (int i = 0; i < states.size(); i++) {
    double step_err = (states[i].pos - gt_states[i].pos).norm();
    err += step_err;
    std::cout << step_err << " ";
  }
  std::cout << std::endl;
  return err;
}

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "missing arg for the csv file" << std::endl;
  }

  std::string path = argv[1];
  std::vector<Measurement, Eigen::aligned_allocator<Measurement>> data = readSensorData(path);    
  
  if (false) {
    output_measurement(data[0]);
    output_measurement(data[1]);

    Eigen::Vector3d rotated = data[0].omega * DT;
    double angle = rotated.norm();
    Eigen::Vector3d axis = rotated.normalized();
    Eigen::Quaterniond q_add(Eigen::AngleAxisd(angle, axis));
    Eigen::Quaterniond q_new = data[0].qwr * q_add;
    Eigen::Matrix3d R_new = q_new.normalized().toRotationMatrix();

    std::cout << "R_new: \n" << R_new << std::endl;
    
    Eigen::Vector3d vel_add = data[0].qwr * data[0].acc * DT;
    std::cout << "vel_new: \n" << vel_add << std::endl;
    return 0;
  }
  
  int cnt = data.size();
  // int cnt = 3;

  // Eigen::Vector3d bias = Eigen::Vector3d::Random();
  Eigen::Vector3d bias;
  bias << 0, 0, 0;
  std::vector<State, Eigen::aligned_allocator<State>> states(cnt);
  std::vector<State, Eigen::aligned_allocator<State>> gt_states(cnt);
  std::cout << "states size: " << states.size() << std::endl;

  for (int i = 0; i < gt_states.size(); i++) {
    gt_states[i].pos = data[i].twr;
    gt_states[i].vel = Eigen::Vector3d::Zero();
    gt_states[i].q = data[i].qwr;
  }

  Problem problem;
  
  ceres::LossFunction* loss_function = nullptr;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (int i = 0; i < cnt; i++) {
    ceres::CostFunction* position_cost_function = PositionError::Create(data[i].twr);
    problem.AddResidualBlock(position_cost_function,
                             loss_function,
                             states[i].pos.data());
    // ceres::CostFunction* pos_cost_function = PoseError::Create(data[i].twr, data[i].qwr);
    // problem.AddResidualBlock(pos_cost_function,
    //                          loss_function,
    //                          states[i].pos.data(),
    //                          states[i].q.coeffs().data());
    // problem.SetParameterization(states[i].q.coeffs().data(),
    //                             quaternion_local_parameterization);      
    if (i > 0) {
      ceres::CostFunction* pred_cost_function = PredictionError::Create(data[i].acc, data[i].omega);
      problem.AddResidualBlock(pred_cost_function,
                               loss_function,
                               states[i - 1].pos.data(),
                               states[i - 1].vel.data(),
                               states[i - 1].q.coeffs().data(),
                               states[i].pos.data(),
                               states[i].vel.data(),
                               states[i].q.coeffs().data(),
                               bias.data());
      // ceres::CostFunction* pred_cost_function = OriPredictionError::Create(data[i].acc, data[i].omega);
      // problem.AddResidualBlock(pred_cost_function,
      //                          loss_function,
      //                          states[i - 1].q.coeffs().data(),
      //                          states[i].q.coeffs().data());
      problem.SetParameterization(states[i - 1].q.coeffs().data(),
                                  quaternion_local_parameterization);      
      problem.SetParameterization(states[i].q.coeffs().data(),
                                  quaternion_local_parameterization);      
    }
  } 

  // set bias to constant
  // problem.SetParameterBlockConstant(bias.data());
  // Eigen::Quaterniond q_noise(Eigen::AngleAxisd(0.1, Eigen::Vector3d::Random().normalized()));
  // states[0].q = data[0].qwr * q_noise;
  // states[0].q = data[0].qwr;
  // problem.SetParameterBlockConstant(states[0].q.coeffs().data());

  ceres::Solver::Options options;
	options.max_num_iterations = 200;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  
  if (false) {
    output_measurement(data[0]);
    output_measurement(data[1]);
    output_measurement(data[2]);

    for (int i = 0; i < states.size(); i++) {
      std::cout << "Estimated State:" << i << " \n" << "R: \n" << states[i].q.matrix()
                                              << "\nt: \n" << states[i].pos
                                              << "\nvel: \n" << states[i].vel << std::endl;
    }
  }

  std::string est_filename = "./results/est_no_ori_measure_nofixfirst_states.txt";
  // std::string est_filename = "./results/est_position_predic_ori_states.txt";
  save_states(est_filename, states);
  std::string gt_filename = "./results/gt_states.txt";
  save_states(gt_filename, gt_states);
  
  double err = abs_pos_error(states, gt_states); 
  std::cout << "absolute position error: " << err << std::endl;
  return 0;
}