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
		// Eigen::Matrix<T, 3, 1> residuals(residuals_ptr);
		// residuals.template block<3, 1>(0, 0) = pos_hat - pos_measured_.template cast<T>();
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
    // Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
		// Eigen::Matrix<T, 3, 1> pos_delta = pos_hat - pos_measured_.template cast<T>();

    // Eigen::Quaternion<T> q_hat(q_hat_ptr);
    // Eigen::Quaternion<T> q_delta = q_hat.conjugate() * q_measured_.template cast<T>();
		// residuals_ptr[0] = pos_delta.norm() + T(2.0) * q_delta.vec().norm();

		Eigen::Matrix<T, 6, 1> residuals;
    Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
		residuals.template block<3, 1>(0, 0) = pos_hat - pos_measured_.template cast<T>();

    Eigen::Quaternion<T> q_hat(q_hat_ptr);
    Eigen::Quaternion<T> q_delta = q_hat.conjugate() * q_measured_.template cast<T>();
		residuals.template block<3, 1>(3, 0) = T(2.0) * q_delta.vec();
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

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "missing arg for the csv file" << std::endl;
  }

  std::string path = argv[1];
  std::vector<Measurement, Eigen::aligned_allocator<Measurement>> data = readSensorData(path);    

   
  output_measurement(data[0]);
  output_measurement(data[1]);
  output_measurement(data[2]);

  return 0; 

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

  Eigen::Vector3d pos1 = Eigen::Vector3d::Random(); // position 
	Eigen::Quaterniond q1 = Eigen::Quaterniond::UnitRandom(); // orientation

	std::cout << "Initial state: " << std::endl;
	output_pose(pos1, q1);

	ceres::CostFunction* pose_cost_function = PoseError::Create(data[0].twr, 
																															data[0].qwr);
	problem.AddResidualBlock(pose_cost_function, 
													 loss_function, 
													 pos1.data(), 
													 q1.coeffs().data());
	
	// for (auto& measure : data) {
  //   ceres::CostFunction* pos_cost_function = PoseError::Create(measure.twr, measure.qwr);
  //   problem.AddResidualBlock(pos_cost_function,
  //                            loss_function,
  //                            states[cnt].pos.data(),
  //                            states[cnt].q.coeffs().data());
  //   problem.SetParameterization(states[cnt].q.coeffs().data(),
  //                               quaternion_local_parameterization);      

	// 	// ceres::CostFunction* position_cost_function = PositionError::Create(measure.twr);
	// 	// problem.AddResidualBlock(position_cost_function, loss_function, states[cnt].pos.data());
	//   cnt++;
  //   if (cnt >= states.size()) {
  //     break;
  //   }
  // } 

  ceres::Solver::Options options;
	options.max_num_iterations = 20;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

	std::cout << "Ground Truth: " << std::endl;
	output_pose(data[0].twr, data[0].qwr);
	std::cout << "Final State: " << std::endl;
	output_pose(pos1, q1);

  output_measurement(data[0]);
  output_measurement(data[1]);
  output_measurement(data[2]);

  return 0;
}