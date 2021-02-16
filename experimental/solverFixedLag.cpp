#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <pangolin/pangolin.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using Eigen::Isometry3d;
using Eigen::Vector3d;
using Eigen::Quaterniond;
using Eigen::Matrix3d;
using std::vector;

const double DT = 1.0 / 18;
// const Eigen::Vector3d GRAVITY{0, 0, 0};
const Eigen::Vector3d GRAVITY{0, 0, -9.8};

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);
void DrawTrajectoryComparison(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>,
                              vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);
struct State {
  Vector3d pos = Vector3d::Random(); 
  Vector3d vel = Vector3d::Random();  
  Quaterniond q = Quaterniond::UnitRandom(); 
  Vector3d bias = Vector3d::Zero(); 
  State() {}

  State(Vector3d& pos, Vector3d& vel, Quaterniond& q, Vector3d& bias)
    : pos(pos), vel(vel), q(q), bias(bias) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct Measurement{
  Matrix3d Rwr;
  Quaterniond qwr;
  Vector3d twr;
  Vector3d acc;
  Vector3d omega; 

  Measurement(Matrix3d Rwr,                
              Quaterniond qwr,
              Vector3d twr,
              Vector3d acc,
              Vector3d omega)
    : Rwr(Rwr), qwr(qwr), twr(twr), acc(acc), omega(omega) {}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct StateError {
  StateError(const Vector3d& pos_state,
						 const Vector3d& vel_state,
             const Quaterniond& q_state,
             const Vector3d& bias_state,
             const Eigen::Matrix<double, 12, 12>& sqrt_cov)
    : pos_state_(pos_state), vel_state_(vel_state), q_state_(q_state), bias_state_(bias_state), sqrt_cov_(sqrt_cov) {}

  template <typename T>
  bool operator()(const T* const pos_hat_ptr,
                  const T* const vel_hat_ptr,
                  const T* const q_hat_ptr,
                  const T* const bias_hat_ptr,
                  T* residuals_ptr) const {   
    Eigen::Matrix<T, 12, 1> residuals;
    
    Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
    Eigen::Matrix<T, 3, 1> vel_hat(vel_hat_ptr);
    Eigen::Quaternion<T> q_hat(q_hat_ptr);
    Eigen::Matrix<T, 3, 1> bias_hat(bias_hat_ptr); 

    // pos error
    residuals.template block<3, 1>(0, 0) = pos_hat - pos_state_.template cast<T>();

    // vel error
    residuals.template block<3, 1>(3, 0) = vel_hat - vel_state_.template cast<T>();

    // quat error
    Eigen::Quaternion<T> q_delta = q_state_.conjugate().template cast<T>() * q_hat;
    residuals.template block<3, 1>(6, 0) = q_delta.vec();

    // bias error 
    residuals.template block<3, 1>(9, 0) = bias_hat - bias_state_.template cast<T>();

    // marginal factor
    residuals = sqrt_cov_ * residuals;
    for (int i = 0; i < residual_size; i++) {
      residuals_ptr[i] = residuals[i];
    }
    return true;
  } 
  
  static CostFunction* Create(const Vector3d& pos_state,
															const Vector3d& vel_state,
															const Quaterniond& q_state,
                              const Vector3d& bias_state,
                              const Eigen::Matrix<double, 12, 12>& sqrt_cov) {
    return new AutoDiffCostFunction<StateError, 12, 3, 3, 4, 3>(
      new StateError(pos_state, vel_state, q_state, bias_state, sqrt_cov));
  }

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  int residual_size = 12;
  const Vector3d pos_state_;
  const Vector3d vel_state_;
  const Quaterniond q_state_;
  const Vector3d bias_state_;
  const Eigen::Matrix<double, 12, 12> sqrt_cov_;
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
    Eigen::Quaternion<T> q_hat(q_hat_ptr);
    
    // pos error 
    Eigen::Matrix<T, 3, 1> pos_delta;
    residuals.template block<3, 1>(0, 0) = pos_hat - pos_measured_.template cast<T>();

    // quat error
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
                  const T* const bias_b_ptr,
                  const T* const pos_e_ptr,
                  const T* const vel_e_ptr,
                  const T* const q_e_ptr,
                  const T* const bias_e_ptr,
                  T* residuals_ptr) const {
    Eigen::Matrix<T, 12, 1> residuals;

    const Eigen::Matrix<T, 3, 1> pos_b(pos_b_ptr);
    const Eigen::Matrix<T, 3, 1> pos_e(pos_e_ptr);
    const Eigen::Matrix<T, 3, 1> vel_b(vel_b_ptr);
    const Eigen::Matrix<T, 3, 1> vel_e(vel_e_ptr);
    const Eigen::Quaternion<T> q_b(q_b_ptr);
    const Eigen::Quaternion<T> q_e(q_e_ptr);
    const Eigen::Matrix<T, 3, 1> bias_b(bias_b_ptr);
    const Eigen::Matrix<T, 3, 1> bias_e(bias_e_ptr);

    // pos error
    residuals.template block<3, 1>(0, 0) = pos_b + vel_b * DT - pos_e;

    // vel error
    residuals.template block<3, 1>(3, 0) = vel_b + (q_b * (acc_measured_.template cast<T>() - bias_b) - GRAVITY) * DT - vel_e;

    // quat errorsqrt_cov
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

    // bias error
    residuals.template block<3, 1>(9, 0) = bias_e - bias_b; 

    for (int i = 0; i < residual_size; i++) {
      residuals_ptr[i] = residuals[i];
    }
    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& acc_measured,
                              const Eigen::Vector3d& omega_measured) {
    return new AutoDiffCostFunction<PredictionError, 12, 3, 3, 4, 3, 3, 3, 4, 3>(
      new PredictionError(acc_measured, omega_measured));
  }

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  int residual_size = 12;
  const Eigen::Vector3d acc_measured_;
  const Eigen::Vector3d omega_measured_;
};

void print_state(const State& state) {	
  Eigen::AngleAxisd ori(state.q);

  std::cout << "state pos: \n" << state.pos << "\n"
            << "state vel: \n" << state.vel << "\n"
            << "state ori: \n" << ori.angle() << " " << ori.axis() << "\n"
            << "state bias: \n" << state.bias << std::endl;
}

class FixLagSmoother {	
public:
	std::vector<State, Eigen::aligned_allocator<State>> get_all_states() {
		return all_states;
	}

	bool step(Measurement& measurement) {
		all_states.push_back(State());	
		int state_num = all_states.size();
		int marginal_idx = state_num - wind_size;
    std::cout << "current state_num: " << state_num << "marginal_idx: " << marginal_idx << std::endl;
		
    
    Problem problem;
		ceres::LossFunction* loss_function = nullptr;
		ceres::LocalParameterization* quaternion_local_parameterization =
				new ceres::EigenQuaternionParameterization;

		ceres::CostFunction* marginal_cost_function = StateError::Create(all_states[marginal_idx].pos,
																																		 all_states[marginal_idx].vel,
																																		 all_states[marginal_idx].q,
																																		 all_states[marginal_idx].bias,
                                                                     sqrt_cov);
		problem.AddResidualBlock(marginal_cost_function,
														 loss_function,
														 all_states[marginal_idx].pos.data(),
														 all_states[marginal_idx].vel.data(),
														 all_states[marginal_idx].q.coeffs().data(),
														 all_states[marginal_idx].bias.data());
    problem.SetParameterization(all_states[marginal_idx].q.coeffs().data(),
                                quaternion_local_parameterization);      
		
		for (int i = marginal_idx + 1; i < state_num; i++) {
			ceres::CostFunction* pos_cost_function = PoseError::Create(measurement.twr,
																																 measurement.qwr); 
			problem.AddResidualBlock(pos_cost_function,
															 loss_function,
															 all_states[i].pos.data(),
															 all_states[i].q.coeffs().data());
			problem.SetParameterization(all_states[i].q.coeffs().data(),
																	quaternion_local_parameterization);      
			ceres::CostFunction* pred_cost_function = PredictionError::Create(measurement.acc, 
																																				measurement.omega);
			problem.AddResidualBlock(pred_cost_function,
															 loss_function,
															 all_states[i - 1].pos.data(),
															 all_states[i - 1].vel.data(),
															 all_states[i - 1].q.coeffs().data(),
															 all_states[i - 1].bias.data(),
															 all_states[i].pos.data(),
															 all_states[i].vel.data(),
															 all_states[i].q.coeffs().data(),
															 all_states[i].bias.data());
			problem.SetParameterization(all_states[i - 1].q.coeffs().data(),
																  quaternion_local_parameterization);      
			problem.SetParameterization(all_states[i].q.coeffs().data(),
																	quaternion_local_parameterization);      
		};
    
		ceres::Solver::Options options;
		options.max_num_iterations = 200;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		// options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << "Iteration: " << marginal_idx + 1 << "\n" << summary.FullReport() << "\n";
    print_state(all_states[marginal_idx]);

		// ceres covariance matrix estimation
		update_marginal_llt(marginal_idx + 1, problem);
	}

	bool update_marginal_llt(int marginal_idx, ceres::Problem& problem) {			
		ceres::Covariance::Options options;
		ceres::Covariance covariance(options);

		std::vector<std::pair<const double*, const double*>> covariance_blocks;
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].pos.data(), all_states[marginal_idx].pos.data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].pos.data(), all_states[marginal_idx].vel.data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].pos.data(), all_states[marginal_idx].q.coeffs().data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].pos.data(), all_states[marginal_idx].bias.data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].vel.data(), all_states[marginal_idx].vel.data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].vel.data(), all_states[marginal_idx].q.coeffs().data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].vel.data(), all_states[marginal_idx].bias.data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].q.coeffs().data(), all_states[marginal_idx].q.coeffs().data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].q.coeffs().data(), all_states[marginal_idx].bias.data()));
		covariance_blocks.push_back(std::make_pair(all_states[marginal_idx].bias.data(), all_states[marginal_idx].bias.data()));

		ceres::LocalParameterization* quaternion_local_parameterization =
				new ceres::EigenQuaternionParameterization;
    problem.SetParameterization(all_states[marginal_idx].q.coeffs().data(),
                                quaternion_local_parameterization);      
		CHECK(covariance.Compute(covariance_blocks, &problem));

    double covariance_pp[3 * 3];
    double covariance_pv[3 * 3];
    double covariance_pq[3 * 3];
    double covariance_pb[3 * 3];
    double covariance_vv[3 * 3];
    double covariance_vq[3 * 3];
    double covariance_vb[3 * 3];
    double covariance_qq[3 * 3];
    double covariance_qb[3 * 3];
    double covariance_bb[3 * 3];
		covariance.GetCovarianceBlock(all_states[marginal_idx].pos.data(), all_states[marginal_idx].pos.data(), covariance_pp);
		covariance.GetCovarianceBlock(all_states[marginal_idx].pos.data(), all_states[marginal_idx].vel.data(), covariance_pv);
		covariance.GetCovarianceBlockInTangentSpace(all_states[marginal_idx].pos.data(), all_states[marginal_idx].q.coeffs().data(), covariance_pq);
		covariance.GetCovarianceBlock(all_states[marginal_idx].pos.data(), all_states[marginal_idx].bias.data(), covariance_pb);
		covariance.GetCovarianceBlock(all_states[marginal_idx].vel.data(), all_states[marginal_idx].vel.data(), covariance_vv);
		covariance.GetCovarianceBlockInTangentSpace(all_states[marginal_idx].vel.data(), all_states[marginal_idx].q.coeffs().data(), covariance_vq);
		covariance.GetCovarianceBlock(all_states[marginal_idx].vel.data(), all_states[marginal_idx].bias.data(), covariance_vb);
		covariance.GetCovarianceBlockInTangentSpace(all_states[marginal_idx].q.coeffs().data(), all_states[marginal_idx].q.coeffs().data(), covariance_qq);
		covariance.GetCovarianceBlockInTangentSpace(all_states[marginal_idx].q.coeffs().data(), all_states[marginal_idx].bias.data(), covariance_qb);
		covariance.GetCovarianceBlock(all_states[marginal_idx].bias.data(), all_states[marginal_idx].bias.data(), covariance_bb);

		Eigen::Matrix<double, 12, 12> cov;
		cov.block<3, 3>(0, 0) = Eigen::Matrix3d(covariance_pp);
		cov.block<3, 3>(0, 3) = Eigen::Matrix3d(covariance_pv);
		cov.block<3, 3>(0, 6) = Eigen::Matrix3d(covariance_pq);
		cov.block<3, 3>(0, 9) = Eigen::Matrix3d(covariance_pb);
		cov.block<3, 3>(3, 0) = Eigen::Matrix3d(covariance_pv).transpose();
		cov.block<3, 3>(3, 3) = Eigen::Matrix3d(covariance_vv);
		cov.block<3, 3>(3, 6) = Eigen::Matrix3d(covariance_vq);
		cov.block<3, 3>(3, 9) = Eigen::Matrix3d(covariance_vb);
		cov.block<3, 3>(6, 0) = Eigen::Matrix3d(covariance_pq).transpose();
		cov.block<3, 3>(6, 3) = Eigen::Matrix3d(covariance_pv).transpose();
		cov.block<3, 3>(6, 6) = Eigen::Matrix3d(covariance_qq);
		cov.block<3, 3>(6, 9) = Eigen::Matrix3d(covariance_qb);
		cov.block<3, 3>(9, 0) = Eigen::Matrix3d(covariance_pb).transpose();
		cov.block<3, 3>(9, 3) = Eigen::Matrix3d(covariance_vb).transpose();
		cov.block<3, 3>(9, 6) = Eigen::Matrix3d(covariance_qb).transpose();
		cov.block<3, 3>(9, 9) = Eigen::Matrix3d(covariance_bb);

    std::cout << "cov: \n" << cov << std::endl;
    std::cout << "cov_inv: \n" << cov.inverse() << std::endl;

    // (x_2 - x_2_hat).T * cov_inv(H) * (x_2 - x_2_hat)
    // cov_inv = U.T * U = L * L.T
    // b = U * (x_2 - x_2_hat)
    // cost = b.T * b
    Eigen::LLT<Eigen::Matrix<double, 12, 12>> lltOfcovinv(cov.inverse()); // compute the Cholesky decomposition
    sqrt_cov = lltOfcovinv.matrixU();

    sqrt_cov.block<3, 3>(9, 9) = Matrix3d::Identity();
    std::cout << "sqrt_cov: \n" << sqrt_cov << std::endl;
    return true;
	}

	bool initialize(const int& wind_size, const State& init_state) {
		this->wind_size = wind_size;		
    all_states.push_back(init_state);
    return true;
	}

private:
	int wind_size = 2;
  Eigen::Matrix<double, 12, 12> sqrt_cov = Eigen::Matrix<double, 12, 12>::Identity();
	Eigen::Vector3d bias = Eigen::Vector3d::Zero();
	Eigen::Matrix3d state_hessian;
	Eigen::Matrix3d state_b;
	std::vector<State, Eigen::aligned_allocator<State>> all_states;
};


void save_states(const std::string& filename, 
                 std::vector<State, Eigen::aligned_allocator<State>>& states) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);

  for (auto& state : states) {
    Eigen::Matrix3d rot = state.q.matrix();
    outfile << rot << "\n" << state.pos.transpose() << "\n" << state.vel.transpose() << "\n\n";
  }
}

std::vector<Measurement, Eigen::aligned_allocator<Measurement>> readSensorData(std::string path) {
  std::vector<Measurement, Eigen::aligned_allocator<Measurement>> ret;

  std::ifstream csvFile;
  csvFile.open(path);

  std::string line;
  while(std::getline(csvFile, line)) {
    std::vector<double> row;
    // std::cout << "line:" << line << std::endl;
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
    // std::cout << "Rwr: " << Rwr << std::endl;
    // std::cout << "qwr: " << qwr.w() << " " << qwr.vec() << std::endl; 
    // std::cout << "twr: " << twr << std::endl;
    // std::cout << "acc: " << acc << std::endl;
    // std::cout << "omega: " << omega << std::endl;
    
    ret.push_back(Measurement(Rwr, qwr, twr, acc, omega));
  }

  return ret;
}


int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "missing arg for the csv file" << std::endl;
  }

  std::string path = argv[1];
  std::vector<Measurement, Eigen::aligned_allocator<Measurement>> data = readSensorData(path);    
  int cnt = data.size();
  // cnt = 15;
	int wind_size = 2;
 
  std::vector<State, Eigen::aligned_allocator<State>> gt_states(cnt);
  std::cout << "states size: " << gt_states.size() << std::endl;

  for (int i = 0; i < gt_states.size(); i++) {
    gt_states[i].pos = data[i].twr;
    gt_states[i].vel = Eigen::Vector3d::Zero();
    gt_states[i].q = data[i].qwr;
  }
  gt_states[0].vel = Eigen::Vector3d({0, 94.25, 0});
	FixLagSmoother smoother;

  Vector3d init_vel({0, 0, 0});
  Vector3d init_bias({0, 0, 0});
  State init_state(data[0].twr, init_vel, data[0].qwr, init_bias);
	smoother.initialize(wind_size, init_state);

	for (int i = wind_size - 1; i < cnt; i++) {
		smoother.step(data[i]);
	}
  
  std::vector<State, Eigen::aligned_allocator<State>> states = smoother.get_all_states();

  std::string est_filename = "./results/increm_states.txt";
  save_states(est_filename, states);


  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> gt_poses;
  for (int i = 0; i < cnt; i++) {
    Isometry3d Twr(states[i].q.matrix());
    Twr.pretranslate(states[i].pos / 20); // manually divided by 20 to zoom out
    poses.push_back(Twr);
  }
  for (int i = 0; i < cnt; i++) {
    Isometry3d Twr(gt_states[i].q.matrix());
    Twr.pretranslate(gt_states[i].pos / 20); // manually divided by 20 to zoom out
    gt_poses.push_back(Twr);
  }
  DrawTrajectoryComparison(poses, gt_poses);
	return 0;
}
void DrawTrajectoryComparison(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses,
                              vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> gt_poses) {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      // 画每个位姿的三个坐标轴
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // 画出连线
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(1.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < gt_poses.size(); i++) {
      // 画每个位姿的三个坐标轴
      Vector3d Ow = gt_poses[i].translation();
      Vector3d Xw = gt_poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = gt_poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = gt_poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // 画出连线
    for (size_t i = 0; i < gt_poses.size(); i++) {
      glColor3f(0.0, 1.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = gt_poses[i], p2 = gt_poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      // 画每个位姿的三个坐标轴
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    // 画出连线
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }
}
