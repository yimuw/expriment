#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <pangolin/pangolin.h>

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using Eigen::Isometry3d;
using Eigen::Vector3d;
using Eigen::Quaterniond;
using Eigen::Matrix3d;
using Eigen::cos;
using Eigen::sin;
using std::vector;

const double DT = 1.0 / 18;
// const Eigen::Vector3d GRAVITY{0, 0, 0};
const Eigen::Vector3d GRAVITY{0, 0, -9.8};

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);
void DrawTrajectoryComparison(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>,
                              vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

struct State {
  Eigen::Vector3d pos = Eigen::Vector3d::Random(); // position 
  Eigen::Vector3d vel = Eigen::Vector3d::Random(); // velocity
  Eigen::Vector3d aaxis = Eigen::Vector3d::Random(); // Lie Algebra (Angle axis)
  // Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom(); // pose Qwr

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
            const Eigen::Matrix3d& R_measured)
    : pos_measured_(pos_measured), R_measured_(R_measured) {}

  template <typename T>
  bool operator()(const T* const pos_hat_ptr,
                  const T* const aaxis_hat_ptr,
                  T* residuals_ptr) const {   
    Eigen::Matrix<T, 6, 1> residuals;
    
    Eigen::Matrix<T, 3, 1> pos_hat(pos_hat_ptr);
    Eigen::Matrix<T, 3, 1> pos_delta;
    residuals.template block<3, 1>(0, 0) = pos_hat - pos_measured_.template cast<T>();

    Eigen::Matrix<T, 3, 1> axis_hat(aaxis_hat_ptr);
    Eigen::AngleAxisd aaxis_m(R_measured_);

    T angle_hat = -axis_hat.norm();
    axis_hat.normalize();
    T sin_half_hat = sin(angle_hat / T(2.0));
    T cos_half_hat = cos(angle_hat / T(2.0));

    T angle_m = T(aaxis_m.angle());
    Eigen::Matrix<T, 3, 1> axis_m = aaxis_m.axis().template cast<T>();
    T sin_half_m = sin(angle_m / T(2.0));
    T cos_half_m = cos(angle_m / T(2.0));

    T cos_half_delta = cos_half_hat * cos_half_m - sin_half_hat * sin_half_m * axis_hat.dot(axis_m);
    Eigen::Matrix<T, 3, 1> axis_delta = sin_half_hat * cos_half_m * axis_hat + 
                                        cos_half_hat * sin_half_m * axis_m + 
                                        sin_half_hat * sin_half_m * axis_hat.cross(axis_m);

    residuals.template block<3, 1>(3, 0) = axis_delta;
    for (int i = 0; i < 6; i++) {
      residuals_ptr[i] = residuals[i];
    }
    return true;
  } 
  
  static CostFunction* Create(const Eigen::Vector3d& pos_measured,
                              const Eigen::Matrix3d& R_measured) {
    return new AutoDiffCostFunction<PoseError, 6, 3, 3>(
      new PoseError(pos_measured, R_measured));
  }

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const Eigen::Vector3d pos_measured_;
  const Eigen::Matrix3d R_measured_;
};

struct PredictionError{
  PredictionError(const Eigen::Vector3d& acc_measured,
                  const Eigen::Vector3d& omega_measured)
    : acc_measured_(acc_measured), omega_measured_(omega_measured) {}

  template <typename T>
  bool operator()(const T* const pos_b_ptr,
                  const T* const vel_b_ptr,
                  const T* const aaxis_b_ptr,
                  const T* const pos_e_ptr,
                  const T* const vel_e_ptr,
                  const T* const aaxis_e_ptr,
                  const T* const bias_ptr,
                  T* residuals_ptr) const {
    Eigen::Matrix<T, 9, 1> residuals;

    // rot error
    Eigen::Matrix<T, 3, 1> axis_b(aaxis_b_ptr);
    Eigen::Matrix<T, 3, 1> axis_e(aaxis_e_ptr);
    Eigen::Matrix<T, 3, 1> axis_d = omega_measured_.template cast<T>() * DT;

    int choice = 1;
    if (choice == 0) {
      // Eigen::AngleAxis<T> aaxis_b(axis_b.norm(), axis_b.normalized());
      // Eigen::AngleAxis<T> aaxis_e(axis_e.norm(), axis_e.normalized());
      // Eigen::AngleAxis<T> aaxis_d(axis_d.norm(), axis_d.normalized());

      // Eigen::AngleAxis<T> aaxis_delta(aaxis_d.inverse() * aaxis_b.inverse() * aaxis_e);
      // residuals.template block<3, 1>(6, 0) = aaxis_delta.angle() * aaxis_delta.axis();
    } else if (choice == 1) {
    }
    // Reference: https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
    T angle_b = -axis_b.norm();
    axis_b.normalize();
    T sin_half_b = sin(angle_b / T(2));
    T cos_half_b = cos(angle_b / T(2));

    T angle_e = axis_e.norm();
    axis_e.normalize();
    T sin_half_e = sin(angle_e / T(2));
    T cos_half_e = cos(angle_e / T(2));

    T cos_half_be = cos_half_b * cos_half_e - sin_half_b * sin_half_e * axis_b.dot(axis_e);
    Eigen::Matrix<T, 3, 1> axis_be = sin_half_b * cos_half_e * axis_b + cos_half_b * sin_half_e * axis_e + sin_half_b * sin_half_e * axis_b.cross(axis_e);
    T sin_half_be = axis_be.norm();
    axis_be.normalize();

    T angle_d = -axis_d.norm();
    axis_d.normalize();
    T sin_half_d = sin(angle_d / T(2));
    T cos_half_d = cos(angle_d / T(2));

    T cos_half_delta = cos_half_d * cos_half_be - sin_half_d * sin_half_be * axis_d.dot(axis_be);
    Eigen::Matrix<T, 3, 1> axis_delta = sin_half_d * cos_half_be * axis_d + cos_half_d * sin_half_be * axis_be + sin_half_d * sin_half_be * axis_d.cross(axis_be);

    residuals.template block<3, 1>(6, 0) = axis_delta;
  
    angle_b = -angle_b;
    T sin_b = sin(angle_b);
    T cos_b = cos(angle_b);
    Eigen::Matrix<T, 3, 3> axis_b_cross = Eigen::Matrix<T, 3, 3>::Zero();
    axis_b_cross(0, 1) = -axis_b[2]; 
    axis_b_cross(1, 0) = axis_b[2]; 
    axis_b_cross(0, 2) = axis_b[1]; 
    axis_b_cross(2, 0) = -axis_b[1]; 
    axis_b_cross(1, 2) = -axis_b[0]; 
    axis_b_cross(2, 1) = axis_b[0]; 
    
    Eigen::Matrix<T, 3, 3> rot_b = cos_b * Eigen::Matrix<T, 3, 3>::Identity() +
                                   (T(1.0) - cos_b) * axis_b * axis_b.transpose() + 
                                   sin_b * axis_b_cross;
    // vel error
    const Eigen::Matrix<T, 3, 1> bias(bias_ptr);
    const Eigen::Matrix<T, 3, 1> vel_b(vel_b_ptr);
    const Eigen::Matrix<T, 3, 1> vel_e(vel_e_ptr);
    residuals.template block<3, 1>(3, 0) = vel_b + (rot_b * (acc_measured_.template cast<T>() - bias) - GRAVITY) * DT - vel_e;

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
    return new AutoDiffCostFunction<PredictionError, 9, 3, 3, 3, 3, 3, 3, 3>(
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
    Eigen::AngleAxisd temp(data[i].Rwr);
    gt_states[i].aaxis = temp.angle() * temp.axis();
  }

  Problem problem;
  
  ceres::LossFunction* loss_function = nullptr;

  for (int i = 0; i < cnt; i++) {
    ceres::CostFunction* pos_cost_function = PoseError::Create(data[i].twr, data[i].Rwr);
    problem.AddResidualBlock(pos_cost_function,
                             loss_function,
                             states[i].pos.data(),
                             states[i].aaxis.data());
    if (i > 0) {
      ceres::CostFunction* pred_cost_function = PredictionError::Create(data[i].acc, data[i].omega);
      problem.AddResidualBlock(pred_cost_function,
                               loss_function,
                               states[i - 1].pos.data(),
                               states[i - 1].vel.data(),
                               states[i - 1].aaxis.data(),
                               states[i].pos.data(),
                               states[i].vel.data(),
                               states[i].aaxis.data(),
                               bias.data());
    }
  } 

  ceres::Solver::Options options;
	options.max_num_iterations = 200;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  
  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> gt_poses;
  for (int i = 0; i < cnt; i++) {
    Eigen::AngleAxisd temp(states[i].aaxis.norm(), states[i].aaxis.normalized());
    Isometry3d Twr(temp.matrix());
    Twr.pretranslate(states[i].pos / 20); // manually divided by 20 to zoom out
    poses.push_back(Twr);
  }
  for (int i = 0; i < cnt; i++) {
    Eigen::AngleAxisd temp(gt_states[i].aaxis.norm(), gt_states[i].aaxis.normalized());
    Isometry3d Twr(temp.matrix());
    Twr.pretranslate(gt_states[i].pos / 20); // manually divided by 20 to zoom out
    gt_poses.push_back(Twr);
  }

  double err = abs_pos_error(states, gt_states); 
  std::cout << "absolute position error: " << err << std::endl;
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