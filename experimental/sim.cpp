#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/navigation/Scenario.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace gtsam;


/* ************************************************************************* */
int main(int argc, char* argv[]) {
  // Start with a camera on x-axis looking at origin
  double radius = 30;
  const Point3 up(0, 0, 1), target(0, 0, 0);
  const Point3 position(radius, 0, 0);
  const auto camera = PinholeCamera<Cal3_S2>::Lookat(position, target, up);
  const auto pose_0 = camera.pose();
  // Now, create a constant-twist scenario that makes the camera orbit the
  // origin
  double angular_velocity = M_PI,  // rad/sec
      delta_t = 1.0 / 18;          // makes for 10 degrees per step
  Vector3 angular_velocity_vector(0, -angular_velocity, 0);
  Vector3 linear_velocity_vector(radius * angular_velocity, 0, 0);
  auto scenario = ConstantTwistScenario(angular_velocity_vector,
                                        linear_velocity_vector, pose_0);


  std::ofstream f;
  std::string path = "data.csv";
  f.open(path);
  f << std::setprecision(15);
  // Simulate poses and imu measurements, adding them to the factor graph
  for (size_t i = 0; i < 36; ++i) {
    double t = i * delta_t;
    auto pose = scenario.pose(t);

    const Vector3 GRAVITY{0,0,0};
    Vector3 measuredAcc = scenario.acceleration_b(t) -
                            scenario.rotation(t).transpose() * GRAVITY;
    Vector3 measuredOmega = scenario.omega_b(t);

    std::cout << "pose: " << pose << std::endl;
    // std::cout << "measuredAcc: " << measuredAcc << std::endl;
    // std::cout << "measuredOmega: " << measuredOmega << std::endl;

    Matrix m = pose.matrix();
    for(int r = 0; r < 4; ++r) {
      for(int c = 0; c < 4; ++c) {
        f << m(r,c);
        f << ",";
      }
    }

    for(int j = 0; i < 3; ++i) {
      f << measuredAcc(j);
      f << ",";
    }

    for(int j = 0; i < 3; ++i) {
      f << measuredOmega(j);
      f << ",";
    }

    f << "\n";
  }
  f.close();

  std::cout << "save file at :" << path << std::endl;  


  return 0;
}
/* ************************************************************************* */
