#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>
#include <fstream>

// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

struct State{
  Eigen::Matrix3d Rwr;
  Eigen::Vector3d twr;
  Eigen::Vector3d vel;

  State(Eigen::Matrix3d& Rwr, Eigen::Vector3d& twr, Eigen::Vector3d& vel)
    : Rwr(Rwr), twr(twr), vel(vel) {}
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// path to trajectory file
string est_traj_file = "./est_states.txt";
string gt_traj_file = "./gt_states.txt";

std::vector<std::vector<double>> readSensorData(std::string);
std::vector<State, Eigen::aligned_allocator<State>> readStates(std::string);
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);
void DrawTrajectoryComparison(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>,
                              vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {
  // if(argc < 2) {
  //   std::cout << "missing arg for the csv file" << std::endl;
  // }

  // std::string path = argv[1];
  // std::vector<std::vector<double>> data = readSensorData(path);
  
  std::vector<State, Eigen::aligned_allocator<State>> states = readStates(est_traj_file);
  std::vector<State, Eigen::aligned_allocator<State>> gt_states = readStates(gt_traj_file);
  
  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> gt_poses;

  // for (std::vector<double> &state : data) {
  //   cout << "state size: " << state.size() << endl;
  //   for (int i = 0; i < 16; i++) {
  //     cout << state[i] << " ";
  //   }
  //   cout << endl;
  //   Matrix3d Rwr;
  //   Vector3d twr;
  //   Rwr << state[0], state[1], state[2],
  //          state[4], state[5], state[6],
  //          state[8], state[9], state[10];
  //   twr << state[3], state[7], state[11];
  //   cout << "Rwr: " << Rwr << endl;
  //   cout << "twr: " << twr << endl;
  //   Isometry3d Twr(Rwr);
  //   Twr.pretranslate(twr);
  //   poses.push_back(Twr);
  // } 

  // Isometry3d Twr(Eigen::Matrix3d::Identity());
  // Twr.pretranslate(Eigen::Vector3d::Zero());
  // poses.push_back(Twr);
  // gt_poses.push_back(Twr);
  
  int cnt = states.size();

  for (int i = 0; i < cnt; i++) {
    Isometry3d Twr(states[i].Rwr);
    Twr.pretranslate(states[i].twr);
    poses.push_back(Twr);
  }
  for (int i = 0; i < cnt; i++) {
    Isometry3d Twr(gt_states[i].Rwr);
    Twr.pretranslate(gt_states[i].twr);
    gt_poses.push_back(Twr);
  }
  
  /*
  ifstream fin(trajectory_file);
  if (!fin) {
    cout << "cannot find trajectory file at " << trajectory_file << endl;
    return 1;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));
    poses.push_back(Twr);
  }

  */
  cout << "read total " << poses.size() << " pose entries" << endl;

  // draw trajectory in pangolin
  DrawTrajectoryComparison(poses, gt_poses);
  return 0;
}

std::vector<std::vector<double>> readSensorData(std::string path) {
  std::vector<std::vector<double>> ret;

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
    ret.push_back(row);
  }
  return ret;
}

std::vector<State, Eigen::aligned_allocator<State>> readStates(std::string filename) {
  std::vector<State, Eigen::aligned_allocator<State>> states;
  std::ifstream file(filename);
  if (!file) {
    cout << "Error opening the file: " << filename << endl;
    return states; 
  }
  cout << "Start reading: " << filename << endl;
  if (file.is_open()) {
    for (int i = 0; i < 36; i++) {
      Eigen::Matrix3d Rwr;
      Eigen::Vector3d twr;
      Eigen::Vector3d vel;
      file >> Rwr(0, 0) >> Rwr(0, 1) >> Rwr(0, 2)
          >> Rwr(1, 0) >> Rwr(1, 1) >> Rwr(1, 2)
          >> Rwr(2, 0) >> Rwr(2, 1) >> Rwr(2, 2)
          >> twr(0) >> twr(1) >> twr(2)
          >> vel(0) >> vel(1) >> vel(2);
      states.push_back(State(Rwr, twr, vel));
    }
  }
  return states;
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
