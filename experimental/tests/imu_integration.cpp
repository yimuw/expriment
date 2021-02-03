#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;

Matrix3d euler_to_mat( const double roll,
                        const double pitch,
                        const double yaw )
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q.matrix();
}

Matrix3d qua_integration(const Matrix3d &R0, const std::vector<Vector3d> &w_vec) 
{
    std::cout << "============= qua_integration =============" << std::endl;

    Eigen::Quaterniond q(R0);

    for(int i = 0; i < w_vec.size(); ++i) {
        Vector3d omega_measured_ = w_vec[i];
        constexpr double DT = 1.0;

        Eigen::Quaterniond q_new;
        Eigen::Quaterniond q_omega;
        Eigen::Quaterniond q_add; 
        q_omega.w() = 0;
        q_omega.vec() = omega_measured_ * DT * 0.5;
        q_add = q_omega * q;
        q_new.w() = q.w() + q_add.w();
        q_new.vec() = q.vec() + q_add.vec();

        q = q_new;

        std::cout << "i:" << i << " w:" << omega_measured_.transpose() << "\nR:" << q.matrix() << std::endl;
    }
    return q.matrix();
}


inline Eigen::Matrix3d skewm(const Eigen::Vector3d& v)
{
    double x = v(0);
    double y = v(1);
    double z = v(2);

    Eigen::Matrix3d S;
    S << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;

    return S;
}

inline Eigen::Matrix3d exp_hat_so3(const Eigen::Vector3d& v)
{
    const double theta = v.norm();
    if (theta < 1e-10)
    {
        return Matrix3d::Identity();
    }
    const Vector3d w = v / theta;
    Matrix3d W = skewm(w);
    // NOTE(Ning): W*W = -I + w * w^T
    // NOTE(Ning): Rodrigues rotation formula
    Matrix3d SO3 = Matrix3d::Identity() + std::sin(theta) * W + (1 - std::cos(theta)) * W * W;
    return SO3;
}

Matrix3d mat_integration(const Matrix3d &R0, const std::vector<Vector3d> &w_vec) {
    std::cout << "============= mat_integration =============" << std::endl;
    Eigen::Matrix3d R = R0;
    for(int i = 0; i < w_vec.size(); ++i) {
        Vector3d w = w_vec[i];
        R = R * exp_hat_so3(w);
        std::cout << "i:" << i << " w:" << w.transpose() << "\nR:" << R << std::endl;
    }

    return R;
}

void test1()
{
    std::cout << "test1" << std::endl;
    std::vector<Eigen::Vector3d> w_vec = 
        {
            {0,0,0},
            {1,0,0},
            {0,1,0},
            {0,0,1},
            {1,1,1},
            {-1,-2,-3}
        };

    Matrix3d R0 = euler_to_mat(0.1,0.2,-0.1);

    mat_integration(R0, w_vec);

    qua_integration(R0, w_vec);
}

void test2()
{
    std::cout << "test2" << std::endl;

    std::vector<Eigen::Vector3d> w_vec = 
    {
        {0,0,0},
        {0,0,0}
    };

    Matrix3d R0 = euler_to_mat(0.0,0.0,0.0);

    mat_integration(R0, w_vec);

    qua_integration(R0, w_vec);
}


void test3()
{
    std::cout << "test3" << std::endl;

    std::vector<Eigen::Vector3d> w_vec = 
    {
        {0.01,0.01,0.01},
        {0.01,0.01,0.01},
        {0.01,0.01,0.01},
        {0.01,0.01,0.01}
    };

    Matrix3d R0 = euler_to_mat(0.0,0.0,0.0);

    mat_integration(R0, w_vec);

    qua_integration(R0, w_vec);
}

int main() {
    test1();
    test2();
    test3();
}