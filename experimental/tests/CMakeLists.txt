add_executable(imu_integration
               imu_integration.cpp)

target_include_directories(imu_integration
                            PRIVATE
                            ${EIGEN3_INCLUDE_DIR})

