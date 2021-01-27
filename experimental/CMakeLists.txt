find_package(Ceres REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(sim
               sim.cpp)

target_link_libraries(sim
                      gtsam
                      ${CERES_LIBRARIES})

add_executable(solver
               solver.cpp)

target_include_directories(solver
                            PRIVATE
                            ${CERES_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR})



target_link_libraries(solver
                      ${CERES_LIBRARIES})

