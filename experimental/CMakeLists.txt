find_package(Ceres REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(sim
               sim.cpp)

target_link_libraries(sim
                      gtsam
                      ${CERES_LIBRARIES})

add_executable(solverFixedLag
               solverFixedLag.cpp)
target_include_directories(solverFixedLag
                            PRIVATE
                            ${CERES_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR})
target_link_libraries(solverFixedLag
                      ${CERES_LIBRARIES})

add_executable(solver
               solver.cpp)
target_include_directories(solver
                            PRIVATE
                            ${CERES_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR})
target_link_libraries(solver
                      ${CERES_LIBRARIES})

add_executable(simple_solver
               simple_solver.cpp)
target_include_directories(simple_solver
                            PRIVATE
                            ${CERES_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR})
target_link_libraries(simple_solver
                      ${CERES_LIBRARIES})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
target_link_libraries(solverFixedLag ${Pangolin_LIBRARIES})



add_subdirectory(tests)