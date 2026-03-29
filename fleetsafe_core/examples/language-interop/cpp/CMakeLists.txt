cmake_minimum_required(VERSION 3.14)
project(robot_control)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Fetch fleetsafe_core-lcm for message headers
FetchContent_Declare(fleetsafe_core_lcm
  GIT_REPOSITORY https://github.com/Fleet-Safe-VLA/fleetsafe_core-lcm.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fleetsafe_core_lcm)

# Find LCM via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(LCM REQUIRED lcm)

add_executable(robot_control main.cpp)

target_include_directories(robot_control PRIVATE
  ${fleetsafe_core_lcm_SOURCE_DIR}/generated/cpp_lcm_msgs
  ${LCM_INCLUDE_DIRS}
)

target_link_libraries(robot_control PRIVATE ${LCM_LIBRARIES})
target_link_directories(robot_control PRIVATE ${LCM_LIBRARY_DIRS})
