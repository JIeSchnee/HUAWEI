cmake_minimum_required(VERSION 3.8)
project(coremark_pro)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(coremark_pro REQUIRED)

# Add the include directory of the header file
include_directories(
  include
  /home/jiezou/Documents/coremark-pro/mith/include/
  /home/jiezou/Documents/coremark-pro/mith/al/include/
  # /home/jiezou/ros2_ws/src/coremark_pro/include/external/
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


add_executable(cjpeg src/workload_0.cpp)
add_executable(nnet_test src/workload_1.cpp)
add_executable(linear src/workload_2.cpp)
add_executable(loops src/workload_3.cpp)
add_executable(radix src/workload_4.cpp)
add_executable(zip src/workload_5.cpp)
add_executable(core src/workload_6.cpp)
add_executable(sha src/workload_7.cpp)
add_executable(parser_125 src/workload_8.cpp)
add_executable(monitor src/monitor.cpp)


add_executable(nnet_E src/workload_1E.cpp)
add_executable(linear_E src/workload_2E.cpp)
add_executable(loops_E src/workload_3E.cpp)
add_executable(radix_E src/workload_4E.cpp)
add_executable(zip_E src/workload_5E.cpp)
add_executable(core_E src/workload_6E.cpp)
add_executable(sha_E src/workload_7E.cpp)
add_executable(parser_E src/workload_8E.cpp)

target_link_libraries(cjpeg 
/home/jiezou/Documents/coremark-pro/cjpeg.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(nnet_test 
/home/jiezou/Documents/coremark-pro/nnet_test.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(linear 
/home/jiezou/Documents/coremark-pro/linear.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(loops 
/home/jiezou/Documents/coremark-pro/loops.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(radix 
/home/jiezou/Documents/coremark-pro/radix.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(zip 
/home/jiezou/Documents/coremark-pro/zip.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(core 
/home/jiezou/Documents/coremark-pro/core.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(sha 
/home/jiezou/Documents/coremark-pro/sha.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(parser_125 
/home/jiezou/Documents/coremark-pro/parser-125k.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)


target_link_libraries(nnet_E 
/home/jiezou/Documents/coremark-pro/nnet_test.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(linear_E 
/home/jiezou/Documents/coremark-pro/linear.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(loops_E 
/home/jiezou/Documents/coremark-pro/loops.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(radix_E 
/home/jiezou/Documents/coremark-pro/radix.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(zip_E 
/home/jiezou/Documents/coremark-pro/zip.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(core_E 
/home/jiezou/Documents/coremark-pro/core.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(sha_E 
/home/jiezou/Documents/coremark-pro/sha.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)
target_link_libraries(parser_E 
/home/jiezou/Documents/coremark-pro/parser-125k.a
/home/jiezou/Documents/coremark-pro/builds/linux64/gcc64/obj/mith.a
)


ament_target_dependencies(cjpeg rclcpp)
ament_target_dependencies(nnet_test rclcpp)
ament_target_dependencies(linear rclcpp)
ament_target_dependencies(loops rclcpp)
ament_target_dependencies(radix rclcpp)
ament_target_dependencies(zip rclcpp)
ament_target_dependencies(core rclcpp)
ament_target_dependencies(sha rclcpp)
ament_target_dependencies(parser_125 rclcpp)
ament_target_dependencies(monitor rclcpp)

ament_target_dependencies(nnet_E rclcpp)
ament_target_dependencies(linear_E rclcpp)
ament_target_dependencies(loops_E rclcpp)
ament_target_dependencies(radix_E rclcpp)
ament_target_dependencies(zip_E rclcpp)
ament_target_dependencies(core_E rclcpp)
ament_target_dependencies(sha_E rclcpp)
ament_target_dependencies(parser_E rclcpp)



install(TARGETS
  cjpeg
  nnet_test
  linear
  loops
  radix
  zip
  core
  sha
  parser_125
  monitor
  nnet_E
  linear_E
  loops_E
  radix_E
  zip_E
  core_E
  sha_E
  parser_E
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
