cmake_minimum_required(VERSION 2.8.3)
project(motion-tracker)

find_package(catkin REQUIRED COMPONENTS
  OpenCV
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})



