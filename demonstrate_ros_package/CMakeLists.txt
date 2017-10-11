cmake_minimum_required(VERSION 2.8.3)
project(demonstrate_leap)

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -O3 -DNDEBUG")
find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs trajectory_msgs std_msgs geometry_msgs rospy)

catkin_package(
   INCLUDE_DIRS include
   CATKIN_DEPENDS roscpp
)


include_directories(include
  ${catkin_INCLUDE_DIRS}
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(arch_64 TRUE)
else()
    set(arch_64 FALSE)
endif()

if(arch_64)
    link_directories(${LINK_DIRECTORIES} ${PROJECT_SOURCE_DIR}/lib/x64)
else()
    link_directories(${LINK_DIRECTORIES} ${PROJECT_SOURCE_DIR}/lib/x86)
endif()


add_executable(leap_al5d src/leap_al5d.cpp)
target_link_libraries(leap_al5d Leap ${catkin_LIBRARIES})

install(TARGETS leap_al5d
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  lib/x64/libLeap.so
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
