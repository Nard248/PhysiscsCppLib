# Install script for directory: C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/pkgs/eigen3_x64-windows")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "OFF")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/AdolcForward"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/AlignedVector3"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/ArpackSupport"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/AutoDiff"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/BVH"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/EulerAngles"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/FFT"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/IterativeSolvers"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/KroneckerProduct"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/LevenbergMarquardt"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/MatrixFunctions"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/MoreVectorization"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/MPRealSupport"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/NonLinearOptimization"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/NumericalDiff"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/OpenGLSupport"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/Polynomials"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/Skyline"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/SparseExtra"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/SpecialFunctions"
    "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/src/3.4.0-1beca8819c.clean/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/blds/eigen3/x64-windows-rel/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

