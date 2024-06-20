# defined since 2.8.3
if (CMAKE_VERSION VERSION_LESS 2.8.3)
  get_filename_component (CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
endif ()

# Allows loading FFTW3 settings from another project
set (FFTW3_CONFIG_FILE "${CMAKE_CURRENT_LIST_FILE}")

set (FFTW3l_LIBRARIES fftw3l)
set (FFTW3l_LIBRARY_DIRS C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/pkgs/fftw3_x64-windows/debug/lib)
set (FFTW3l_INCLUDE_DIRS C:/Users/meloy/source/repos/TestPureCodeEquation/TestPureCodeEquation/vcpkg_installed/vcpkg/pkgs/fftw3_x64-windows/debug/include)

include ("${CMAKE_CURRENT_LIST_DIR}/FFTW3LibraryDepends.cmake")

if (CMAKE_VERSION VERSION_LESS 2.8.3)
  set (CMAKE_CURRENT_LIST_DIR)
endif ()
