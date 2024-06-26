#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "FFTW3::fftw3l" for configuration "Release"
set_property(TARGET FFTW3::fftw3l APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(FFTW3::fftw3l PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/fftw3l.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/fftw3l.dll"
  )

list(APPEND _cmake_import_check_targets FFTW3::fftw3l )
list(APPEND _cmake_import_check_files_for_FFTW3::fftw3l "${_IMPORT_PREFIX}/lib/fftw3l.lib" "${_IMPORT_PREFIX}/bin/fftw3l.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
