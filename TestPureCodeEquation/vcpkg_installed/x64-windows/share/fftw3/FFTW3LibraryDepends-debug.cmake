#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "FFTW3::fftw3" for configuration "Debug"
set_property(TARGET FFTW3::fftw3 APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(FFTW3::fftw3 PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/fftw3.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/fftw3.dll"
  )

list(APPEND _cmake_import_check_targets FFTW3::fftw3 )
list(APPEND _cmake_import_check_files_for_FFTW3::fftw3 "${_IMPORT_PREFIX}/debug/lib/fftw3.lib" "${_IMPORT_PREFIX}/debug/bin/fftw3.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
