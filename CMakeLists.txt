cmake_minimum_required(VERSION 3.16)
project(gym-os2r VERSION 1.0.0)

# Add custom functions / macros
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYTHON_INSTALL_PREFIX "" CACHE STRING "Set the python install prefix to empty string. Only install if it is set as a cmake arg.")
mark_as_advanced(PYTHON_INSTALL_PREFIX)

if(PYTHON_INSTALL_PREFIX NOT STREQUAL "")
  install(DIRECTORY "./gym_os2r" DESTINATION "${PYTHON_INSTALL_PREFIX}/gym_os2r")
endif()
add_subdirectory(docs)
