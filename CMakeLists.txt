cmake_minimum_required(VERSION 3.16)
project(gym-os2r VERSION 1.0.0)

# Add custom functions / macros
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT PYTHON_INSTALL_PREFIX)
    set(PYTHON_INSTALL_PREFIX ${Python3_SITELIB})
    set(PYTHONPATHS "$ENV{PYTHONPATH}:${PROJECT_SOURCE_DIR}")
else()
    set(PYTHONPATHS "$ENV{PYTHONPATH}:${PROJECT_SOURCE_DIR}:${PYTHON_INSTALL_PREFIX}")
endif()

set(PYTHON_INSTALL_PREFIX "${PYTHON_INSTALL_PREFIX}" CACHE STRING "Set the python install prefix to empty string. Only install if it is set as a cmake arg.")
mark_as_advanced(PYTHON_INSTALL_PREFIX)

install(DIRECTORY "./gym_os2r" DESTINATION "${PYTHON_INSTALL_PREFIX}")

option(BUILD_DOCS "Enable the configuration and building of the documentation." FALSE)
option(BUILD_DOCS_SUPERBUILD "Build the docs with the purpose of using in the superbuild doc compilation. \
This will make it so the docs are not built only configured." FALSE)

IF(BUILD_DOCS)
    add_subdirectory(docs)
ENDIF()
