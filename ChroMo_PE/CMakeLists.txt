cmake_minimum_required(VERSION 3.15)
project(EDM_SOLVER LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EDM_SOLVER_BIN "edm_nonlinear_solver")

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/*.cpp)

if(MSVC)
    add_compile_options(
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:Release>:/arch:AVX2>
    )
else()
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
    )
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# OpenMP is optional on MSVC: pandas loads Intel's OpenMP at runtime and a
# second OpenMP runtime (vcomp140.dll) causes a crash in the same process.
# On GCC/Clang the #pragma omp directives provide real parallelism.
if(NOT MSVC)
    find_package(OpenMP)
endif()

add_library(${EDM_SOLVER_BIN} MODULE ${SOURCE_FILES})

target_link_libraries(${EDM_SOLVER_BIN} PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

pybind11_extension(${EDM_SOLVER_BIN})
if (NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    pybind11_strip(${EDM_SOLVER_BIN})
endif ()

set_target_properties(${EDM_SOLVER_BIN} PROPERTIES CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${EDM_SOLVER_BIN} PRIVATE OpenMP::OpenMP_CXX)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if (ipo_supported)
    set_property(TARGET ${EDM_SOLVER_BIN} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()
