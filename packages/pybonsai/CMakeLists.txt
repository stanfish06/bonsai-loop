cmake_minimum_required(VERSION 3.18)
# No language yet: without the upstream checkouts nothing is compiled, and the
# package must still install on a machine that has no C++ compiler.
project(pybonsai LANGUAGES NONE)

# ---- options -------------------------------------------------------------
# Upstream checkouts; default to the pinned submodules.
set(PYBONSAI_PENJING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/penjing" CACHE PATH
    "penjing checkout to build against")
set(PYBONSAI_BONSAI_PORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bonsai_port" CACHE PATH
    "bonsai_port checkout that penjing builds against")
# Link libstdc++ and libgcc into the module so it loads next to wheels that ship an older libstdc++.
option(PYBONSAI_STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically" OFF)
# --------------------------------------------------------------------------

# The upstream repositories are private. Without them only pybonsai/__init__.py is
# installed, and importing pybonsai raises an ImportError that says why.
if (NOT EXISTS "${PYBONSAI_PENJING_DIR}/CMakeLists.txt" OR
    NOT EXISTS "${PYBONSAI_BONSAI_PORT_DIR}/CMakeLists.txt")
    message(WARNING
        "pybonsai: ${PYBONSAI_PENJING_DIR} or ${PYBONSAI_BONSAI_PORT_DIR} is empty, "
        "so the C++ extension is not built. With access to both repositories, run "
        "`git submodule update --init packages/pybonsai/third_party/penjing "
        "packages/pybonsai/third_party/bonsai_port` and reinstall.")
    return()
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The upstream static libraries end up inside a shared module, so they need -fPIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# bonsai_port fetches NLopt, which builds shared by default; a shared libnlopt would stay
# behind in the build tree and the installed module could not load it.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# penjing adds bonsai_port itself from BONSAI_PORT_SOURCE_DIR. EXCLUDE_FROM_ALL builds
# only the libraries the module links, not upstream's executables or tests.
set(BONSAI_PORT_SOURCE_DIR "${PYBONSAI_BONSAI_PORT_DIR}" CACHE PATH "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(PENJING_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${PYBONSAI_PENJING_DIR}" penjing EXCLUDE_FROM_ALL)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)
find_package(OpenMP REQUIRED)

cython_transpile(src/pybonsai/_core.pyx LANGUAGE CXX OUTPUT_VARIABLE core_cxx)

python_add_library(_core MODULE "${core_cxx}" src/pybonsai/_shim.cpp WITH_SOABI)
target_include_directories(_core PRIVATE src/pybonsai)
target_link_libraries(_core PRIVATE
    penjing_pairwise_distances
    penjing_convergence_detection
    OpenMP::OpenMP_CXX)
if (PYBONSAI_STATIC_LIBSTDCXX)
    target_link_options(_core PRIVATE -static-libstdc++ -static-libgcc)
endif()

install(TARGETS _core DESTINATION pybonsai)
