cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(lumacam)

# Find Geant4 (compatible with 10.6)
find_package(Geant4 10.6 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# Set C++ standard to 14 explicitly
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_definitions(-std=c++14)  # Fallback if CMake ignores the standard

# Define your sources and headers
set(SOURCES
    main.cc
    MaterialBuilder.cc
    GeometryConstructor.cc
    ParticleGenerator.cc
    EventProcessor.cc
    SimulationManager.cc
    LumaCamMessenger.cc
    SimConfig.cc
)

set(HEADERS
    SimConfig.hh
    MaterialBuilder.hh
    GeometryConstructor.hh
    ParticleGenerator.hh
    EventProcessor.hh
    SimulationManager.hh
    LumaCamMessenger.hh
)

# Use SSLG4 and OPSim paths from setup.py
if(NOT SSLG4_INCLUDE_DIR OR NOT SSLG4_SOURCE_DIR OR NOT OPSIM_INCLUDE_DIR OR NOT OPSIM_SOURCE_DIR)
    message(FATAL_ERROR "SSLG4_INCLUDE_DIR, SSLG4_SOURCE_DIR, OPSIM_INCLUDE_DIR, and OPSIM_SOURCE_DIR must be set via CMake command line.")
endif()

include_directories(${SSLG4_INCLUDE_DIR} ${OPSIM_INCLUDE_DIR})
file(GLOB SSLG4_SOURCES "${SSLG4_SOURCE_DIR}/*.cc")
file(GLOB OPSIM_SOURCES "${OPSIM_SOURCE_DIR}/*.cc")

# Validate key headers
if(NOT EXISTS "${SSLG4_INCLUDE_DIR}/OrganicScintillatorFactory.hh")
    message(FATAL_ERROR "OrganicScintillatorFactory.hh not found in ${SSLG4_INCLUDE_DIR}. Verify SSLG4 submodule.")
endif()
if(NOT EXISTS "${OPSIM_INCLUDE_DIR}/VMaterialFactory.hh")
    message(FATAL_ERROR "VMaterialFactory.hh not found in ${OPSIM_INCLUDE_DIR}. Verify OPSim submodule.")
endif()

# Combine sources
add_executable(lumacam ${SOURCES} ${SSLG4_SOURCES} ${OPSIM_SOURCES} ${HEADERS})
target_link_libraries(lumacam ${Geant4_LIBRARIES})

# Set output directory to build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Handle SSGL4 macros
file(GLOB_RECURSE SSLG4_MACROS "${SSLG4_SOURCE_DIR}/../macros/*/*.mac")
foreach(_macro ${SSLG4_MACROS})
    get_filename_component(_macro_name ${_macro} NAME)
    get_filename_component(_macro_dir ${_macro} DIRECTORY)
    file(RELATIVE_PATH _relative_dir ${SSLG4_SOURCE_DIR}/../macros ${_macro_dir})
    configure_file(
        ${_macro}
        ${CMAKE_BINARY_DIR}/sslg4/macros/${_relative_dir}/${_macro_name}
        COPYONLY
    )
endforeach()

# Handle SSGL4 data files
file(GLOB_RECURSE SSLG4_DATA "${SSLG4_SOURCE_DIR}/../data/*/*/*")
foreach(_data ${SSLG4_DATA})
    get_filename_component(_data_name ${_data} NAME)
    get_filename_component(_data_dir ${_data} DIRECTORY)
    file(RELATIVE_PATH _relative_dir ${SSLG4_SOURCE_DIR}/../data ${_data_dir})
    configure_file(
        ${_data}
        ${CMAKE_BINARY_DIR}/sslg4/data/${_relative_dir}/${_data_name}
        COPYONLY
    )
endforeach()

# Install the executable, macros, and data
install(TARGETS lumacam DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/sslg4/macros/ DESTINATION sslg4/macros)
install(DIRECTORY ${CMAKE_BINARY_DIR}/sslg4/data/ DESTINATION sslg4/data)