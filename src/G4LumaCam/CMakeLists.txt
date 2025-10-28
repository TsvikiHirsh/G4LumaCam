cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(lumacam)

find_package(Geant4 10.0 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

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

add_executable(lumacam ${SOURCES} ${HEADERS})
target_link_libraries(lumacam ${Geant4_LIBRARIES})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(lumacam)

