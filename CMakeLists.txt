project(biped-evolve)

cmake_minimum_required(VERSION 2.8.4)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Find GLUT
find_package(GLUT REQUIRED)

# Glob all source files
set(SRC_DIR ${biped-evolve_SOURCE_DIR}/src)
file(GLOB SOURCES ${SRC_DIR}/*.cpp)

# Create executable
add_executable(biped-evolve ${SOURCES})
target_link_libraries(biped-evolve ode drawstuff ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})