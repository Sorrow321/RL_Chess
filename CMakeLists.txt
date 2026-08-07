cmake_minimum_required(VERSION 3.15)
project(chess_alphazero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Embedded movegen: a single vendored header, pinned by commit.
# See third_party/chess-library/VERSION.
add_library(chess_lib INTERFACE)
target_include_directories(chess_lib INTERFACE ${CMAKE_SOURCE_DIR}/third_party/chess-library)

add_library(engine INTERFACE)
target_include_directories(engine INTERFACE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(engine INTERFACE chess_lib)
target_compile_options(engine INTERFACE -O3 -march=native)

find_package(Threads REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(chess_engine cpp/chess_engine.cpp)
target_link_libraries(chess_engine PRIVATE engine Threads::Threads)

# Place chess_engine*.so at the repo root so `import chess_engine` works from here
set_target_properties(chess_engine PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- tests -------------------------------------------------------------------
enable_testing()

add_executable(perft_test tests/perft_test.cpp)
target_link_libraries(perft_test PRIVATE engine Threads::Threads)

add_executable(rules_test tests/rules_test.cpp)
target_link_libraries(rules_test PRIVATE engine)

# The full suite is ~440M nodes and finishes in under a second, so there is no
# reason to run anything less. tests/run_perft.sh wraps this with a build.
add_test(NAME perft COMMAND perft_test)
add_test(NAME rules COMMAND rules_test)
