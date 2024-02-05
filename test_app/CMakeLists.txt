cmake_minimum_required(VERSION 3.18)
project(dlopen_test C CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# tvm runtime config: minimize runtime components
set(USE_RPC OFF)
#set(USE_METAL ON)
set(USE_MICRO OFF)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_GRAPH_EXECUTOR_DEBUG OFF)
set(USE_AOT_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_GTEST OFF)
set(USE_LIBBACKTRACE OFF)
#set(BUILD_DUMMY_LIBTVM ON)
if (NOT DEFINED TVM_HOME)
  set(TVM_HOME ../3rdparty/tvm)
endif (NOT DEFINED TVM_HOME)
message(STATUS "TVM_HOME: ${TVM_HOME}")
add_subdirectory(${TVM_HOME} tvm EXCLUDE_FROM_ALL)


set(
  MLC_LLM_INCLUDES
  ${TVM_HOME}/include
  ${TVM_HOME}/3rdparty/dlpack/include
  ${TVM_HOME}/3rdparty/dmlc-core/include
  ${TVM_HOME}/3rdparty/picojson
)

link_directories(../dist/Llama-2-7b-chat-hf-q4f16_1-MLC)

add_executable(${PROJECT_NAME} main.cc)
target_include_directories(${PROJECT_NAME} PRIVATE ${MLC_LLM_INCLUDES})
target_link_libraries(${PROJECT_NAME} PRIVATE tvm_runtime Llama-2-7b-chat-hf-q4f16_1-android)
