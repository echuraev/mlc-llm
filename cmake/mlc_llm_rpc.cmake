set(TVM_CPP_RPC_DIR ${TVM_HOME}/apps/cpp_rpc)
set(DLPACK_PATH ${TVM_HOME}/3rdparty/dlpack/include)
set(DMLC_PATH ${TVM_HOME}/3rdparty/dmlc-core/include)
set(TVM_RPC_SOURCES
  ${TVM_CPP_RPC_DIR}/main.cc
  ${TVM_CPP_RPC_DIR}/rpc_env.cc
  ${TVM_CPP_RPC_DIR}/rpc_server.cc
)
tvm_file_glob(GLOB RUNTIME_RPC_SRCS ${TVM_HOME}/src/runtime/rpc/*.cc)
list(APPEND TVM_RPC_SOURCES ${RUNTIME_RPC_SRCS})
link_directories(${CMAKE_BINARY_DIR})

set(TVM_RPC_LINKER_LIBS "")

if(WIN32)
  list(APPEND TVM_RPC_SOURCES ${TVM_CPP_RPC_DIR}/win32_process.cc)
endif()

# Set output to same directory as the other TVM libs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(mlc_llm_rpc ${TVM_RPC_SOURCES})

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET mlc_llm_rpc PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(WIN32)
  target_compile_definitions(mlc_llm_rpc PUBLIC -DNOMINMAX)
endif()

if (OS)
   if (OS STREQUAL "Linux")
      set_property(TARGET mlc_llm_rpc PROPERTY LINK_FLAGS -lpthread)
   endif()
endif()

if(USE_OPENCL)
   if (ANDROID_ABI)
     if(DEFINED ENV{ANDROID_NDK_MAJOR})
       if($ENV{ANDROID_NDK_MAJOR} VERSION_LESS "23")
         set_property(TARGET mlc_llm_rpc PROPERTY LINK_FLAGS -fuse-ld=gold)
       endif()
     endif()
   endif()
endif()

target_include_directories(
  mlc_llm_rpc
  PUBLIC ${TVM_HOME}/include
  PUBLIC ${DLPACK_PATH}
  PUBLIC ${DMLC_PATH}
)

if (BUILD_FOR_ANDROID AND USE_HEXAGON)
  get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
    DSPRPC_LIB DSPRPC_LIB_DIRS
  )
  if(DSPRPC_LIB_DIRS)
    link_directories(${DSPRPC_LIB_DIRS})
  else()
    message(WARNING "Could not locate some Hexagon SDK components")
  endif()
  list(APPEND TVM_RPC_LINKER_LIBS cdsprpc log)
endif()

if(USE_ETHOSN)
  if (ETHOSN_RUNTIME_LIBRARY)
    list(APPEND TVM_RPC_LINKER_LIBS ${ETHOSN_RUNTIME_LIBRARY})
  else()
    message(WARNING "Could not locate Arm(R) Ethos(TM)-N runtime library components")
  endif()
endif()

if(BUILD_STATIC_RUNTIME)
  list(APPEND TVM_RPC_LINKER_LIBS -Wl,--whole-archive tvm_runtime -Wl,--no-whole-archive)
else()
  list(APPEND TVM_RPC_LINKER_LIBS tvm_runtime)
endif()
list(APPEND TVM_RPC_LINKER_LIBS mlc_llm)

#list(APPEND TVM_RPC_LINKER_LIBS Llama-2-7b-chat-hf-q4f16_1-metal)
list(APPEND TVM_RPC_LINKER_LIBS Llama-2-7b-chat-hf-q4f16_1-opencl)

target_link_libraries(mlc_llm_rpc ${TVM_RPC_LINKER_LIBS})

