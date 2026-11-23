cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_HIP_COMPILER)
  set(CMAKE_HIP_COMPILER /opt/rocm/bin/amdclang++)
endif()
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1152)
endif()

project(nn LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hipblas REQUIRED CONFIG)
find_package(rocblas REQUIRED CONFIG)

set(NN_HIP_SOURCES
  src/elementwise.hip
  src/activate.hip
  src/embedding.hip
  src/layer_norm.hip
  src/rms_norm.hip
  src/multi_head_attention.hip
  src/sgd.hip
  src/adamw.hip
  src/loss.hip
  src/relu.hip
  src/conv2d.hip
  src/max_pool2d.hip
  src/batch_norm2d.hip
  src/global_avg_pool2d.hip
)

add_library(nn STATIC
  src/context.cpp
  src/error.cpp
  src/tensor.cpp
  src/index_tensor.cpp
  src/init.cpp
  src/config.cpp
  src/linear.cpp
  src/gelu.cpp
  src/swiglu.cpp
  src/dropout.cpp
  src/transformer_block.cpp
  src/transformer.cpp
  src/rnn.cpp
  src/last_step.cpp
  src/flatten.cpp
  src/network.cpp
  src/optim.cpp
  src/checkpoint.cpp
  src/builders.cpp
  src/residual_block.cpp
  ${NN_HIP_SOURCES}
)

set_source_files_properties(${NN_HIP_SOURCES} PROPERTIES LANGUAGE HIP)

target_include_directories(nn
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(nn PUBLIC roc::hipblas roc::rocblas)

target_compile_options(nn PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-deprecated-declarations>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>)

set_target_properties(nn PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(nn_tests
  tests/test_main.cpp
  tests/test_shapes.cpp
  tests/test_gradients.cpp
  tests/test_checkpoint.cpp
  tests/test_mnist.cpp
  tests/test_bf16.cpp
  tests/mnist.cpp
)

set_target_properties(nn_tests PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(nn_tests PRIVATE nn)
target_include_directories(nn_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(nn_tests PRIVATE -Wno-deprecated-declarations)

enable_testing()
add_test(NAME shapes COMMAND nn_tests shapes)
add_test(NAME grads COMMAND nn_tests grads)
add_test(NAME ckpt COMMAND nn_tests ckpt)
add_test(NAME mnist COMMAND nn_tests mnist)
add_test(NAME bf16 COMMAND nn_tests bf16)
set_tests_properties(shapes grads ckpt PROPERTIES TIMEOUT 900)
set_tests_properties(mnist PROPERTIES TIMEOUT 7200)
