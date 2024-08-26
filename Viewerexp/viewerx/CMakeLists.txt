
# optional libraries
find_package(Pangolin REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

unset(CMAKE_CUDA_ARCHITECTURES CACHE)
unset(CUDA_NVCC_FLAGS CACHE)

# Set desired CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")

# Define CUDA NVCC flags correctly
set(CUDA_NVCC_FLAGS "")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--generate-code=arch=compute_89,code=[compute_89,sm_89]")

string(REPLACE ";" " " CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS}")

#pybind11_add_module(dpviewerx viewer.cpp viewer_cuda.cu)
pybind11_add_module(viewerxx viewer.cpp viewer_cuda.cu)
include_directories(${Pangolin_INCLUDE_DIRS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(dpviewerx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
# target_link_libraries(dpviewerx PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})
target_compile_definitions(viewerxx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(viewerxx PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})

