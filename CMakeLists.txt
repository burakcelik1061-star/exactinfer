cmake_minimum_required(VERSION 3.24)
project(engine_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT MSVC)
  message(FATAL_ERROR "Su an yalnizca MSVC (Visual Studio) destekleniyor; build.ps1 ile derleyin.")
endif()

option(EL_CUDA "CUDA arka ucunu derle (bulunursa)" ON)
if(EL_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 20xx (Turing); PTX da gömülür, daha yeni kartlarda JIT ile çalışır
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(WARNING "CUDA derleyicisi bulunamadi: yalnizca CPU arka ucu derlenecek")
    set(EL_CUDA OFF)
  endif()
endif()

add_library(elcore STATIC
  src/gguf.cpp
  src/tokenizer.cpp
  src/ops.cpp
  src/json.cpp
  src/constrain.cpp
  src/model.cpp
  src/quant.cpp
)
if(EL_CUDA)
  target_sources(elcore PRIVATE src/cuda/model_cuda.cu)
  target_link_libraries(elcore PUBLIC CUDA::cudart_static)
  target_compile_definitions(elcore PUBLIC EL_HAVE_CUDA=1)
else()
  target_sources(elcore PRIVATE src/cuda_stub.cpp)
endif()

target_include_directories(elcore PUBLIC src)
target_compile_options(elcore PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /W4 /permissive- /EHsc /arch:AVX2 /openmp /Zc:__cplusplus>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/EHsc,/W3 -lineinfo --expt-relaxed-constexpr -Xptxas=-warn-spills>
)
target_compile_definitions(elcore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)

add_executable(el src/main.cpp)
target_link_libraries(el PRIVATE elcore)
