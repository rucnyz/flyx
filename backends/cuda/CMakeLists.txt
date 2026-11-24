# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# CUDA backend: one loadable module (CUDA settings: the top-level CMakeLists.txt).
#   kernels/       the kernels, one translation unit per op family (static library flyx_cuda_kernels)
#   regions/   host: what each node runs on its block, plans
#   runtime/   host: sequence state, prefill passes, rounds
#   loader/    host: the description bound to blocks, weights onto the device
#   variants/  tuning tables per GPU, turned into dispatch rules by tools/codegen

# Tuning tables -> generated C++ (configurations to compile, dispatch rules per GPU).
file(GLOB FLYX_TUNING_TABLES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/variants/*/tables/*.toml)
set(FLYX_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
set(FLYX_GEMM_TABLES ${FLYX_GEN_DIR}/flyx/gemm_tables.inc)
add_custom_command(
  OUTPUT ${FLYX_GEMM_TABLES}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/codegen/gen_gemm_tables.py
          --candidates ${CMAKE_CURRENT_SOURCE_DIR}/variants/candidates.toml --out ${FLYX_GEMM_TABLES}
          ${FLYX_TUNING_TABLES}
  DEPENDS ${CMAKE_SOURCE_DIR}/tools/codegen/gen_gemm_tables.py ${CMAKE_CURRENT_SOURCE_DIR}/variants/candidates.toml
          ${FLYX_TUNING_TABLES}
  COMMENT "Generating GEMM tuning tables")
add_custom_target(flyx_gemm_tables DEPENDS ${FLYX_GEMM_TABLES})

# Kernels: one translation unit per op family (kernels/), plus what they share (kernels/common/).  The GEMMs on GGML
# block weights compile as one unit per group of block types (gemm_blocks_<group>.cu, side by side: as one they
# took 100 s of a build's 99) and one for their staging and API (gemm_blocks.cu); so do their rounds in 8 bits
# (gemm_blocks_round_<group>.cu: every configuration the variants compile, for every type, alone and before a run in
# Q8_0; gemm_blocks_round.cu), and the dp4a kernel's SwiGLU variant for every type (gemm_blocks_round_glu.cu).
# Non-FP8 heavy kernel units (GGML block GEMMs, strata attention, GDN): they use no sm_86-only instructions, so when
# sm_80 is built its cubin runs on sm_86 via CUDA minor-version binary compatibility.  sm_89 keeps its own: the
# prefill GEMM in 8 bits converts its sums exactly from zero there (kMagic's note), 5% faster prefill on an L4 and an
# L40S than sm_80's code, for 8.6 MB more of the release's libflyx-cuda.so (51.7 -> 60.3 MB).
set(FLYX_CUDA_COMPAT_BLOCK_SOURCES
    kernels/gemm_blocks_legacy.cu kernels/gemm_blocks_k.cu
    kernels/gemm_blocks_iq_low.cu kernels/gemm_blocks_iq_high.cu
    kernels/gemm_blocks_round_legacy.cu kernels/gemm_blocks_round_k.cu
    kernels/gemm_blocks_round_iq_low.cu kernels/gemm_blocks_round_iq_high.cu kernels/gemm_blocks_round_glu.cu)
set(FLYX_CUDA_COMPAT_SOURCES ${FLYX_CUDA_COMPAT_BLOCK_SOURCES} kernels/attention_strata.cu kernels/gdn.cu)
set(FLYX_CUDA_TABLE_SOURCES
    kernels/gemm.cu kernels/gemm_decode.cu kernels/attention_flash.cu
    kernels/gemm_blocks.cu kernels/gemm_blocks_round.cu ${FLYX_CUDA_COMPAT_BLOCK_SOURCES})
set(FLYX_CUDA_KERNEL_SOURCES
    kernels/common/launch.cu
    kernels/gemm.cu kernels/gemm_decode.cu kernels/gemm_prefill.cu
    kernels/gemm_blocks.cu kernels/gemm_blocks_round.cu kernels/gemm_blocks_tables.cu
    kernels/attention.cu kernels/attention_flash.cu kernels/attention_rows.cu kernels/tree.cu
    kernels/gdn_chunk.cu kernels/rows.cu kernels/sampling.cu kernels/draft.cu kernels/draft_head.cu
    kernels/dv.cu kernels/head_cert.cu kernels/stop_decide.cu kernels/vision.cu kernels/dspine.cu)

# The device code's flags: line tables unless left out, the fatbinary's compression (top-level CMakeLists.txt).
set(FLYX_CUDA_DEVICE_FLAGS "")
if(MSVC)
  # nvcc over MSVC's headers warns where it does not over GCC's: INFINITY is (float)(1e+300 * 1e+300) there, a double
  # that overflows (#221-D, at every use), and a variable that only some instances of a template use (an `if
  # constexpr` drops the use in the others) is "never referenced" or "set but never used" there (#177-D, #550-D).
  # About 4,650 lines a build, none of them in the Linux build of the same code.
  list(APPEND FLYX_CUDA_DEVICE_FLAGS --diag-suppress=177,221,550)
  if(FLYX_STATIC_RUNTIME)
    list(APPEND FLYX_CUDA_DEVICE_FLAGS -Xcompiler=/Gy)
  endif()
else()
  list(APPEND FLYX_CUDA_DEVICE_FLAGS -Xcompiler=-fvisibility=hidden)
  if(FLYX_STATIC_RUNTIME)
    list(APPEND FLYX_CUDA_DEVICE_FLAGS -Xcompiler=-ffunction-sections,-fdata-sections)
  endif()
endif()
if(FLYX_CUDA_LINEINFO)
  list(APPEND FLYX_CUDA_DEVICE_FLAGS -lineinfo)
endif()
if(FLYX_CUDA_COMPRESS)
  list(APPEND FLYX_CUDA_DEVICE_FLAGS --compress-mode=${FLYX_CUDA_COMPRESS})
endif()

function(flyx_configure_cuda_lib target archs)
  cmake_parse_arguments(ARG "USES_TABLES" "" "" ${ARGN})
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${archs}")
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${FLYX_CUDA_DEVICE_FLAGS}>")
  if(ARG_USES_TABLES)
    add_dependencies(${target} flyx_gemm_tables)
    target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
                               ${CMAKE_SOURCE_DIR}/third_party ${FLYX_GEN_DIR})
    target_compile_definitions(${target} PUBLIC FLYX_TUNE_CANDIDATES=$<IF:$<BOOL:${FLYX_TUNE_CANDIDATES}>,1,0>)
  else()
    target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)
  endif()
endfunction()

add_library(flyx_cuda_kernels STATIC ${FLYX_CUDA_KERNEL_SOURCES})
flyx_configure_cuda_lib(flyx_cuda_kernels "${FLYX_CUDA_ARCHS}" USES_TABLES)

set(flyx_compat_archs ${FLYX_CUDA_ARCHS})
if("80-real" IN_LIST flyx_compat_archs)
  list(REMOVE_ITEM flyx_compat_archs "86-real")
endif()
add_library(flyx_cuda_compat_kernels STATIC ${FLYX_CUDA_COMPAT_SOURCES})
flyx_configure_cuda_lib(flyx_cuda_compat_kernels "${flyx_compat_archs}" USES_TABLES)
target_link_libraries(flyx_cuda_kernels PUBLIC flyx_cuda_compat_kernels)

# Kernels for sm_120a only (block-scaled FP4 tensor cores): their own library, machine code only (an sm_120a kernel
# runs on no other GPU), a stub where sm_120 is not built.  The launchers check the device at run time.
set(flyx_arch_numbers ${FLYX_CUDA_ARCHS})
list(TRANSFORM flyx_arch_numbers REPLACE "-(real|virtual)$" "")
if("120" IN_LIST flyx_arch_numbers)
  add_library(flyx_cuda_sm120a STATIC kernels/gemm_w4a4.cu)
  flyx_configure_cuda_lib(flyx_cuda_sm120a "120a-real")
  target_link_libraries(flyx_cuda_kernels PUBLIC flyx_cuda_sm120a)
else()
  target_sources(flyx_cuda_kernels PRIVATE kernels/gemm_w4a4_none.cu)
endif()

set_source_files_properties(${FLYX_CUDA_TABLE_SOURCES} runtime/state.cpp
                            PROPERTIES OBJECT_DEPENDS ${FLYX_GEMM_TABLES})

# Host code only: built by the C++ compiler, kernels come from the static library.
set(FLYX_CUDA_HOST_SOURCES
  backend.cpp
  loader/bind.cpp loader/weights.cpp
  runtime/state.cpp runtime/runner.cpp runtime/graphs.cpp runtime/sleep.cpp runtime/device_stats.cpp runtime/tiers.cpp
  runtime/vision_disk.cpp
  regions/batch.cpp regions/layers.cpp regions/linear.cpp regions/plan.cpp)
add_library(flyx-cuda MODULE ${FLYX_CUDA_HOST_SOURCES})
target_link_libraries(flyx-cuda PRIVATE flyx_cuda_kernels CUDA::cudart_static ${CMAKE_DL_LIBS})
set_target_properties(flyx-cuda PROPERTIES PREFIX "lib" CUDA_ARCHITECTURES "${FLYX_CUDA_ARCHS}")

if(FLYX_STANDALONE_EXE)
  add_library(flyx_cuda_static STATIC ${FLYX_CUDA_HOST_SOURCES})
  target_compile_definitions(flyx_cuda_static PRIVATE FLYX_STATIC_BACKEND)
  target_link_libraries(flyx_cuda_static PUBLIC flyx_cuda_kernels CUDA::cudart_static ${CMAKE_DL_LIBS})
  set_target_properties(flyx_cuda_static PROPERTIES CUDA_ARCHITECTURES "${FLYX_CUDA_ARCHS}")
endif()

# The kernel tests (tests/backends/cuda) and timings (bench/microbench) link flyx_cuda_kernels.
