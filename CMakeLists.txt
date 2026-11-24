# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(flyx VERSION 0.2.0 LANGUAGES C CXX)

option(FLYX_CUDA "Build the CUDA backend (needs CUDA 13)" ON)
option(FLYX_TESTS "Build tests" ON)
option(FLYX_COMPILER_CACHE "Compile through sccache or ccache when one is installed" ON)

# A compiler cache, when installed: a source it has compiled before with the same flags (another commit's tree, a
# branch switched back, a clean build) costs a look-up instead of a compile.  sccache first: it caches nvcc's
# compiles too. A launcher the caller set is kept.
if(FLYX_COMPILER_CACHE)
  find_program(FLYX_CACHE_PROGRAM NAMES sccache ccache)
  if(FLYX_CACHE_PROGRAM)
    foreach(lang C CXX CUDA)
      if(NOT CMAKE_${lang}_COMPILER_LAUNCHER)
        set(CMAKE_${lang}_COMPILER_LAUNCHER ${FLYX_CACHE_PROGRAM})
      endif()
    endforeach()
  endif()
endif()
option(FLYX_STATIC_RUNTIME "Link the C++ runtime statically (release packages need only libc and the driver)" OFF)
option(FLYX_STANDALONE_EXE "Build a single-file executable (flyx-standalone) with backends linked in" OFF)
# The device code's line tables (-lineinfo): which source line each instruction came from, for profilers.  They and
# the PTX text they point into are most of a build's device code bytes; the release package leaves them out.  They
# never change the machine code.
option(FLYX_CUDA_LINEINFO "Line tables in the device code, for profilers" ON)
option(FLYX_TUNE_CANDIDATES "Compile every tuning candidate configuration (needed by cuda_bench tune)" ON)
set(FLYX_CUDA_COMPRESS "" CACHE STRING "nvcc --compress-mode of the device code (empty: nvcc's default)")

# CUDA architectures: FLYX_CUDA_ARCHS when given (the release package lists its own), otherwise those of the GPUs in
# this machine as nvidia-smi reports them, so that a build runs where it was built: their machine code only (-real),
# no PTX, which only a GPU without machine code of its own would use; 120 (RTX 50 series) when there is no GPU to
# ask.  Architectures before 8.0 are left out: the backend needs 8.0 or newer.
set(FLYX_CUDA_ARCHS "" CACHE STRING "CUDA architectures for the CUDA backend (empty: this machine's GPUs)")
if(FLYX_CUDA AND FLYX_CUDA_ARCHS STREQUAL "")
  find_program(FLYX_NVIDIA_SMI nvidia-smi HINTS /usr/lib/wsl/lib)
  set(flyx_found_archs "")
  if(FLYX_NVIDIA_SMI)
    execute_process(COMMAND ${FLYX_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE flyx_caps RESULT_VARIABLE flyx_smi_status ERROR_QUIET)
    if(flyx_smi_status EQUAL 0)
      string(REGEX MATCHALL "[0-9]+\\.[0-9]+" flyx_caps "${flyx_caps}")
      foreach(cap IN LISTS flyx_caps)
        string(REPLACE "." "" arch "${cap}")
        if(arch GREATER_EQUAL 80)
          list(APPEND flyx_found_archs ${arch}-real)
        endif()
      endforeach()
      list(REMOVE_DUPLICATES flyx_found_archs)
    endif()
  endif()
  if(flyx_found_archs)
    set(flyx_archs_from "this machine's GPUs")
  else()
    set(flyx_found_archs 120-real)
    set(flyx_archs_from "no GPU of compute capability 8.0 or newer found")
  endif()
  set(FLYX_CUDA_ARCHS "${flyx_found_archs}" CACHE STRING
      "CUDA architectures for the CUDA backend (empty: this machine's GPUs)" FORCE)
  message(STATUS "CUDA architectures: ${FLYX_CUDA_ARCHS} (${flyx_archs_from}; FLYX_CUDA_ARCHS chooses others)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(FLYX_STATIC_RUNTIME AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffunction-sections> $<$<COMPILE_LANGUAGE:C,CXX>:-fdata-sections>)
  add_link_options(-static-libstdc++ -static-libgcc -Wl,--gc-sections)
endif()
if(FLYX_STATIC_RUNTIME AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # the CRT in each binary
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Gy>)
  add_link_options(/OPT:REF /OPT:ICF)
endif()
if(MSVC)
  # Sources are UTF-8; template-heavy files outgrow the default section count; windows.h's min / max macros
  # would shadow std::min / std::max.  CCCL (CUDA 13.1 and later) refuses cl's traditional preprocessor, and the host
  # half of every CUDA file goes through cl too: the conforming one for both.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
                      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/bigobj,/Zc:__cplusplus,/Zc:preprocessor>")
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# __FILE__ as the path under the repository, not the build machine's: a vendored library's error texts carry it
# (xgrammar's reach API clients), and a release would otherwise say where on someone's disk it was built.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=>)
elseif(MSVC)
  file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/" FLYX_SOURCE_PREFIX)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/d1trimfile:${FLYX_SOURCE_PREFIX}>)
endif()

# Every program on Windows carries core/os/utf8.manifest: its ANSI code page is UTF-8 (Windows 10 1903 and later), so
# the C and C++ runtimes take the engine's UTF-8 strings as file names (std::ifstream, fopen: a prompt file, a
# --prompts list, the Hub's cache under a user folder named in Chinese), not the system's code page (GBK, ...).
function(flyx_program target)
  if(WIN32)
    target_sources(${target} PRIVATE ${CMAKE_SOURCE_DIR}/core/os/utf8.manifest)
  endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
endif()

# ---- third party ------------------------------------------------------------------
add_library(flyx_utf8proc STATIC third_party/utf8proc/utf8proc.c)
target_compile_definitions(flyx_utf8proc PUBLIC UTF8PROC_STATIC)
target_include_directories(flyx_utf8proc PUBLIC third_party/utf8proc)

# llama.cpp's Jinja engine, for chat templates (third_party/jinja/README.flyx.md).
add_library(flyx_jinja STATIC
  third_party/jinja/jinja/caps.cpp
  third_party/jinja/jinja/lexer.cpp
  third_party/jinja/jinja/parser.cpp
  third_party/jinja/jinja/runtime.cpp
  third_party/jinja/jinja/string.cpp
  third_party/jinja/jinja/value.cpp
  third_party/jinja/json.cpp
  third_party/jinja/unicode.cpp)
# Its headers say "json.h" and "jinja/value.h"; only the engine's own sources see the first.
target_include_directories(flyx_jinja PRIVATE third_party/jinja third_party)
target_include_directories(flyx_jinja SYSTEM INTERFACE third_party/jinja)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(flyx_jinja PRIVATE -w)  # third-party code, kept as upstream wrote it
endif()

# XGrammar's C++ core, for constrained decoding (third_party/xgrammar/README.flyx.md).
file(GLOB_RECURSE FLYX_XGRAMMAR_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/third_party/xgrammar/cpp/*.cc)
add_library(flyx_xgrammar STATIC ${FLYX_XGRAMMAR_SOURCES})
target_include_directories(flyx_xgrammar SYSTEM PUBLIC third_party/xgrammar/include
                           third_party/xgrammar/3rdparty/picojson third_party/xgrammar/3rdparty/dlpack/include)
target_compile_definitions(flyx_xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(flyx_xgrammar PRIVATE -w)  # third-party code, kept as upstream wrote it
endif()

find_package(Threads REQUIRED)

# ---- core (no CUDA allowed) -----------------------------------------------------------
if(WIN32)
  set(FLYX_OS_SOURCE core/os/os_windows.cpp core/os/http_windows.cpp)
else()
  set(FLYX_OS_SOURCE core/os/os_posix.cpp core/os/http_posix.cpp)
endif()
add_library(flyx_core STATIC
  ${FLYX_OS_SOURCE}
  core/os/http.cpp
  core/os/log.cpp
  core/hub/hub_names.cpp
  core/hub/hub_cache.cpp
  core/hub/hub_fetch.cpp
  core/hub/sha256.cpp
  core/modelfile/modelfile.cpp
  core/modelfile/flyx_format.cpp
  core/modelfile/gguf_format.cpp
  core/modelfile/drafter_dir.cpp
  core/tokenizer/tokenizer.cpp
  core/grammar/grammar.cpp
  core/render/chat.cpp
  core/render/chat_json.cpp
  core/vision/image.cpp
  core/plan/plans.cpp
  core/plan/budget.cpp
  core/session/session.cpp
  core/session/prefix_cache.cpp
  core/session/store.cpp
  core/exec/engine.cpp
  core/exec/generation.cpp
  core/exec/request.cpp
  core/exec/reply_text.cpp
  core/exec/scheduler.cpp
  core/exec/telemetry.cpp
  core/resources/ledger.cpp
  core/speculation/controller.cpp
  core/speculation/confidence.cpp
  core/speculation/lookup.cpp
  core/speculation/chooser.cpp
  core/speculation/batch_schedule.cpp
  core/speculation/calibrate.cpp)
target_include_directories(flyx_core PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(flyx_core PUBLIC flyx_utf8proc ${CMAKE_DL_LIBS} Threads::Threads)
target_link_libraries(flyx_core PRIVATE flyx_jinja flyx_xgrammar)

# ---- model definitions and chat formats: core reaches them only through core/interfaces/
# (interfaces::open_model, interfaces::open_chat); their objects go into the core library, so whatever links
# core has them.
add_library(flyx_models OBJECT models/model.cpp models/dv.cpp models/dflash2.cpp models/dspine.cpp models/llama/llama.cpp models/qwen35/qwen35.cpp
            models/vision/preprocess.cpp
            models/formats/formats.cpp models/formats/pytext.cpp models/formats/calls.cpp
            models/formats/tags.cpp models/formats/grammar.cpp models/formats/qwen35.cpp)
target_include_directories(flyx_models PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/third_party)
target_sources(flyx_core PRIVATE $<TARGET_OBJECTS:flyx_models>)

# C API as a shared library (Python bindings and other embedders link this).
# Compiling capi.cpp directly into the shared target lets the linker pull only the core objects the C API uses
# instead of whole-archiving server/hub code.
add_library(flyx SHARED core/exec/capi.cpp)
target_compile_definitions(flyx PRIVATE FLYX_BUILDING)
target_link_libraries(flyx PRIVATE flyx_core)

# One executable, flyx: the command line and, as `flyx serve`, the HTTP API, so the core is linked into one program,
# not two. flyx-server beside it is a link to it: called by that name it is `flyx serve`.
# The dashboard flyx serve serves at /, built in: no file to find next to the executable.
# Its files (apps/flyx-server/dashboard/) and the libraries it imports (Preact, htm) are served under /dashboard/.
file(GLOB_RECURSE FLYX_DASHBOARD_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard
     ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/*.html ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/*.js
     ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/*.css ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/*.svg
     ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/*.ico ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/*.png)
set(FLYX_DASHBOARD_LIST "")
set(FLYX_DASHBOARD_DEPS "")
foreach(f IN LISTS FLYX_DASHBOARD_SOURCES)
  string(APPEND FLYX_DASHBOARD_LIST "${f}=${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/${f}\n")
  list(APPEND FLYX_DASHBOARD_DEPS ${CMAKE_SOURCE_DIR}/apps/flyx-server/dashboard/${f})
endforeach()
foreach(f preact/preact.module.js preact/hooks.module.js htm/htm.module.js)
  get_filename_component(n ${f} NAME)
  string(APPEND FLYX_DASHBOARD_LIST "vendor/${n}=${CMAKE_SOURCE_DIR}/third_party/${f}\n")
  list(APPEND FLYX_DASHBOARD_DEPS ${CMAKE_SOURCE_DIR}/third_party/${f})
endforeach()
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generated/dashboard_files.txt CONTENT "${FLYX_DASHBOARD_LIST}")
set(FLYX_DASHBOARD_CPP ${CMAKE_BINARY_DIR}/generated/dashboard_files.cpp)
add_custom_command(OUTPUT ${FLYX_DASHBOARD_CPP}
                   COMMAND ${CMAKE_COMMAND} -DLIST=${CMAKE_BINARY_DIR}/generated/dashboard_files.txt
                           -DOUT=${FLYX_DASHBOARD_CPP} -DNAMESPACE=flyx::server -DNAME=kDashboardFiles
                           -P ${CMAKE_SOURCE_DIR}/cmake/embed_files.cmake
                   DEPENDS ${FLYX_DASHBOARD_DEPS} ${CMAKE_BINARY_DIR}/generated/dashboard_files.txt
                           ${CMAKE_SOURCE_DIR}/cmake/embed_files.cmake
                   COMMENT "Building the dashboard in" VERBATIM)
set(FLYX_CLI_SOURCES
    apps/flyx-cli/main.cpp apps/flyx-cli/monitor.cpp
    apps/flyx-server/main.cpp apps/flyx-server/http_util.cpp apps/flyx-server/openai_chat.cpp
    apps/flyx-server/openai_responses.cpp apps/flyx-server/anthropic_messages.cpp
    apps/flyx-server/telemetry.cpp apps/flyx-server/model_host.cpp apps/flyx-server/manage.cpp
    apps/flyx-server/request_log.cpp apps/flyx-server/system_log.cpp apps/flyx-server/agent_launch.cpp
    apps/flyx-server/gpu_apps.cpp
    ${FLYX_DASHBOARD_CPP}
    apps/common/engine_flags.cpp apps/common/numbers.cpp apps/common/model_source.cpp)
if(WIN32)
  list(APPEND FLYX_CLI_SOURCES apps/flyx-cli/flyx.rc)
endif()
add_executable(flyx-cli ${FLYX_CLI_SOURCES})
target_compile_definitions(flyx-cli PRIVATE FLYX_VERSION="${PROJECT_VERSION}"
                           FLYX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
                           FLYX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
                           FLYX_VERSION_PATCH=${PROJECT_VERSION_PATCH})
set_source_files_properties(${FLYX_DASHBOARD_CPP} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
target_link_libraries(flyx-cli PRIVATE flyx_core Threads::Threads)
set_target_properties(flyx-cli PROPERTIES OUTPUT_NAME flyx)
flyx_program(flyx-cli)
if(NOT WIN32)  # Windows: flyx serve (a link there needs the right to make one)
  add_custom_command(TARGET flyx-cli POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E create_symlink flyx ${CMAKE_BINARY_DIR}/bin/flyx-server
                     VERBATIM)
endif()

# Headers nearly every translation unit of these targets reads, parsed once per target instead of once per file:
# nlohmann/json is about 25,000 lines (core's interfaces carry it everywhere), cpp-httplib about 10,000.
set(FLYX_PCH_JSON ${CMAKE_SOURCE_DIR}/third_party/nlohmann/json.hpp)
set(FLYX_PCH_HTTP ${CMAKE_SOURCE_DIR}/third_party/httplib/httplib.h)
target_precompile_headers(flyx_core PRIVATE ${FLYX_PCH_JSON})
target_precompile_headers(flyx_models PRIVATE ${FLYX_PCH_JSON})
target_precompile_headers(flyx-cli PRIVATE ${FLYX_PCH_JSON} ${FLYX_PCH_HTTP})

# ---- backends: one loadable module each ---------------------------------------------
add_library(flyx-cpu_ref MODULE backends/cpu_ref/cpu_ref.cpp)
target_include_directories(flyx-cpu_ref PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/third_party)
set_target_properties(flyx-cpu_ref PROPERTIES PREFIX "lib")

if(FLYX_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # For every directory with CUDA sources (the backend, its kernel tests, the
    # microbenchmarks): the CUDA runtime is linked statically, so the only thing
    # a user needs installed is the NVIDIA driver.
    enable_language(CUDA)
    # FindCUDAToolkit looks for some 80 libraries (we link one: cudart_static). Under WSL the PATH carries the
    # Windows PATH, dozens of /mnt/c directories that are slow to search: the look-up took 75 of a 76 s configure.
    # The toolkit is found from the compiler, not from PATH.
    set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)
    find_package(CUDAToolkit 13.0 REQUIRED)
    unset(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
    add_subdirectory(backends/cuda)
  else()
    message(WARNING "CUDA compiler not found: building without the CUDA backend")
  endif()
endif()

if(FLYX_STANDALONE_EXE)
  add_library(flyx_cpu_ref_static STATIC backends/cpu_ref/cpu_ref.cpp)
  target_compile_definitions(flyx_cpu_ref_static PRIVATE FLYX_STATIC_BACKEND)
  target_include_directories(flyx_cpu_ref_static PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
                             ${CMAKE_SOURCE_DIR}/third_party)
  add_executable(flyx-standalone ${FLYX_CLI_SOURCES} apps/flyx-cli/builtin_backends.cpp)
  target_compile_definitions(flyx-standalone PRIVATE FLYX_STANDALONE_EXE=1 FLYX_VERSION="${PROJECT_VERSION}"
                             FLYX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
                             FLYX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
                             FLYX_VERSION_PATCH=${PROJECT_VERSION_PATCH})
  target_link_libraries(flyx-standalone PRIVATE flyx_core flyx_cpu_ref_static Threads::Threads)
  if(TARGET flyx_cuda_static)
    target_compile_definitions(flyx-standalone PRIVATE FLYX_HAVE_BUILTIN_CUDA=1)
    target_link_libraries(flyx-standalone PRIVATE flyx_cuda_static)
  endif()
  target_precompile_headers(flyx-standalone PRIVATE ${FLYX_PCH_JSON} ${FLYX_PCH_HTTP})
  flyx_program(flyx-standalone)
endif()

# ---- install: the layout of the release package ------------------------------------------
#   bin/flyx, bin/flyx-server (a link)  (find backends in ../lib/flyx; Windows: flyx.exe, no link)
#   lib/libflyx.so                         C API (Windows: flyx.dll, flyx.lib)
#   lib/flyx/libflyx-<backend>.so       backends, loaded at start-up (Windows: .dll)
#   include/flyx/flyx.h, python/flyx (ctypes binding)
install(TARGETS flyx-cli RUNTIME DESTINATION bin)
if(NOT WIN32)
  install(CODE "file(CREATE_LINK flyx \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/flyx-server\" SYMBOLIC)")
endif()
install(TARGETS flyx LIBRARY DESTINATION lib RUNTIME DESTINATION lib ARCHIVE DESTINATION lib)  # Windows: flyx.dll, flyx.lib
install(TARGETS flyx-cpu_ref LIBRARY DESTINATION lib/flyx)
if(TARGET flyx-cuda)
  install(TARGETS flyx-cuda LIBRARY DESTINATION lib/flyx)
endif()
install(FILES include/flyx/flyx.h DESTINATION include/flyx)
install(DIRECTORY python/flyx DESTINATION python PATTERN "__pycache__" EXCLUDE)
install(FILES README.md LICENSE NOTICE DESTINATION .)
install(DIRECTORY third_party/ DESTINATION third_party FILES_MATCHING
        PATTERN "LICENSE*" PATTERN "NOTICE" PATTERN "README.*.md" PATTERN "picojson.h")

# ---- tests and microbenchmarks ------------------------------------------------------------
if(FLYX_TESTS)
  if(NOT Python3_FOUND)
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  endif()
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(bench)
endif()
