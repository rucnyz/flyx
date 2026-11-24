# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# Tests, laid out like the sources they check:
#   core/tokenizer/      tokenizer vs HuggingFace on recorded cases; its BPE vs the rule on vocabularies made there
#   core/modelfile/      the GGUF reader (the container); make_gguf.py writes the GGUF fixtures with the gguf package;
#                        the .flyx reader on files that are wrong (write_flyx.h writes them)
#   ops/dtype/           GGML block weights as float64 vs the gguf package's decoder
#   ops/                 the float64 references' sums side by side: each output the bits of its sum alone; DV's
#                        trees, selector table (ops/draft/) and canvas (ops/canvas/) vs the Python reference
#   models/              a model's description from a GGUF file means the one from its .flyx file; DV's description
#                        (blocks, state, programs, the drafter chosen) from the tiny Llama's DV file
#   models/drafters/     the DFlash2 drafter the architectures share: its config, the dense targets that take one
#                        from a GGUF drafter file, drafter files that are refused
#   core/exec/           telemetry (what the server reports to monitors), the request pipeline, a reply's text,
#                        the scheduler (requests taking turns), the C API
#   core/render/         chat templates vs transformers
#   core/speculation/    which round a session launches next (the drafter's widths, lookup rounds)
#   core/plan/           the memory budget (a context shrinks to what fits); the round plans of a drafter that drafts
#                        alone (DV)
#   models/formats/      chat formats: reading replies back; renderers written for a template vs the Jinja engine;
#                        reply grammars
#   backends/cuda/       every CUDA kernel vs the float64 ops (ops/)
#   e2e/                 the whole engine vs the Python reference (golden; make_tiny.py builds the tiny
#                        models and their golden outputs), and a real model vs transformers; a drafter on a
#                        prompt set vs plain decoding and bifill-dv's references (check_dv.py: its counting and
#                        scoring, and its rounds against the teacher-forced reference's, tested in check_dv_test.py)
#   apps/flyx-server/    the stored responses and the HTTP API (run against a live server)
#   apps/flyx-cli/       the command line's file paths and flags, flyx bench on a prompt set (the tiny model)
#   apps/common/         numbers given on the command line
#   bench/ab/            two builds' records of one prompt set compared answer for answer, round for round
#                        (same_runs.py)
#   python/reference/    the Python reference vs transformers; its DV drafting emits plain decoding's tokens, its
#                        canvas is transformers', the DV runners' results files keep their settings (free and
#                        teacher-forced)
#   python/convert/      the converter: configs and tokenizer pipelines it refuses, a post-processor's prefix; DV's
#                        export and what is refused of it; a DV file's tensors are its sources' (check_dv_convert.py)

function(flyx_add_unit_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE flyx_core ${ARGN})
endfunction()

function(flyx_add_py_test name script)
  cmake_parse_arguments(ARG "" "TIMEOUT;SKIP_RETURN_CODE" "ARGS;FIXTURES;ENVIRONMENT;PY_FLAGS" ${ARGN})
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${ARG_PY_FLAGS} ${script} ${ARG_ARGS})
  if(ARG_FIXTURES)
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED "${ARG_FIXTURES}")
  endif()
  if(ARG_TIMEOUT)
    set_tests_properties(${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_SKIP_RETURN_CODE)
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE ${ARG_SKIP_RETURN_CODE})
  endif()
  if(ARG_ENVIRONMENT)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${ARG_ENVIRONMENT}")
  endif()
endfunction()

function(flyx_add_golden_test name)
  cmake_parse_arguments(ARG "SERIAL" "ENV" "ARGS;FIXTURES" ${ARGN})
  add_test(NAME ${name} COMMAND golden_test ${ASSETS} ${ARG_ARGS})
  set(fixtures ${ARG_FIXTURES})
  if(NOT fixtures)
    set(fixtures assets)
  endif()
  set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED "${fixtures}")
  if(ARG_ENV)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${ARG_ENV}")
  endif()
  if(ARG_SERIAL)
    set_tests_properties(${name} PROPERTIES RUN_SERIAL TRUE)
  endif()
endfunction()

flyx_add_unit_test(tokenizer_test core/tokenizer/tokenizer_test.cpp)
flyx_add_unit_test(merge_test core/tokenizer/merge_test.cpp)
flyx_add_unit_test(flyx_format_test core/modelfile/flyx_test.cpp)
flyx_add_unit_test(drafter_dir_test core/modelfile/drafter_dir_test.cpp)
flyx_add_unit_test(hub_test core/hub/hub_test.cpp Threads::Threads)
flyx_add_unit_test(chat_test core/render/chat_test.cpp)
flyx_add_unit_test(tags_test models/formats/tags_test.cpp)
flyx_add_unit_test(qwen35_format_test models/formats/qwen35_test.cpp)
flyx_add_unit_test(grammar_test models/formats/grammar_test.cpp)
flyx_add_unit_test(telemetry_test core/exec/telemetry_test.cpp)
flyx_add_unit_test(snapshot_store_test core/session/store_test.cpp)
flyx_add_unit_test(parked_edit_test core/session/parked_edit_test.cpp)
flyx_add_unit_test(request_test core/exec/request_test.cpp)
flyx_add_unit_test(reply_text_test core/exec/reply_text_test.cpp)
flyx_add_unit_test(chooser_test core/speculation/chooser_test.cpp)
flyx_add_unit_test(calibrate_test core/speculation/calibrate_test.cpp)
flyx_add_unit_test(budget_test core/plan/budget_test.cpp)
flyx_add_unit_test(golden_test e2e/golden_test.cpp)
flyx_add_unit_test(gguf_test core/modelfile/gguf_test.cpp)
flyx_add_unit_test(ggml_test ops/dtype/ggml_test.cpp)
flyx_add_unit_test(sums_test ops/sums_test.cpp)
flyx_add_unit_test(dv_tree_test ops/draft/dv_tree_test.cpp)
flyx_add_unit_test(canvas_test ops/canvas/canvas_test.cpp)
flyx_add_unit_test(gguf_models_test models/gguf_test.cpp)
flyx_add_unit_test(dflash2_models_test models/drafters/dflash2_test.cpp)
flyx_add_unit_test(dv_models_test models/dv_test.cpp)
flyx_add_unit_test(llama_models_test models/llama/llama_test.cpp)
flyx_add_unit_test(gguf_e2e_test e2e/gguf_test.cpp)
flyx_add_unit_test(vision_test e2e/vision_test.cpp)
add_dependencies(vision_test flyx-cpu_ref)
if(TARGET flyx-cuda)
  add_dependencies(vision_test flyx-cuda)
endif()

add_executable(numbers_test apps/common/numbers_test.cpp ${CMAKE_SOURCE_DIR}/apps/common/numbers.cpp)
target_include_directories(numbers_test PRIVATE ${CMAKE_SOURCE_DIR})
add_executable(strata_snapshot_test backends/cuda/strata_snapshot_test.cpp)
target_include_directories(strata_snapshot_test PRIVATE ${CMAKE_SOURCE_DIR})
add_executable(response_store_test apps/flyx-server/response_store_test.cpp)
target_include_directories(response_store_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(response_store_test PRIVATE Threads::Threads)
add_executable(request_log_test apps/flyx-server/request_log_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/request_log.cpp)
target_include_directories(request_log_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(request_log_test PRIVATE flyx_core Threads::Threads)
add_executable(agent_launch_test apps/flyx-server/agent_launch_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/agent_launch.cpp)
target_include_directories(agent_launch_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party)
add_executable(gpu_apps_test apps/flyx-server/gpu_apps_test.cpp ${CMAKE_SOURCE_DIR}/apps/flyx-server/gpu_apps.cpp)
target_include_directories(gpu_apps_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(gpu_apps_test PRIVATE flyx_core)

add_library(flyx-test-host-kv SHARED core/session/backend_adapter.cpp)
target_include_directories(flyx-test-host-kv PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(flyx-test-host-kv PROPERTIES PREFIX "lib")
flyx_add_unit_test(session_budget_test core/session/budget_test.cpp flyx-test-host-kv)
add_dependencies(session_budget_test flyx-cpu_ref)
flyx_add_unit_test(generation_test core/exec/generation_test.cpp flyx-test-host-kv)
add_dependencies(generation_test flyx-cpu_ref)
flyx_add_unit_test(plans_test core/plan/plans_test.cpp)
flyx_add_unit_test(scheduler_test core/exec/scheduler_test.cpp flyx-test-host-kv)

add_executable(capi_test core/exec/capi_test.cpp ${CMAKE_SOURCE_DIR}/core/exec/capi.cpp)
target_compile_definitions(capi_test PRIVATE FLYX_BUILDING)
target_link_libraries(capi_test PRIVATE flyx_core)

# Tiny random model (BF16 and an NVFP4 copy) + golden outputs from the Python
# reference (needs torch, transformers, tokenizers, safetensors).  Generated
# once per build tree.
set(ASSETS ${CMAKE_BINARY_DIR}/assets)
add_test(NAME make_assets
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/e2e/make_tiny.py ${ASSETS})
set_tests_properties(make_assets PROPERTIES FIXTURES_SETUP assets)
# The tiny models again as GGUF files, and random blocks of every GGML block
# type with what the gguf package (a test dependency) decodes them to.
add_test(NAME make_gguf_assets
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/core/modelfile/make_gguf.py ${ASSETS})
set_tests_properties(make_gguf_assets PROPERTIES FIXTURES_REQUIRED assets FIXTURES_SETUP gguf_assets)

add_test(NAME gguf_reader COMMAND gguf_test ${ASSETS})
add_test(NAME ggml_blocks COMMAND ggml_test ${ASSETS})
add_test(NAME gguf_models COMMAND gguf_models_test ${ASSETS})
# The DFlash2 drafter the architectures share (models/drafters/): its config checked before a tensor is bound, the
# dense targets that take one from llama.cpp's GGUF layout, and what is refused of a drafter file.
add_test(NAME dflash2_models COMMAND dflash2_models_test ${ASSETS})
# Qwen2's pre-tokenizer split (the published Qwen3 checkpoints), from GGUF vocabularies, against tokenizers.
add_test(NAME gguf_qwen2_nfc COMMAND tokenizer_test ${ASSETS}/qwen2_nfc.gguf ${ASSETS}/qwen2_nfc.json)
add_test(NAME gguf_qwen2_none COMMAND tokenizer_test ${ASSETS}/qwen2_none.gguf ${ASSETS}/qwen2_none.json)
# The tiny models from their GGUF files: the golden outputs, drafting, sampling, reuse.
add_test(NAME gguf_cpu_ref COMMAND gguf_e2e_test ${ASSETS} cpu_ref)
flyx_add_golden_test(golden_cpu_ref_qwen3_gguf ARGS cpu_ref qwen3 ${ASSETS}/tiny_qwen3_gguf.gguf FIXTURES gguf_assets)
set_tests_properties(gguf_reader ggml_blocks gguf_models dflash2_models gguf_qwen2_nfc gguf_qwen2_none gguf_cpu_ref
                     PROPERTIES FIXTURES_REQUIRED gguf_assets)

flyx_add_py_test(reference_vs_transformers ${CMAKE_SOURCE_DIR}/tests/python/reference/check_vs_hf.py
                 ARGS ${ASSETS}/hf_target ${ASSETS}/tiny.flyx FIXTURES assets)
add_test(NAME tokenizer COMMAND tokenizer_test ${ASSETS}/tiny.flyx ${ASSETS}/tokenizer_cases.json)
# MiniCPM5's pre-tokenizer (numbers in groups of three first), on the tiny Llama's tokenizer, against tokenizers.
add_test(NAME tokenizer_minicpm5
         COMMAND tokenizer_test ${ASSETS}/tiny_llama.flyx ${ASSETS}/tokenizer_cases_minicpm5.json)
add_test(NAME relative_backend_dir COMMAND flyx-cli run ${ASSETS}/tiny.flyx --backend cpu_ref -p hello -n 1)
set_tests_properties(relative_backend_dir PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:flyx-cli>"
                     ENVIRONMENT "FLYX_BACKEND_DIR=." FIXTURES_REQUIRED assets)

flyx_add_py_test(cli_utf8_paths ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-cli/utf8_paths.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(kv_method_flags ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-cli/kv_method.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(cli_usage ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-cli/usage.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(remembered_model ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/remembered_model.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)
flyx_add_py_test(queued_request_ttft ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/queued_request_ttft.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(streaming_tool_calls ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/streaming_tool_calls.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(api_compat ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/api_compat.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(server_hardening ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/hardening.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 120)
flyx_add_py_test(queued_request_log ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/queued_request_log.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(client_conversations ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/client_conversations.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(codex_threads ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/codex_threads.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets TIMEOUT 90)
flyx_add_py_test(model_timestamps ${CMAKE_CURRENT_SOURCE_DIR}/apps/flyx-server/model_timestamps.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS}/tiny.flyx FIXTURES assets)

add_test(NAME chat COMMAND chat_test ${ASSETS}/tiny.flyx ${ASSETS}/chat_cases.json
                                     ${CMAKE_SOURCE_DIR}/tests/core/render/templates)
# The tokenizer's BPE against the rule a pass per merge, and the .flyx reader on wrong files: on files the tests
# write themselves (no model, no Python).
add_test(NAME bpe_merges COMMAND merge_test ${CMAKE_BINARY_DIR})
add_test(NAME flyx_reader COMMAND flyx_format_test ${CMAKE_BINARY_DIR})
# Models named as the Hugging Face Hub names them: fetching from a fake Hub on this machine, and a
# DFlash2 drafter read from its authors' directory as python/convert writes it.
add_test(NAME hub COMMAND hub_test)
set_tests_properties(hub PROPERTIES SKIP_RETURN_CODE 77)
flyx_add_py_test(hub_concurrent ${CMAKE_CURRENT_SOURCE_DIR}/core/hub/check_concurrent_fetch.py
                 ARGS $<TARGET_FILE:hub_test> SKIP_RETURN_CODE 77 TIMEOUT 90)
add_test(NAME drafter_dir COMMAND drafter_dir_test ${ASSETS})
set_tests_properties(drafter_dir PROPERTIES FIXTURES_REQUIRED gguf_assets)

# Numbers on the command line: all of the text a number, inside what the option takes.
add_test(NAME flag_numbers COMMAND numbers_test)
add_test(NAME chat_replies COMMAND tags_test)
# Qwen3.5's own renderer against the Jinja engine on generated conversations, and its calls read back.
add_test(NAME chat_qwen35 COMMAND qwen35_format_test ${ASSETS}/tiny.flyx
                                  ${CMAKE_SOURCE_DIR}/tests/core/render/templates/qwen3.5.jinja)
# Reply grammars (tool_choice, response_format) in both call languages, on the tiny model's tokenizer.
add_test(NAME reply_grammars COMMAND grammar_test ${ASSETS}/tiny.flyx)
add_test(NAME telemetry COMMAND telemetry_test)
add_test(NAME snapshot_store COMMAND snapshot_store_test)
add_test(NAME strata_snapshot COMMAND strata_snapshot_test)
add_test(NAME parked_edit COMMAND parked_edit_test ${ASSETS}/tiny.flyx)
set_tests_properties(parked_edit PROPERTIES FIXTURES_REQUIRED assets)
add_test(NAME response_store COMMAND response_store_test)
add_test(NAME request_log COMMAND request_log_test)
add_test(NAME agent_launch COMMAND agent_launch_test)
add_test(NAME gpu_apps COMMAND gpu_apps_test)
# Which round next: the drafter's widths by acceptance, lookup rounds, the rest after a poor one (no model).
add_test(NAME round_chooser COMMAND chooser_test)
# Round costs timed at load at two contexts: what a device takes, not what another program's work made them (no model).
add_test(NAME round_costs_plausible COMMAND calibrate_test)
# The memory budget: a context the caller did not ask for shrinks to what fits (no model, no GPU).
add_test(NAME memory_budget COMMAND budget_test)
# The round plans: a drafter whose state follows its own rounds (DV) beside plain decoding's and lookup's (pinned:
# alone) where the backend runs them, else as without it.
add_test(NAME round_plans COMMAND plans_test)
# A reply's text: characters shown whole, stop strings, which tokens count as shown (the tiny model's tokenizer).
add_test(NAME reply_text COMMAND reply_text_test ${ASSETS}/tiny.flyx)
# The request pipeline every entry point shares, on the tiny model (float64 backend) with Qwen3.5's template; a raw
# prompt on the tiny Llama, whose tokenizer puts <s> before a text (MiniCPM5's post-processor).
add_test(NAME request COMMAND request_test ${ASSETS}/tiny.flyx
                                 ${CMAKE_SOURCE_DIR}/tests/core/render/templates/qwen3.5.jinja
                                 ${ASSETS}/tiny_llama.flyx)
add_test(NAME ref_sums COMMAND sums_test)
add_test(NAME vision COMMAND vision_test)
set_tests_properties(vision PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vision_test>")
# DV's float64 references against the Python reference's cases (make_tiny.py's dv_cases.json): trees on 1000 random
# tables of each kind and edge cases, cov's term, selector tables, the canvas's input, statistics and slide.
add_test(NAME dv_trees COMMAND dv_tree_test ${ASSETS})
add_test(NAME dv_canvas COMMAND canvas_test ${ASSETS})
# The Python reference's DV drafting on the tiny Llama emits its plain decoding's tokens (both trees, a V-LoRA); its
# canvas rows are transformers' Llama with the canvas adapter merged in, and its trees grow again from those canvases,
# also when it commits another run's counts.
flyx_add_py_test(reference_dv ${CMAKE_SOURCE_DIR}/tests/python/reference/dv_test.py ARGS ${ASSETS} FIXTURES assets)
# DV's description from the tiny Llama's DV file (models/dv.h): its blocks, canvas and programs, the drafter the
# options choose, sections that cannot be drafted refused.
add_test(NAME dv_models COMMAND dv_models_test ${ASSETS})
# The target's own adapters (a V-LoRA, ABI 43) from the tiny Llama's V-LoRA files (models/llama): on every linear of
# every layer, their rank and scale; headers and tensors that do not fit refused.
add_test(NAME llama_models COMMAND llama_models_test ${ASSETS})
# The tiny DV files' tensors are export_dv.py's, and the reference loads them to their checkpoints' and transformers'
# weights: the beam tree, the pool tree (a causal canvas) and the canvas of 32 slots.
flyx_add_py_test(dv_convert ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv.flyx ${ASSETS}/dv_tiny/export --canvas ${ASSETS}/dv_tiny/canvas.pt
                      --selector ${ASSETS}/dv_tiny/selector.pt --hf ${ASSETS}/hf_llama
                 FIXTURES assets)
flyx_add_py_test(dv_convert_pool ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv_pool.flyx ${ASSETS}/dv_tiny/export_pool
                      --canvas ${ASSETS}/dv_tiny/canvas_causal.pt --selector ${ASSETS}/dv_tiny/selector.pt
                      --hf ${ASSETS}/hf_llama
                 FIXTURES assets)
flyx_add_py_test(dv_convert_32 ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv32.flyx ${ASSETS}/dv_tiny/export_32 --canvas ${ASSETS}/dv_tiny/canvas.pt
                      --selector ${ASSETS}/dv_tiny/selector.pt --hf ${ASSETS}/hf_llama
                 FIXTURES assets)
# The same with a V-LoRA beside the DV (its adapters are bifill-dv's checkpoint's, byte for byte, as the reference
# loads them), and a V-LoRA alone (another scale).
flyx_add_py_test(dv_convert_lora ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_dv_lora.flyx ${ASSETS}/dv_tiny/export_dv_lora
                      --canvas ${ASSETS}/dv_tiny/canvas.pt --selector ${ASSETS}/dv_tiny/selector.pt
                      --vlora ${ASSETS}/dv_tiny/vlora32.pt --hf ${ASSETS}/hf_llama
                 FIXTURES assets)
flyx_add_py_test(lora_convert ${CMAKE_SOURCE_DIR}/tests/python/convert/check_dv_convert.py PY_FLAGS -I
                 ARGS ${ASSETS}/tiny_llama_lora.flyx ${ASSETS}/dv_tiny/export_vlora --vlora ${ASSETS}/dv_tiny/vlora16.pt
                      --hf ${ASSETS}/hf_llama
                 FIXTURES assets)
# The bf16 prefill residual's decision on each backend (-v): taken by an all-W4A4 stack, never with adapters.
if(TARGET flyx-cuda)
  set(FLYX_RESIDUAL_BACKENDS cpu_ref cuda)
else()
  set(FLYX_RESIDUAL_BACKENDS cpu_ref)
endif()
flyx_add_py_test(residual_decision ${CMAKE_SOURCE_DIR}/tests/e2e/residual_decision.py PY_FLAGS -I
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS} ${FLYX_RESIDUAL_BACKENDS} FIXTURES assets)
# The DV runner's results file goes on only with the settings and thread count that made it, past a row a stopped run
# cut off (no model is loaded).
flyx_add_py_test(reference_dv_gsm8k ${CMAKE_SOURCE_DIR}/tests/python/reference/dv_gsm8k_test.py)
# The DV reference forced to a bench record's answers (and replaying its rounds): what it is forced to, the rows and
# settings it keeps, its rounds against a free run's where the answers agree (a stand-in for the model); whole runs on
# the tiny Llama's DV: a .flyx file and its checkpoints run the same rounds, the options that choose questions,
# threads, a V-LoRA, a stopped run going on.
flyx_add_py_test(reference_dv_forced ${CMAKE_SOURCE_DIR}/tests/python/reference/dv_forced_test.py ARGS ${ASSETS}
                 FIXTURES assets)
# What the converter refuses (a scaled rotary in either key's form) and the tokens a post-processor puts before a text.
flyx_add_py_test(converter ${CMAKE_SOURCE_DIR}/tests/python/convert/convert_test.py)
# A drafter's answers and tokens per forward counted as bifill-dv counts them, GSM8K scored as it scores them, its
# rounds against the teacher-forced reference's, and what the checks refuse (tests/e2e/check_dv.py, dv_ref_json.py;
# no model).
flyx_add_py_test(dv_counts ${CMAKE_SOURCE_DIR}/tests/e2e/check_dv_test.py)
# flyx bench --prompts: each line's prompt, the generated tokens and rounds in the record (an end token accepted as a
# draft marked), drafted answers plain's, bad prompts and output paths refused before anything runs.
flyx_add_py_test(bench_prompts ${CMAKE_SOURCE_DIR}/tests/apps/flyx-cli/bench_prompts_test.py
                 ARGS $<TARGET_FILE:flyx-cli> ${ASSETS} ${CMAKE_SOURCE_DIR}/tests/core/render/templates/qwen3.5.jinja
                 FIXTURES assets)
# Two builds' flyx bench --prompts records of one command: the same answers and rounds over the prompts both hold, and
# what fails (bench/ab/same_runs.py; records made up, no model).
flyx_add_py_test(same_runs ${CMAKE_SOURCE_DIR}/tests/bench/ab/same_runs_test.py)
# DV's rounds chosen, replayed on the kept traces (bench/replay/dv_choice): the sets made from them,
# DV pinned and plain decoding at the traces' speeds, a constant set on the command line taking effect, and what is
# refused.
flyx_add_py_test(dv_choice_replay ${CMAKE_SOURCE_DIR}/tests/bench/replay/dv_choice_test.py
                 ARGS $<TARGET_FILE:dv_choice>)
flyx_add_py_test(layering ${CMAKE_SOURCE_DIR}/tools/check_layering.py ARGS ${CMAKE_SOURCE_DIR})
flyx_add_py_test(numerics ${CMAKE_SOURCE_DIR}/tools/check_numerics.py ARGS ${CMAKE_SOURCE_DIR})
flyx_add_py_test(docs ${CMAKE_SOURCE_DIR}/tools/check_docs.py ARGS ${CMAKE_SOURCE_DIR})
flyx_add_py_test(agent_replay ${CMAKE_CURRENT_SOURCE_DIR}/bench/replay/agent_replay_test.py)
flyx_add_py_test(trace_layer_roles ${CMAKE_CURRENT_SOURCE_DIR}/bench/trace/layer_roles_test.py)
flyx_add_py_test(two_at_once ${CMAKE_CURRENT_SOURCE_DIR}/bench/replay/two_at_once_test.py TIMEOUT 15)
if(UNIX)
  flyx_add_py_test(drafter_captures ${CMAKE_CURRENT_SOURCE_DIR}/bench/drafter_captures.py)
  flyx_add_py_test(external_comparisons ${CMAKE_CURRENT_SOURCE_DIR}/bench/external_comparisons.py)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  flyx_add_py_test(quickstart_venv ${CMAKE_CURRENT_SOURCE_DIR}/tools/quickstart_venv.py)
  flyx_add_py_test(cpu_session_allocation ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu_ref/allocation_test.py
                   ARGS ${ASSETS}/tiny.flyx $<TARGET_FILE:flyx> FIXTURES assets)
endif()
# Tuning tables: a GPU's own rules for any family, inherited ones, and the tables refused.
flyx_add_py_test(gemm_tables ${CMAKE_SOURCE_DIR}/tests/tools/codegen/gen_gemm_tables_test.py ARGS ${CMAKE_SOURCE_DIR})

# The scheduler: requests at once take turns and give what they give alone; a busy session is waited for.
add_test(NAME scheduler_cpu_ref COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref)
add_test(NAME scheduler_queue COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref queue)
add_test(NAME scheduler_prewarmed_history COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref history)
add_test(NAME scheduler_interrupted_prewarm COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref interrupted-prewarm)
add_test(NAME scheduler_reassigned_prewarm COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref reassigned-prewarm)
add_test(NAME scheduler_repeated_prewarm COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref repeated-prewarm)
add_test(NAME scheduler_marked_system COMMAND scheduler_test ${ASSETS}/tiny.flyx cpu_ref marked-system)
add_test(NAME session_host_budget COMMAND session_budget_test ${ASSETS}/tiny.flyx)
add_test(NAME generation_edges COMMAND generation_test ${ASSETS}/tiny.flyx)
# The C API: a chat request with the file's template, and with the one the load options give.
add_test(NAME capi COMMAND capi_test ${ASSETS}/tiny.flyx cpu_ref)
flyx_add_py_test(python_binding ${CMAKE_CURRENT_SOURCE_DIR}/python/flyx/binding_test.py
                 ARGS ${ASSETS}/tiny.flyx $<TARGET_FILE:flyx> FIXTURES assets)
flyx_add_py_test(convert_utf8 ${CMAKE_CURRENT_SOURCE_DIR}/python/convert/utf8_test.py
                 PY_FLAGS -X utf8=0 ARGS ${ASSETS} $<TARGET_FILE:flyx>
                 FIXTURES assets ENVIRONMENT "LC_ALL=C" "PYTHONCOERCECLOCALE=0")
flyx_add_py_test(convert_write ${CMAKE_CURRENT_SOURCE_DIR}/python/convert/write_test.py)
flyx_add_py_test(convert_cli ${CMAKE_CURRENT_SOURCE_DIR}/python/convert/cli_test.py
                 PY_FLAGS -X utf8=1 ARGS ${ASSETS} FIXTURES assets)

flyx_add_golden_test(golden_cpu_ref ARGS cpu_ref)
flyx_add_golden_test(golden_cpu_ref_nvfp4 ARGS cpu_ref nvfp4)
# A second architecture (Qwen3) through models/, against transformers: no backend knows it by name.
flyx_add_golden_test(golden_cpu_ref_qwen3 ARGS cpu_ref qwen3)
# Llama's attention without q/k norms (MiniCPM5's layers), against transformers.
flyx_add_golden_test(golden_cpu_ref_llama ARGS cpu_ref llama)
# The same with its DV drafter in the file: as loaded by default (cpu_ref has no DV rounds) and with --drafter none;
# with its beam tree and with a pool tree (whose tree fields are not zeros).
flyx_add_golden_test(golden_cpu_ref_llama_dv ARGS cpu_ref llama ${ASSETS}/tiny_llama_dv.flyx)
flyx_add_golden_test(golden_cpu_ref_llama_dv_pool ARGS cpu_ref llama ${ASSETS}/tiny_llama_dv_pool.flyx)
# The tiny Llama with a V-LoRA of its own (the target's adapters, ABI 43) against transformers with it merged; and
# beside its DV drafter (decoded as without DV here).
flyx_add_golden_test(golden_cpu_ref_llama_lora ARGS cpu_ref llama_lora)
flyx_add_golden_test(golden_cpu_ref_llama_dv_lora ARGS cpu_ref llama_dv_lora)
# GDN heads of 128: the CUDA prefill takes the chunked recurrence.
flyx_add_golden_test(golden_cpu_ref_gdn128 ARGS cpu_ref gdn128)
# A DFlash2 drafter whose layers attend a sliding window (z-lab's 27B drafter's), shorter than its block here.
flyx_add_golden_test(golden_cpu_ref_window ARGS cpu_ref window)
# A DSpine drafter in place of DFlash2's (the whole battery: its rounds, oracle runs, rollback, sampling, ...), and one
# whose layers attend a causal sliding window of 8 (the 27B's drafter's), shorter than its block.
flyx_add_golden_test(golden_cpu_ref_dspine ARGS cpu_ref dspine)
flyx_add_golden_test(golden_cpu_ref_dspine1 ARGS cpu_ref dspine1)  # a drafter of one layer
flyx_add_golden_test(golden_cpu_ref_dspine_window ARGS cpu_ref dspine_window)
# The tiny Qwen3 with a DSpine drafter (taps and context in a second architecture's programs): transformers' tokens.
flyx_add_golden_test(golden_cpu_ref_qwen3_dspine ARGS cpu_ref qwen3 ${ASSETS}/tiny_qwen3_dspine.flyx)

set_tests_properties(tokenizer tokenizer_minicpm5 chat chat_qwen35 reply_grammars request reply_text
                     scheduler_cpu_ref scheduler_queue scheduler_prewarmed_history scheduler_interrupted_prewarm
                     scheduler_reassigned_prewarm scheduler_repeated_prewarm scheduler_marked_system session_host_budget generation_edges capi
                     dv_trees dv_canvas dv_models llama_models PROPERTIES FIXTURES_REQUIRED assets)

if(TARGET flyx-cuda)
  # Packed block weights against the float64 backend on the same files; the generic regions too.
  add_test(NAME gguf_cuda COMMAND gguf_e2e_test ${ASSETS} cuda)
  add_test(NAME gguf_cuda_unfused COMMAND gguf_e2e_test ${ASSETS} cuda fusion=0)
  # Programmatic dependent launch (each kernel starts while the one before finishes) on and off, whatever the GPU's
  # table makes the default: the same tokens.
  add_test(NAME gguf_cuda_pdl COMMAND gguf_e2e_test ${ASSETS} cuda pdl=1)
  add_test(NAME gguf_cuda_no_pdl COMMAND gguf_e2e_test ${ASSETS} cuda pdl=0)
  flyx_add_golden_test(golden_cuda_qwen3_gguf ARGS cuda qwen3 ${ASSETS}/tiny_qwen3_gguf.gguf FIXTURES gguf_assets)
  # A tiny Qwen3 of GGML block weights (tiny_qwen3_gguf.gguf is float32, read as bf16), held to its own plain
  # decoding (golden_quant.json): tree rounds of more than 16 rows on packed weights, a round's GEMMs and its packed
  # head 16 rows at a time; Q8_0, and Q4_K for the K types' round kernels.
  flyx_add_golden_test(golden_cuda_quant_q8_0 ARGS cuda quant ${ASSETS}/quant_Q8_0.gguf FIXTURES gguf_assets)
  flyx_add_golden_test(golden_cuda_quant_q4_k ARGS cuda quant ${ASSETS}/quant_Q4_K.gguf FIXTURES gguf_assets)
  set_tests_properties(gguf_cuda gguf_cuda_unfused gguf_cuda_pdl gguf_cuda_no_pdl
                       PROPERTIES FIXTURES_REQUIRED gguf_assets)

  # A family of kernels a file (op_test_*.cu), helpers in op_test.cuh, main() in op_test.cu.
  add_executable(cuda_op_test backends/cuda/op_test.cu backends/cuda/op_test_gemm.cu backends/cuda/op_test_blocks.cu
                 backends/cuda/op_test_attention.cu backends/cuda/op_test_gdn.cu backends/cuda/op_test_nvfp4.cu
                 backends/cuda/op_test_head.cu backends/cuda/op_test_sampling.cu backends/cuda/op_test_layers.cu
                 backends/cuda/op_test_dv.cu backends/cuda/op_test_vision.cu backends/cuda/op_test_dspine.cu)
  target_link_libraries(cuda_op_test PRIVATE flyx_cuda_kernels)
  add_dependencies(cuda_op_test flyx_gemm_tables)  # op_test_gemm.cu reads the generated tables
  set_target_properties(cuda_op_test PROPERTIES CUDA_ARCHITECTURES "${FLYX_CUDA_ARCHS}")
  add_test(NAME cuda_ops COMMAND cuda_op_test)

  add_test(NAME scheduler_cuda COMMAND scheduler_test ${ASSETS}/tiny.flyx cuda)
  set_tests_properties(scheduler_cuda PROPERTIES FIXTURES_REQUIRED assets)
  # DV's rounds in three requests at once: every round its warm-up (rounds of several sessions all warm-ups), a cycle
  # of warm-ups, trees and plain rounds, and its rounds chosen: plain decoding's text, greedy, sampled, penalized.
  add_test(NAME scheduler_cuda_dv COMMAND scheduler_test ${ASSETS}/tiny_llama_dv31.flyx cuda dv)
  set_tests_properties(scheduler_cuda_dv PROPERTIES FIXTURES_REQUIRED assets)

  flyx_add_golden_test(golden_cuda ARGS cuda)
  flyx_add_golden_test(golden_cuda_nvfp4 ARGS cuda nvfp4)
  # the generic implementation of every region (no fused overrides) must pass as well
  flyx_add_golden_test(golden_cuda_unfused ARGS cuda nvfp4 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_pdl ARGS cuda ENV FLYX_BACKEND_OPTIONS=pdl=1)
  flyx_add_golden_test(golden_cuda_no_pdl ARGS cuda ENV FLYX_BACKEND_OPTIONS=pdl=0)
  flyx_add_golden_test(golden_cuda_qwen3 ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=embed=host)
  flyx_add_golden_test(golden_cuda_qwen3_unfused ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  # The tiny Qwen3's head is not its embedding: the table in host memory by default, here on the GPU.
  flyx_add_golden_test(golden_cuda_qwen3_embed_gpu ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=embed=gpu)
  # Llama without q/k norms: the generic regions too, and its 16 query heads on 2 KV heads on tensor cores for every
  # row.
  flyx_add_golden_test(golden_cuda_llama ARGS cuda llama)
  flyx_add_golden_test(golden_cuda_llama_unfused ARGS cuda llama ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_llama_rows_mma ARGS cuda llama ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  # The tiny Llama's NVFP4 copy, held to its own plain decoding: tree rounds past 16 rows on gemm_fp4_rows.
  flyx_add_golden_test(golden_cuda_llama_nvfp4 ARGS cuda llama_nvfp4)
  flyx_add_golden_test(golden_cuda_llama_nvfp4_unfused ARGS cuda llama_nvfp4 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  # A V-LoRA (the target's own adapters, ABI 43) on every pass: against transformers with it merged, with the generic
  # regions (gate 2) and the verify rows' attention on tensor cores; tree rounds of 31 to 63 drafts give plain
  # decoding's tokens (gate 4); beside NVFP4 linears, held to its own plain decoding.
  flyx_add_golden_test(golden_cuda_llama_lora ARGS cuda llama_lora)
  flyx_add_golden_test(golden_cuda_llama_lora_unfused ARGS cuda llama_lora ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_llama_lora_rows_mma ARGS cuda llama_lora
                       ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  flyx_add_golden_test(golden_cuda_llama_nvfp4_lora ARGS cuda llama_nvfp4_lora)
  flyx_add_golden_test(golden_cuda_llama_nvfp4_lora_unfused ARGS cuda llama_nvfp4_lora
                       ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_llama_dv ARGS cuda llama ${ASSETS}/tiny_llama_dv.flyx)
  flyx_add_golden_test(golden_cuda_llama_dv_pool ARGS cuda llama ${ASSETS}/tiny_llama_dv_pool.flyx)
  flyx_add_golden_test(golden_cuda_llama_dv32 ARGS cuda llama ${ASSETS}/tiny_llama_dv32.flyx)
  # DV beside a V-LoRA: the verify rows with the target's adapters, the canvas with the base weights and its own.
  flyx_add_golden_test(golden_cuda_llama_dv_lora ARGS cuda llama_dv_lora)
  # DV's rounds (gate 4, its drafts against the reference) with the generic regions and with the verify rows'
  # attention on tensor cores
  foreach(dv dv dv_pool dv_lora)
    set(dv_file ${ASSETS}/tiny_llama_${dv}.flyx)
    set(dv_golden llama)
    if(dv STREQUAL "dv_lora")
      set(dv_golden llama_dv_lora)
    endif()
    flyx_add_golden_test(golden_cuda_llama_${dv}_unfused ARGS cuda ${dv_golden} ${dv_file}
                         ENV FLYX_BACKEND_OPTIONS=fusion=0)
    flyx_add_golden_test(golden_cuda_llama_${dv}_rows_mma ARGS cuda ${dv_golden} ${dv_file}
                         ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  endforeach()
  # DV's rounds of 31 drafts (32 verify rows: the wide rows of a file that holds DV) on 32 slots, MiniCPM5-1B's DV
  # files' trees: a beam of 8 with coverage, a pool of 7 ranks; the beam's also with the verify rows' attention on
  # tensor cores.
  flyx_add_golden_test(golden_cuda_llama_dv31 ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx)
  flyx_add_golden_test(golden_cuda_llama_dv31_pool ARGS cuda llama ${ASSETS}/tiny_llama_dv31_pool.flyx)
  flyx_add_golden_test(golden_cuda_llama_dv31_rows_mma ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx
                       ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  # The beam's rounds of 31 drafts beside a V-LoRA of rank 32 (MiniCPM5-1B's g_b2h line): the target's adapters on
  # 32 verify rows (their GEMMs one k_gemm_skinny_mt launch each) next to the canvas's grouped GEMMs and its pass the
  # round's graph leaves out after a no-op round; also with the generic regions (the canvas's GEMMs one by one).
  flyx_add_golden_test(golden_cuda_llama_dv31_lora ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx)
  flyx_add_golden_test(golden_cuda_llama_dv31_lora_unfused ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       ENV FLYX_BACKEND_OPTIONS=fusion=0)
  # The same two trees on 16 slots from a flatter selector: trees that take the beam's 8 nodes a depth and the
  # pool's 7th rank, held whole (the canvas keeps float32's sums up to 16 rows).
  flyx_add_golden_test(golden_cuda_llama_dv31_wide ARGS cuda llama ${ASSETS}/tiny_llama_dv31_wide.flyx)
  flyx_add_golden_test(golden_cuda_llama_dv31_wide_pool ARGS cuda llama ${ASSETS}/tiny_llama_dv31_wide_pool.flyx)
  flyx_add_golden_test(golden_cuda_gdn128 ARGS cuda gdn128)
  flyx_add_golden_test(golden_cuda_gdn128_unfused ARGS cuda gdn128 ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_window ARGS cuda window)
  flyx_add_golden_test(golden_cuda_dspine ARGS cuda dspine)
  flyx_add_golden_test(golden_cuda_dspine_unfused ARGS cuda dspine ENV FLYX_BACKEND_OPTIONS=fusion=0)
  flyx_add_golden_test(golden_cuda_dspine_window ARGS cuda dspine_window)
  flyx_add_golden_test(golden_cuda_dspine1 ARGS cuda dspine1)  # one layer: its cache binary16 beside fp8 ones
  flyx_add_golden_test(golden_cuda_qwen3_dspine ARGS cuda qwen3 ${ASSETS}/tiny_qwen3_dspine.flyx)
  # decode and verification attention on tensor cores for every row (by default only past rows_attention_from keys)
  flyx_add_golden_test(golden_cuda_rows_mma ARGS cuda nvfp4 ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  flyx_add_golden_test(golden_cuda_qwen3_rows_mma ARGS cuda qwen3 ENV FLYX_BACKEND_OPTIONS=rows_attention_from=0)
  # greedy choices from the whole head, not the certified head
  flyx_add_golden_test(golden_cuda_full_head ARGS cuda nvfp4 ENV FLYX_BACKEND_OPTIONS=head_cert=0)
  # Sessions and engines give their device memory back, measured as the device's free memory: that is the whole
  # GPU's, so these run alone (RUN_SERIAL).  Every other GPU test runs beside the others and the CPU tests (ctest
  # -j, as the presets ask): their outputs do not depend on what else runs.
  flyx_add_golden_test(cuda_device_memory ARGS cuda ENV FLYX_TEST_DEVICE_MEMORY=1 SERIAL)
  flyx_add_golden_test(cuda_device_memory_nvfp4 ARGS cuda nvfp4 ENV FLYX_TEST_DEVICE_MEMORY=1 SERIAL)
  # The sessions' round graphs inside what they lease with the K/V and the low-bit copies in host
  # memory, across auto's switch: a process of its own (the driver keeps the graphs' memory of the one before).
  flyx_add_golden_test(cuda_device_memory_tiers ARGS cuda ENV FLYX_TEST_DEVICE_MEMORY=tiers SERIAL)
  # The same for DV's rounds (its trees of 15 and 31, its warm-ups, batched too) and beside a V-LoRA (the adapters'
  # GEMMs in every round's graph), the tiny Llama's DV files (rooms: its graphs are smaller than the prices are tight
  # for); and DV's round graphs (dvgraphs) with the canvas pass the body of a conditional node and inline
  # (dv_conditional=0): within what they are counted at and under twice what they took, the conditional nodes and their
  # bodies on their own too.
  flyx_add_golden_test(cuda_device_memory_dv ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx
                       ENV FLYX_TEST_DEVICE_MEMORY=rooms SERIAL)
  flyx_add_golden_test(cuda_device_memory_dv_lora ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       ENV FLYX_TEST_DEVICE_MEMORY=rooms SERIAL)
  flyx_add_golden_test(cuda_device_memory_dv_graphs ARGS cuda llama ${ASSETS}/tiny_llama_dv31.flyx
                       ENV FLYX_TEST_DEVICE_MEMORY=dvgraphs SERIAL)
  flyx_add_golden_test(cuda_device_memory_dv_lora_graphs ARGS cuda llama_dv_lora ${ASSETS}/tiny_llama_dv31_lora.flyx
                       ENV FLYX_TEST_DEVICE_MEMORY=dvgraphs SERIAL)
  # tree_bench times every width and shape from the same context, the prompt prefilled again before each timing: it
  # exits 1 when a timing's first round starts elsewhere (every one after the first did, each where the last ended).
  add_test(NAME tree_bench_context COMMAND tree_bench ${ASSETS}/tiny_llama.flyx --ctx-at 40 --widths 1,3,7 --rounds 4)
  set_tests_properties(tree_bench_context PROPERTIES FIXTURES_REQUIRED assets)
endif()

# On Windows these programs read their files by UTF-8 paths as flyx does (a build tree under a user folder named in
# Chinese): the same manifest (flyx_program).
get_property(flyx_dir_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(t IN LISTS flyx_dir_targets)
  get_target_property(flyx_target_type ${t} TYPE)
  if(flyx_target_type STREQUAL "EXECUTABLE")
    flyx_program(${t})
  endif()
endforeach()
