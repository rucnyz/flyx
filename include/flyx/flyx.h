/* Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
 *
 * Public C API.  One engine per loaded model; sessions keep conversation
 * state on the device so a follow-up request only pays for its new tokens.
 */
#ifndef FLYX_H
#define FLYX_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(FLYX_BUILDING)
#define FLYX_API __declspec(dllexport)
#else
#define FLYX_API __declspec(dllimport)
#endif
#else
#define FLYX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this API: one number, raised with every change to a struct's
 * layout or a function's signature here (a field added, moved or retyped
 * counts: the structs carry no size of their own).  A binding that mirrors
 * the structs (python/flyx/__init__.py) carries the number it was written
 * for and refuses a library that answers another, rather than reading a
 * struct laid out differently from the one it writes. */
#define FLYX_API_VERSION 3u  /* 3: added mmproj_file, no_vision, vision_device, vision_quant, vision_max_tokens to flyx_load_options */
FLYX_API uint32_t flyx_api_version(void);

typedef struct flyx_engine flyx_engine;
typedef struct flyx_session flyx_session;

typedef struct {
  char message[512];
} flyx_error;

/* flyx_load_options.kv */
#define FLYX_KV_DEFAULT 0
#define FLYX_KV_FP8 1
#define FLYX_KV_16 2

typedef struct {
  int32_t max_ctx;          /* tokens per session; 0 = 8192, or the most that fits in GPU memory (at least 4096);
                               -1 = the model's own, or the longest the GPU and host memory hold (--ctx max) */
  const char* backend;      /* "cuda", "cpu_ref" or NULL to pick the best available */
  const char* backend_dir;  /* where backend libraries live; NULL = next to the executable */
  int32_t speculative;      /* use the model's drafters (default 1) */
  int32_t verbose;
  const char* drafter;      /* "mtp" | "dflash2" | "dflash" | "dspine" | "dv" | "all" (chosen every round) |
                               "none" | NULL = the drafter files' drafters and the model file's MTP layer, else its
                               block drafter (DFlash2, DFlash or DSpine); its DV drafter only when named ("dv",
                               "all").  DV (a Llama file converted
                               with python/convert --dv; docs/design/dv.md) drafts trees, its rounds chosen every
                               round beside plain decoding's and lookup's; with a width, its rounds
                               alone.  Where the backend cannot run its rounds, the model decodes as without it
                               (plain decoding and lookup rounds) and says why; named here ("dv"), the load fails
                               instead */
  int32_t width;            /* drafts per round; 0 = choose each round (default) */
  int32_t prefill_chunk;    /* prompt rows per prefill pass; 0 = by the device memory the weights leave (1024 to 4096) */
  int32_t kv;               /* the K/V cache: FLYX_KV_DEFAULT (0: binary16 where 32 GB of GPU memory are left after
                               the weights or on Llama models, else FP8), FLYX_KV_FP8 (e4m3, a scale per position
                               and head: half the memory) or FLYX_KV_16 (binary16, llama.cpp's) */
  const char* backend_options; /* "key=value,..." for the backend (see its documentation); NULL = none */
  const char* activations;  /* NULL = as each weight's file has it; "exact" = every format's own rounding everywhere;
                               "nvfp4" = NVFP4 weights take NVFP4 inputs in prefill passes whatever the file says
                               (a backend that cannot compute them refuses the model) */
  const char* chat_template; /* a Jinja chat template's source, used instead of the model file's; NULL = the file's */
  const char* drafter_file; /* drafters in files of their own: a GGUF holding the MTP layer, or a DFlash2 drafter file
                               (python/convert --draft alone); several separated by newlines; NULL = none */
  const char* mmproj_file;  /* multimodal projector / vision encoder companion GGUF (e.g. mmproj-BF16.gguf); NULL = none */
  int32_t no_vision;        /* 1 = skip loading any vision encoder even if present in the model or companion */
  const char* vision_device; /* "disk" | "host" | "gpu" | "auto" (NULL = disk: 0 VRAM, 0 locked host RAM) */
  const char* vision_quant;  /* "q8" | "exact" | "nvfp4" (NULL = q8: Q8_0 on float vision linears) */
  int32_t vision_max_tokens; /* per-image merged visual token cap (0 = auto by context size) */
} flyx_load_options;

typedef struct {
  const char* role;         /* "system" | "user" | "assistant" */
  const char* content;
  const char* reasoning;    /* assistant thinking, may be NULL */
} flyx_message;

typedef struct {
  /* Exactly one input: token ids, chat messages (structs or OpenAI-style
   * JSON), or raw text. */
  const int32_t* tokens;
  int32_t n_tokens;
  const flyx_message* messages;
  int32_t n_messages;
  const char* text;
  int32_t enable_thinking;  /* chat only: 1 = model thinks first (template default) */
  int32_t max_new_tokens;   /* 0 = until a stop token or the context is full */
  float temperature;        /* 0 = greedy */
  int32_t ignore_eos;
  float top_p;              /* 0 or 1 = off */
  int32_t top_k;            /* 0 = off */
  uint64_t seed;            /* sampling noise; 0 = random */
  const char* const* stop;  /* stop strings: output ends before the first one */
  int32_t n_stop;
  /* OpenAI chat format as JSON text, instead of `messages`: an array of
   * {"role", "content", "reasoning_content", "tool_calls", "tool_call_id"}. */
  const char* messages_json;
  const char* tools_json;   /* chat only: JSON array of tool definitions (OpenAI format) */
  /* chat only: 1 = deliver thinking as REASONING events and tool calls as
   * TOOL_CALL events; TEXT events then carry only the answer. */
  int32_t split_output;
  /* chat only, OpenAI format as JSON text, NULL = the default ("auto", plain
   * text): tool_choice ("auto", "none", "required" or {"type": "function",
   * "function": {"name": ...}}) and response_format ({"type": "json_object"}
   * or {"type": "json_schema", "json_schema": {"schema": {...}}}). Tool calls
   * are held to their tools' schemas and a JSON answer to its schema. */
  const char* tool_choice_json;
  const char* response_format_json;
  int32_t single_tool_call;  /* chat only: 1 = at most one tool call per reply */
  /* What a token already chosen in this reply loses from its logit (the
   * OpenAI definition; 0 = off): once, and per time it was chosen. */
  float presence_penalty;
  float frequency_penalty;
} flyx_request;

typedef enum {
  FLYX_EVENT_TEXT = 1,       /* decoded text (complete UTF-8) and the tokens behind it */
  FLYX_EVENT_DONE = 2,
  FLYX_EVENT_REASONING = 3,  /* split_output: thinking text */
  FLYX_EVENT_TOOL_CALL = 4   /* split_output: {"name": ..., "arguments": {...}} as JSON text */
} flyx_event_kind;

typedef enum {
  FLYX_FINISH_STOP = 1,  /* stop token */
  FLYX_FINISH_LENGTH = 2,
  FLYX_FINISH_CANCEL = 3,
  FLYX_FINISH_ERROR = 4
} flyx_finish_reason;

typedef struct {
  int32_t prompt_tokens;    /* tokens in the request */
  int32_t reused_tokens;    /* of those, already in the session state */
  int32_t new_tokens;       /* tokens generated */
  int32_t rounds;           /* decode rounds */
  int32_t accepted_drafts;
  int32_t drafted;          /* drafts verified (sum of round widths) */
  double prefill_ms, decode_ms;
  int32_t finish_reason;    /* flyx_finish_reason */
} flyx_usage;
/* finish_reason is FLYX_FINISH_STOP for stop tokens and stop strings;
 * a reply that ends in tool calls also finishes with FLYX_FINISH_STOP. */

typedef struct {
  int32_t kind;             /* flyx_event_kind */
  const char* text;
  size_t text_len;
  const int32_t* tokens;
  int32_t n_tokens;
  const flyx_usage* usage;   /* DONE only */
} flyx_event;

/* Return non-zero to cancel the generation. */
typedef int (*flyx_event_fn)(const flyx_event* ev, void* user);

FLYX_API flyx_engine* flyx_load(const char* path, const flyx_load_options* opts, flyx_error* err);
/* Release this handle once calls using it have returned. Each existing session keeps
 * the model loaded and can continue generating until that session is freed. */
FLYX_API void flyx_unload(flyx_engine* engine);
FLYX_API const char* flyx_backend_name(const flyx_engine* engine);

/* Returns NULL if the session cannot be allocated. */
FLYX_API flyx_session* flyx_session_new(flyx_engine* engine);
/* Release once all calls using this session, including flyx_generate and flyx_cancel, have returned. */
FLYX_API void flyx_session_free(flyx_session* s);
/* Blocks until the generation ends; events arrive on the calling thread.
 * Sessions of one engine may generate on several threads at once; a session
 * serves one generation at a time. */
FLYX_API int flyx_generate(flyx_session* s, const flyx_request* req, flyx_event_fn on_event, void* user,
                     flyx_usage* usage, flyx_error* err);
/* Safe from any thread. */
FLYX_API void flyx_cancel(flyx_session* s);

/* Tokenizer access: the tokens flyx_generate feeds for `text` as a raw prompt (the tokenizer's prefix first, such
   as MiniCPM5's <s>; special tokens read everywhere). Returns the number of tokens (may exceed cap; call again). */
FLYX_API int32_t flyx_tokenize(flyx_engine* e, const char* text, int32_t* out, int32_t cap);

#ifdef __cplusplus
}
#endif
#endif /* FLYX_H */
