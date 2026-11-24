/* Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
 *
 * The boundary between the engine and a backend library.
 *
 * Every backend is a shared library loaded at start-up.  It exports one C
 * symbol, FLYX_BACKEND_ENTRY, returning a table of functions.  Only plain C
 * types cross this boundary, so a backend can be built with a different
 * compiler (nvcc, a vendor toolchain) and shipped separately.
 *
 * Division of labour:
 *   models   describe the model: weights, per-sequence state and typed blocks
 *            (flyx_model_desc), and the Program (an ordered list of nodes
 *            that run blocks) for prefill and for a round of each width.
 *   core     decides WHEN: how a prompt is cut into prefill chunks, when a
 *            round is launched, what to do with each round's tokens (stop
 *            strings, rollback, sessions).
 *   backend  decides HOW: kernels, fusion, memory layout, CUDA graphs.  It
 *            runs the block kinds it implements without knowing the model.  All
 *            state that changes from round to round (position, anchor token,
 *            round counter) lives on the device and is advanced by the round
 *            itself, so a round can be replayed without the CPU touching it.
 */
#ifndef FLYX_BACKEND_ABI_H
#define FLYX_BACKEND_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLYX_BACKEND_ABI_VERSION 45u
#define FLYX_BACKEND_ENTRY "flyx_backend_get_api"
/* How a backend library exports its entry. */
#if defined(_WIN32)
#define FLYX_BACKEND_EXPORT __declspec(dllexport)
#else
#define FLYX_BACKEND_EXPORT __attribute__((visibility("default")))
#endif
#define FLYX_MAX_ROWS 16       /* rows in one round: the anchor plus up to 15 drafts */
/* Rows of a round whose drafts are given (a lookup round, ABI 36), where the backend lowers such plans: a copy of
 * the context verified up to 63 drafts at once.  A round result holds this many tokens. */
#define FLYX_MAX_WIDE_ROWS 64
#define FLYX_MAX_STOP_IDS 8
#define FLYX_MAX_STOP_BUCKETS 8 /* sorted verify checkpoints in an adaptive round plan */
#define FLYX_CHECKPOINTS 4     /* checkpoint slots per sequence */
/* Rounds launched on a sequence and not yet read, at most: the engine reads
 * the oldest before it launches another. */
#define FLYX_MAX_IN_FLIGHT 2
/* Recurrent-state slots per sequence (> FLYX_MAX_IN_FLIGHT): round r starts
 * from slot flyx_round_slot(r) and its commit writes the next one, so two
 * rounds can stay in flight without overwriting the oldest in-flight round's
 * start state, and rewind() reaches back FLYX_ROUND_SLOTS - 1 rounds. */
#define FLYX_ROUND_SLOTS 3

/* The inline rules below are for host and device (CUDA) code alike. */
#if defined(__CUDACC__)
#define FLYX_ABI_INLINE static inline __host__ __device__
#else
#define FLYX_ABI_INLINE static inline
#endif

/* Room at the end of a sequence of max_ctx tokens: every pass leaves room for
 * one more round of FLYX_MAX_ROWS rows.  A prefill that would leave less
 * fails ("context is full"), a round that leaves less stops the sequence
 * (flyx_round_result.stopped), and the engine refuses a prompt that leaves
 * less: one rule for every backend and the core. */
FLYX_ABI_INLINE int flyx_room(int32_t position, int32_t max_ctx) { return position + FLYX_MAX_ROWS <= max_ctx; }

/* The recurrent-state slot round r starts from. */
FLYX_ABI_INLINE int flyx_round_slot(int32_t round) { return round % FLYX_ROUND_SLOTS; }
#ifdef __cplusplus
static_assert(FLYX_ROUND_SLOTS > FLYX_MAX_IN_FLIGHT, "in-flight rounds must not overwrite the oldest start slot");
#endif

typedef enum {
  FLYX_DT_BF16 = 0, FLYX_DT_F16 = 1, FLYX_DT_F32 = 2, FLYX_DT_F64 = 3,
  FLYX_DT_I32 = 4, FLYX_DT_U32 = 5, FLYX_DT_U8 = 6, FLYX_DT_I64 = 7,
  /* NVFP4 linear: logical shape [N, K], two E2M1 values per byte (low nibble
   * first), with scales (flyx_weight) F8E4M3 [N, K/16] and F32 [1]:
   * w = e2m1 * scale * scale2. */
  FLYX_DT_NVFP4 = 8,
  FLYX_DT_F8E4M3 = 9,
  /* GGML block weights, as GGUF files store them: logical shape [N, K] row
   * by row, each row K / values blocks of the type, scales inside the blocks
   * (layouts: third_party/ggml/ggml-common.h; what a stored value is:
   * ops/dtype/ref.h).  GGML's own NVFP4 is not ModelOpt's above. */
  FLYX_DT_GGML_Q4_0 = 10, FLYX_DT_GGML_Q4_1 = 11, FLYX_DT_GGML_Q5_0 = 12, FLYX_DT_GGML_Q5_1 = 13,
  FLYX_DT_GGML_Q8_0 = 14,
  FLYX_DT_GGML_Q2_K = 15, FLYX_DT_GGML_Q3_K = 16, FLYX_DT_GGML_Q4_K = 17, FLYX_DT_GGML_Q5_K = 18,
  FLYX_DT_GGML_Q6_K = 19,
  FLYX_DT_GGML_IQ2_XXS = 20, FLYX_DT_GGML_IQ2_XS = 21, FLYX_DT_GGML_IQ2_S = 22,
  FLYX_DT_GGML_IQ3_XXS = 23, FLYX_DT_GGML_IQ3_S = 24,
  FLYX_DT_GGML_IQ1_S = 25, FLYX_DT_GGML_IQ1_M = 26,
  FLYX_DT_GGML_IQ4_NL = 27, FLYX_DT_GGML_IQ4_XS = 28,
  /* GGML's NVFP4 (ABI 22): blocks of 64 e2m1 values with an unsigned e4m3
   * scale per 16 and no scale per tensor; not ModelOpt's FLYX_DT_NVFP4
   * above, whose values a backend may take it as. */
  FLYX_DT_GGML_NVFP4 = 29
} flyx_dtype;

/* A GGML block type's weights per block and bytes per block; 0 for any other dtype. */
static inline int32_t flyx_block_values(int32_t dtype) {
  if (dtype >= FLYX_DT_GGML_Q4_0 && dtype <= FLYX_DT_GGML_Q8_0) return 32;
  if (dtype == FLYX_DT_GGML_IQ4_NL) return 32;
  if (dtype == FLYX_DT_GGML_NVFP4) return 64;
  return dtype >= FLYX_DT_GGML_Q2_K && dtype <= FLYX_DT_GGML_IQ4_XS ? 256 : 0;
}
static inline int32_t flyx_block_bytes(int32_t dtype) {
  switch (dtype) {
    case FLYX_DT_GGML_Q4_0: return 18;
    case FLYX_DT_GGML_Q4_1: return 20;
    case FLYX_DT_GGML_Q5_0: return 22;
    case FLYX_DT_GGML_Q5_1: return 24;
    case FLYX_DT_GGML_Q8_0: return 34;
    case FLYX_DT_GGML_Q2_K: return 84;
    case FLYX_DT_GGML_Q3_K: return 110;
    case FLYX_DT_GGML_Q4_K: return 144;
    case FLYX_DT_GGML_Q5_K: return 176;
    case FLYX_DT_GGML_Q6_K: return 210;
    case FLYX_DT_GGML_IQ2_XXS: return 66;
    case FLYX_DT_GGML_IQ2_XS: return 74;
    case FLYX_DT_GGML_IQ2_S: return 82;
    case FLYX_DT_GGML_IQ3_XXS: return 98;
    case FLYX_DT_GGML_IQ3_S: return 110;
    case FLYX_DT_GGML_IQ1_S: return 50;
    case FLYX_DT_GGML_IQ1_M: return 56;
    case FLYX_DT_GGML_IQ4_NL: return 18;
    case FLYX_DT_GGML_IQ4_XS: return 136;
    case FLYX_DT_GGML_NVFP4: return 36;
    default: return 0;
  }
}

/* A tensor of the model file (host memory, read-only). */
typedef struct {
  const char* name;            /* the file's name for it, e.g. "target.L3.attn.q" */
  int32_t dtype;               /* flyx_dtype */
  int32_t ndim;
  int64_t shape[4];
  const void* data;
  uint64_t nbytes;
} flyx_tensor_view;

/* ---- The model: weights, state, blocks ----------------------
 *
 * models/<arch> builds these from the model file.  A backend runs the block
 * kinds it implements and allocates the state the table declares; it does
 * not know which model it runs, and fails open() on a block it cannot run.
 * Indices are into the tables of flyx_model_desc; -1 means none.  A
 * backend may lay weights out as it likes (stack q|k|v, quantize what only
 * drafting reads) as long as the math below holds. */

/* How a linear's inputs are rounded before they meet its weight: part of the
 * weight's numerics contract (docs/design/numerics.md and
 * 0032).  models/ fills it from what the file says, so it is the same on
 * every backend and GPU; ops/ holds each rounding in float64. */
typedef enum {
  FLYX_ACT_FORMAT = 0,  /* the format's own: bf16 (bf16, F16, F32 and NVFP4 weights); binary16 over a row's
                              power-of-two scale (GGML blocks) */
  FLYX_ACT_Q8 = 1,      /* GGML blocks, every pass: 8-bit integers with a scale per 32 values (ops/dtype/ref.h:
                              round_q8).  ABI 18: before, rounds (up to FLYX_MAX_ROWS rows) kept FORMAT */
  FLYX_ACT_NVFP4 = 2,   /* NVFP4 weights, passes of more rows: NVFP4 as well (e2m1 values, an e4m3 scale per 16, a
                              scale per row; ops/dtype/ref.h: round_nvfp4).  ABI 17: a backend built before would
                              not know it and compute bf16 */
} flyx_activations;

/* Host memory in pieces of one size (ABI 22): bytes go piece after piece, so
 * memory of any snapshot's size is pieces any other snapshot gave back.  A
 * null piece is skipped (ABI 31): nothing is written there, or
 * read from there (the state its bytes would set is left as it is), so a
 * snapshot larger than the memory at hand goes a few pieces at a time; the
 * first piece (the header) is always given. */
typedef struct {
  void* const* pieces;
  int32_t n;
  int32_t pad0;
  uint64_t piece_bytes;  /* every piece; at least 4096 */
} flyx_host_pieces;

/* A weight: a tensor of the model file; an NVFP4 tensor comes with its scales. */
typedef struct {
  const flyx_tensor_view* t;
  const flyx_tensor_view* scale;    /* NVFP4: F8E4M3 [N, K/16]; else NULL */
  const flyx_tensor_view* scale2;   /* NVFP4: F32 [1]; else NULL */
  int32_t activations;                 /* flyx_activations */
  int32_t pad0;
} flyx_weight;

/* RMSNorm: y = x / sqrt(mean(x^2) + eps) * (offset + w).  w -1: no norm, y = x, where a block says it may be
 * none (an attention block's q_norm and k_norm). */
typedef struct {
  int32_t w;                           /* weight [n] */
  int32_t pad0;
  double offset, eps;
} flyx_norm;

/* Generalized norm for vision blocks (ABI 44): RMSNorm or LayerNorm with optional weight and bias. */
typedef enum {
  FLYX_NORM_RMS = 0,
  FLYX_NORM_LAYER = 1
} flyx_norm_kind;

typedef struct {
  int32_t kind;                        /* flyx_norm_kind */
  int32_t w;                           /* weight [n], -1 = none */
  int32_t b;                           /* bias [n], -1 = none */
  int32_t pad0;
  double offset, eps;
} flyx_vnorm;

/* A low-rank adapter on a linear y = W x (ABI 42): y = W x + scale b (a x), a [rank, in], b [out, rank], both bf16
 * (ABI 43: the scale; DV's canvas adapters have none, scale 1).  rank 0: none, as a block built zeroed has. */
typedef struct {
  int32_t a, b, rank;
  float scale;
} flyx_lora;

/* A layer's linears, in the order an array of adapters lists them (ABI 42). */
typedef enum {
  FLYX_LIN_Q = 0, FLYX_LIN_K = 1, FLYX_LIN_V = 2, FLYX_LIN_O = 3,
  FLYX_LIN_GATE = 4, FLYX_LIN_UP = 5, FLYX_LIN_DOWN = 6,
  FLYX_LINS = 7
} flyx_linear;

typedef enum {
  FLYX_STATE_KV = 1,     /* `layers` K/V caches of n_kv x head_dim per position */
  FLYX_STATE_GDN = 2,    /* gated delta net: state [nv][dv][dk], conv window [conv - 1][2 nk dk + nv dv] */
  FLYX_STATE_RING = 3,   /* `width` floats per position, kept for the positions a rollback can reach */
  /* DV's canvas (ABI 42, flyx_dv_block): per slot of `slots`, its next input token, its expected embedding (`width`
   * floats), its top `topk` candidates with their log-probabilities and its selector projection (`rank` floats); and
   * the position the canvas follows.  Not positional and not kept in checkpoints: a prefill, rewind or restore leaves
   * it unknown, and the drafter starts it over (only drafting reads it). */
  FLYX_STATE_CANVAS = 4
} flyx_state_kind;

/* Per-sequence state.  Positional state (KV, RING) is rewritten after a
 * rollback; recurrent state (GDN) is folded to the accepted rows at commit,
 * rewound inside the last rounds and kept in checkpoints. */
typedef struct {
  int32_t kind;                        /* flyx_state_kind */
  int32_t draft;                       /* only drafting reads it, so it never changes the output */
  int32_t layers, n_kv, head_dim;      /* KV */
  int32_t nk, nv, dk, dv, conv;        /* GDN */
  int32_t width;                       /* RING; CANVAS: the hidden size */
  int32_t head_order;                  /* GDN: flyx_gdn_head_order */
  int32_t slots, topk, rank;           /* CANVAS (ABI 42): slots, candidates a slot keeps, the selector's rank */
  int32_t pad0;
} flyx_state;

/* Layouts a model file may store the same weights in.  A description names
 * the one its file has; nobody reorders a tensor on the host, a backend
 * reads (or uploads) it as it is. */

/* The rows of a gated attention's q weight (2 * n_heads * head_dim of them). */
typedef enum {
  FLYX_GATE_NONE = 0,               /* no output gate: n_heads * head_dim rows */
  FLYX_GATE_AFTER_Q = 1,            /* every head's q, then every head's gate */
  FLYX_GATE_PER_HEAD = 2            /* a head's q, then its gate, head after head (Qwen3.5 checkpoints, GGUF) */
} flyx_gate_layout;

/* The key head value head v of a gated delta net reads, r = nv / nk value
 * heads to a key head.  Every per-value-head tensor follows the same order. */
typedef enum {
  FLYX_GDN_HEADS_GROUPED = 0,       /* v / r: a key head's value heads side by side */
  FLYX_GDN_HEADS_TILED = 1          /* v % nk: the key heads in turn, r times over (GGUF) */
} flyx_gdn_head_order;

/* What a gated delta net's decay weight holds. */
typedef enum {
  FLYX_GDN_DECAY_LOG = 0,           /* A_log */
  FLYX_GDN_DECAY_NEG_EXP = 1        /* -exp(A_log) (GGUF's ssm_a) */
} flyx_gdn_decay;

typedef enum {
  FLYX_BLOCK_EMBED = 1,
  FLYX_BLOCK_ATTN = 2,
  FLYX_BLOCK_GDN = 3,
  FLYX_BLOCK_MLP = 4,
  FLYX_BLOCK_HEAD = 5,
  FLYX_BLOCK_MTP = 6,
  FLYX_BLOCK_DFLASH2 = 7,
  FLYX_BLOCK_DFLASH2_LAYER = 8,
  FLYX_BLOCK_DV = 9,           /* ABI 42 */
  FLYX_BLOCK_DV_LAYER = 10,    /* ABI 42 */
  FLYX_BLOCK_VISION_STEM = 11, /* ABI 44: patch embed + optional bias + optional pos embed / cls */
  FLYX_BLOCK_VIT_ATTN = 12,    /* ABI 44: bidirectional ViT self-attention */
  FLYX_BLOCK_VIT_MLP = 13,     /* ABI 44: ViT FFN (2-layer standard or 3-layer gated) */
  FLYX_BLOCK_VISION_MERGE = 14, /* ABI 44: spatial merge/pool + optional norm + 1/2-layer projector */
  FLYX_BLOCK_DSPINE = 15,      /* ABI 45 */
  FLYX_BLOCK_DSPINE_LAYER = 16 /* ABI 45 */
} flyx_block_kind;

/* Token rows -> residual rows: x = table[token]. */
typedef struct {
  int32_t hidden, vocab;
  int32_t table;                       /* [vocab, hidden] */
} flyx_embed_block;

/* x += o(attention(rope(q_norm(q)), rope(k_norm(k)), v) [* sigmoid(gate)]) with
 * q, k, v projections of norm(x); per head, causal over the sequence, keys
 * and values from its cache.  gate: the q weight also holds the gate's rows
 * (flyx_gate_layout).  Rotary (theta) on the first rot_dim dims of each
 * head, halves rotated.  q_norm and k_norm may be none (w -1: Llama's heads
 * go to the rotary as projected).  A backend built before
 * fails on such a block ("weight -1 missing": CUDA's at open, cpu_ref's in
 * the first pass) rather than computing anything else, so the ABI kept its
 * number. */
typedef struct {
  int32_t hidden, n_heads, n_kv, head_dim, rot_dim;
  int32_t gate;                        /* flyx_gate_layout */
  double theta;
  flyx_norm norm, q_norm, k_norm;
  int32_t q, k, v, o;
  int32_t kv;                          /* state: its K/V cache (layers 1) */
  /* ABI 43: the block's own adapters on q, k, v and o (flyx_linear order), in every pass that runs it (a V-LoRA,
   * python/convert --lora), all four or none, one rank; none with an output gate. */
  flyx_lora lora[4];
  /* ABI 44: 3D Multimodal RoPE (Qwen3.5/3.8). mrope_section[0..2] = (T, H, W) frequency pair counts summing to
   * rot_dim / 2 (all 0 = standard 1D RoPE). mrope_interleaved: 1 = interleaved T,H,W; 0 = contiguous chunks. */
  int32_t mrope_section[3];
  int32_t mrope_interleaved;
} flyx_attn_block;

/* Gated delta net, as ops/ops.h: from h = norm(x), mixed = qkv(h) through
 * the causal conv (conv_w [2 nk dk + nv dv, conv]) and SiLU, q and k
 * l2-normalised, beta = sigmoid(b(h)), g = -exp(A_log) * softplus(a(h) +
 * dt_bias), the delta rule per value head over its state (with the q and k
 * of the key head head_order gives it), then x += out(out_norm(o) *
 * silu(z(h))) with out_norm per value head.  The weight A_log holds what
 * decay says. */
typedef struct {
  int32_t hidden, nk, nv, dk, dv, conv;
  flyx_norm norm, out_norm;
  int32_t qkv, z, b, a, out, conv_w, A_log, dt_bias;
  int32_t state;                       /* its GDN state */
  int32_t head_order;                  /* flyx_gdn_head_order, the state's too */
  int32_t decay;                       /* flyx_gdn_decay */
} flyx_gdn_block;

/* x += down(silu(gate(h)) * up(h)), h = norm(x). */
typedef struct {
  int32_t hidden, inter;
  flyx_norm norm;
  int32_t gate, up, down;
  flyx_lora lora[3];                   /* ABI 43: on gate, up and down, as an attention block's (all or none) */
} flyx_mlp_block;

/* logits = w(norm(x)), then the token choice per row (ops/sampling/sampling.h). */
typedef struct {
  int32_t hidden, vocab;
  flyx_norm norm;
  int32_t w;                           /* [vocab, hidden]; the embedding's table when tied */
} flyx_head_block;

/* Multi-token prediction (python/reference/flyx_ref/mtp.py): from the
 * target's final-normed hidden at a position and the next token,
 * x = fc([norm_emb(embed table[token]) | norm_hidden(hidden)]), the attention
 * and MLP blocks, norm, the head's logits: the token after.  Its ring keeps the
 * target's final-normed hidden per position. */
typedef struct {
  int32_t embed, head, attn, mlp;      /* blocks */
  flyx_norm norm_emb, norm_hidden, norm;
  int32_t fc;                          /* [hidden, 2 * hidden] */
  int32_t ring;                        /* state (width hidden) */
} flyx_mtp_block;

/* DFlash2 block drafter (python/reference/flyx_ref/draft.py): `block`
 * rows (the anchor, then mask_token) through its layers, attending to the
 * context K/V and to each other (not causal; with a window, only to the
 * positions less than `window` from their own), then per row norm, the head's
 * top sel_topk, and the selector walk (sel_hid, sel_pred, sel_succ).  Its
 * context: fc(the residual rows tapped at n_taps points, side by side) ->
 * hidden_norm -> each layer's k (k_norm, rope) and v into its cache.  The
 * embedding and head blocks may hold packed GGML blocks. */
typedef struct {
  int32_t hidden, n_heads, n_kv, head_dim, inter;
  int32_t block, mask_token, n_taps, conv_taps, conv_group, sel_rank, sel_topk;
  double theta;
  int32_t embed, head;                 /* blocks */
  int32_t first_layer, layers;         /* blocks of kind DFLASH2_LAYER */
  flyx_norm hidden_norm, norm;
  int32_t fc, sel_pred, sel_succ, sel_hid;
  int32_t kv;                          /* state: one cache per layer */
  int32_t window;                      /* sliding window of every layer; 0: none */
} flyx_dflash2_block;

/* A DFlash2 layer: attention (q, k, v, o; q/k norms; rotary on all dims) and
 * MLP, each wrapped in a grouped dynamic conv whose coefficients are
 * base [2][conv_taps][hidden] plus proj(h). */
typedef struct {
  flyx_norm norm, post_norm, q_norm, k_norm;
  int32_t q, k, v, o, gate, up, down;
  int32_t attn_conv_proj, attn_conv_base, mlp_conv_proj, mlp_conv_base;
} flyx_dflash2_layer_block;

/* DSpine block drafter (ABI 45; python/reference/flyx_ref/dspine.py): `block` rows (the anchor, then mask_token)
 * through its layers (DSPINE_LAYER blocks), each a Qwen3 layer whose attention sees the context K/V and the block's
 * rows up to its own (causal; with a window, only the positions less than `window` before its own), then the code
 * bridge's write.  With x_r row r's residual after the layer, rms_r = sqrt(mean(x_r^2)):
 *   c_r = sqrt(code) u_r / (|u_r| + bridge_eps), u_r = read (x_r / (rms_r + bridge_eps))      (r >= 1)
 *   c_0 = sqrt(code) codebook[anchor]
 *   x_r += rms_r write m_r,  m_r = sigmoid(gate c_r + pred c_{r-1} + gate_bias) * (prev c_{r-1})   (r >= 1)
 * (read and write the layer's).  Then per row r >= 1: norm, the head's `topk` best tokens (high to low, the lower
 * id on ties), and the walk: row r's draft is the candidate t maximising <x'_r, norm.w * head[t]> (the first on
 * ties), x'_r the row's last write redone with e = sqrt(code) codebook[the draft before it] (the anchor for row 1)
 * in place of c_{r-1}, from the x_r and c_r before that write.  No row reads a row after it: a block of fewer rows
 * drafts the same first drafts.  Its context is DFlash2's: fc(the residual rows tapped at n_taps points) ->
 * hidden_norm -> each layer's k (k_norm, rope) and v into its cache.  The embedding and head blocks may hold packed
 * GGML blocks.  Only drafting reads it: it never changes the output. */
typedef struct {
  int32_t hidden, n_heads, n_kv, head_dim, inter;
  int32_t block, mask_token, n_taps;
  int32_t code, rank, topk;            /* the bridge's code size and message rank; the walk's candidates (<= 16) */
  int32_t window;                      /* sliding window of every layer; 0: none */
  double theta, bridge_eps;
  int32_t embed, head;                 /* blocks */
  int32_t first_layer, layers;         /* blocks of kind DSPINE_LAYER */
  flyx_norm hidden_norm, norm;
  int32_t fc;
  int32_t codebook;                    /* [vocab, code] */
  int32_t gate, pred, prev;            /* [rank, code] */
  int32_t gate_bias;                   /* [rank] */
  int32_t kv;                          /* state: one cache per layer */
  int32_t pad0;
} flyx_dspine_block;

/* A DSpine layer: attention (q, k, v, o; q/k norms; rotary on all dims), MLP, and the bridge's read [code, hidden]
 * and write [hidden, rank] after them. */
typedef struct {
  flyx_norm norm, post_norm, q_norm, k_norm;
  int32_t q, k, v, o, gate, up, down;
  int32_t read, write;
  int32_t pad0;
} flyx_dspine_layer_block;

/* The tree DV drafts (flyx_dv_block.tree; ops/draft/ref.h). */
typedef enum {
  FLYX_DV_TREE_BEAM = 0,  /* beam_tree, bifill-dv's DYN=T:W[:cov]: the `beam` best nodes a depth, then the T best */
  FLYX_DV_TREE_POOL = 1   /* pool_tree, ST T/R/1: rank tuples leaving the selector's best path once at rank <= R */
} flyx_dv_tree;

/* A layer of DV's canvas (ABI 42): the target's attention block `attn` and MLP block `mlp` run the canvas rows with
 * their base weights (never an adapter of the target's own), these norms in place of the blocks' norms, and an
 * adapter on each of the layer's linears (lora[flyx_linear]). */
typedef struct {
  int32_t attn, mlp;                   /* blocks of the target (FLYX_BLOCK_ATTN, FLYX_BLOCK_MLP) */
  flyx_norm norm, post_norm;           /* before the attention and before the MLP */
  flyx_lora lora[FLYX_LINS];
} flyx_dv_layer_block;

/* DV, bifill-dv's drafter (ABI 42; docs/design/dv.md, python/reference/flyx_ref/dv.py): `slots` canvas rows after a
 * round's anchor run the target's layers (DV_LAYER blocks, one per target layer) in a pass of their own, and a
 * selector grows the next round's tree from what they predict.  Its rounds are trees of drafts, chosen every round
 * beside rounds of other kinds (the engine's plain decoding and lookup rounds; pinned: alone).
 *   CONTEXT in a prefill program: the canvas starts over (every slot MASK, z = 0).
 *   CONTEXT in a round, after ACCEPT: the canvas pass.  With s the round's first position (its anchor's), slot w is
 *     a row at position s + 1 + w that sees the cache up to s, the anchor included, and every slot (causal: the
 *     slots up to itself), never the tree.  Its input is
 *       x_w = (cv_w == mask_token ? mask_emb : embedding[cv_w]) + res_out (silu(g) * u),  [u | g] = res_in z_w,
 *     cv_w and z_w the slot's input token and expected embedding (0 for a new slot); after the layers, norm and the
 *     head's weight give its logits: its `topk` best tokens (ties to the lower id), their probabilities p under the
 *     whole vocabulary, z' = sum over the first res_k of p_k embedding[i_k], and h, its normed hidden state.  The
 *     canvas then slides by the n tokens the round committed: slot w takes slot w + n's (none past the canvas: MASK,
 *     z = 0, no candidates), its input the top token where that has probability >= fill, else MASK.
 *   DRAFT (arg1 = FLYX_DRAFT_TREE): the round's tree of arg0 drafts from the slid canvas.  Depth d + 1 is slot d's
 *     candidates, scored by the selector after each candidate of slot d - 1 (the anchor for slot 0):
 *       S(prev, b) = (sel_alpha log max(p_b, 1e-9) + <A embedding[prev] * H h, B embedding[cand_b]> / sqrt(sel_rank))
 *                    / sel_temp,
 *     log-softmax over the slot's candidates, and the tree `tree` names (ops/draft/ref.h).  A canvas that is not the
 *     one the round before left (after a prefill, a rewind, a restore, a round of another kind) drafts arg0 MASKs
 *     under the anchor instead and starts over.
 * Only drafting reads the canvas and these weights: they never change the output. */
typedef struct {
  int32_t hidden, vocab;
  int32_t slots;                       /* N canvas rows */
  int32_t tree;                        /* flyx_dv_tree */
  int32_t tree_width;                  /* T: the drafts a tree holds at most (a round's DRAFT asks for arg0 <= T) */
  int32_t beam;                        /* BEAM: nodes kept a depth */
  int32_t cov;                         /* BEAM: an edge into slot d adds log max(sum_k exp(lp_k), 1e-6), lp_k =
                                          log max(p_k, 1e-9) over its topk candidates (ops::slot_coverage) */
  int32_t pool_ranks;                  /* POOL: the ranks a deviation from the selector's best path reaches */
  int32_t topk;                        /* candidates a slot keeps: 16 (bifill-dv's; the only count models/ give) */
  int32_t res_k;                       /* candidates z' sums: 16, the first res_k of the topk */
  int32_t mask_token;
  int32_t causal;                      /* a slot sees only the slots up to itself */
  float fill;                          /* a slot's top token is its next input from this probability on */
  float sel_temp;                      /* the selector's scores are divided by it (1: bifill-dv's) */
  int32_t embed, head;                 /* the target's blocks: the canvas reads the table and the head's weight */
  int32_t first_layer, layers;         /* DV_LAYER blocks, the target's layers in order */
  flyx_norm norm;                      /* the canvas's final norm, in place of the head's */
  int32_t mask_emb;                    /* [hidden] */
  int32_t res_in;                      /* [2 hidden, hidden]: u's rows, then g's */
  int32_t res_out;                     /* [hidden, hidden] */
  int32_t sel_rank;
  int32_t sel_a, sel_h, sel_b;         /* [sel_rank, hidden], float32 */
  int32_t sel_alpha;                   /* [1], float32 */
  int32_t canvas;                      /* state: FLYX_STATE_CANVAS */
} flyx_dv_block;

/* ---- Composable Vision Encoder Blocks (ABI 44) ----------------------------------
 * Parameterized building blocks shared across vision-language architectures
 * (Qwen3.5/3.8, SigLIP/Gemma3, Pixtral, InternVL, Llama4): adding a new VLM
 * describes its vision encoder using these blocks and a FLYX_PHASE_VISION
 * program under models/<arch>/ without changing any backend. */

typedef enum {
  FLYX_VACT_NONE = 0,
  FLYX_VACT_GELU_TANH = 1,   /* 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
  FLYX_VACT_GELU = 2,        /* 0.5 * x * (1 + erf(x / sqrt(2))) */
  FLYX_VACT_SILU = 3,
  FLYX_VACT_QUICK_GELU = 4   /* x * sigmoid(1.702 * x) */
} flyx_vact_kind;

typedef enum {
  FLYX_VPOS_NONE = 0,
  FLYX_VPOS_ADD_FIXED = 1,   /* pos_w [N, hidden] added directly row-by-row */
  FLYX_VPOS_BILINEAR_2D = 2  /* pos_w [pos_grid_h * pos_grid_w, hidden] bilinearly interpolated (align_corners=1)
                              * and ordered into spatial-merge blocks of merge_size x merge_size */
} flyx_vpos_kind;

typedef enum {
  FLYX_VROPE_NONE = 0,
  FLYX_VROPE_AXIAL_2D = 1    /* 2D axial RoPE over patch (h, w) in spatial-merge order (first half H, second half W) */
} flyx_vrope_kind;

typedef enum {
  FLYX_VMERGE_NONE = 0,
  FLYX_VMERGE_SPATIAL_CONCAT = 1, /* merge_size^2 consecutive rows concatenated -> [N / m^2, hidden * m^2] */
  FLYX_VMERGE_AVG_POOL_2D = 2     /* 2D average pool merge_size x merge_size per grid -> [N / m^2, hidden] */
} flyx_vmerge_kind;

/* Patch embedding linear (single weight or split temporal pair W0|W1 as in GGUF mmproj),
 * optional bias, optional positional embedding addition, optional pre-norm. */
typedef struct {
  int32_t in_dim;            /* patch vector width (e.g. 1536 = 2 * 3 * 16 * 16) */
  int32_t hidden;            /* ViT hidden width (e.g. 1152 on Qwen3.8-27B) */
  int32_t patch_w;           /* [hidden, in_dim] (or first half [hidden, in_dim/2] when patch_w2 >= 0) */
  int32_t patch_w2;          /* -1 = single weight; >= 0 = second temporal slice [hidden, in_dim/2] */
  int32_t patch_b;           /* [hidden], -1 = none */
  int32_t pos_mode;          /* flyx_vpos_kind */
  int32_t pos_w;             /* [pos_grid_h * pos_grid_w, hidden], -1 = none */
  int32_t pos_grid_h;        /* base grid height (e.g. 48) */
  int32_t pos_grid_w;        /* base grid width (e.g. 48) */
  int32_t merge_size;        /* spatial merge factor (e.g. 2) for BILINEAR_2D ordering */
  int32_t cls_token;         /* optional [hidden] prepended per image (-1 = none) */
  int32_t pad0;
  flyx_vnorm pre_norm;       /* optional norm after pos_embed (w = -1: none) */
} flyx_vision_stem_block;

/* Bidirectional block-diagonal ViT self-attention: x += [layer_scale *] o(attn(rope(q), rope(k), v)) + o_b.
 * Supports fused qkv or split q/k/v, optional biases, optional QK-norm, and arbitrary head_dim % 8 == 0. */
typedef struct {
  int32_t hidden, n_heads, head_dim;
  int32_t rope_mode;         /* flyx_vrope_kind */
  int32_t merge_size;        /* spatial merge factor for AXIAL_2D coordinates */
  int32_t pad0;
  double theta;              /* 2D RoPE base (e.g. 10000.0) */
  flyx_vnorm norm;           /* pre-attention norm */
  flyx_vnorm q_norm, k_norm; /* optional per-head Q/K norm (w = -1: none) */
  int32_t qkv, qkv_b;        /* fused [3*nh*hd, hidden] & [3*nh*hd] (-1 if split q/k/v) */
  int32_t q, k, v;           /* split [nh*hd, hidden] (-1 if fused qkv) */
  int32_t q_b, k_b, v_b;     /* split biases (-1 = none) */
  int32_t o, o_b;            /* [hidden, nh*hd] & [hidden] (-1 = none) */
  int32_t layer_scale;       /* optional [hidden] residual scale (-1 = none) */
  int32_t pad1;
} flyx_vit_attn_block;

/* ViT FFN: x += [layer_scale *] fc2(act(fc1(h)) [* up]) where h = norm(x). */
typedef struct {
  int32_t hidden, inter;
  int32_t gated;             /* 0: fc2(act(fc1(h))); 1: fc2(act(gate(h)) * fc1(h)) */
  int32_t act;               /* flyx_vact_kind */
  flyx_vnorm norm;           /* pre-MLP norm */
  int32_t fc1, fc1_b;        /* [inter, hidden] & [inter] (-1 = none) */
  int32_t gate, gate_b;      /* gated=1: [inter, hidden] & [inter] (-1 if !gated) */
  int32_t fc2, fc2_b;        /* [hidden, inter] & [hidden] (-1 = none) */
  int32_t layer_scale;       /* optional [hidden] residual scale (-1 = none) */
  int32_t pad0;
} flyx_vit_mlp_block;

/* Spatial merge / pooling + optional norm (before or after merge) + 1- or 2-layer projector to out_hidden. */
typedef struct {
  int32_t in_hidden;         /* ViT hidden width (e.g. 1152) */
  int32_t merge_size;        /* spatial merge factor (e.g. 2) */
  int32_t merge_mode;        /* flyx_vmerge_kind */
  int32_t post_norm;         /* 0: norm before merge (width in_hidden); 1: norm after merge (width merged_dim) */
  int32_t fc1_out;           /* intermediate dim of 2-layer projector (e.g. 4608), or 0 for 1-layer (fc2 only) */
  int32_t out_hidden;        /* LLM hidden width (e.g. 5120) */
  int32_t act;               /* flyx_vact_kind between fc1 and fc2 */
  int32_t pad0;
  flyx_vnorm norm;           /* post_ln (w = -1: none) */
  int32_t fc1, fc1_b;        /* [fc1_out, merged_dim] & [fc1_out] (-1 if fc1_out == 0) */
  int32_t fc2, fc2_b;        /* [out_hidden, fc1_out or merged_dim] & [out_hidden] (-1 = none) */
} flyx_vision_merge_block;

typedef struct {
  int32_t kind;                        /* flyx_block_kind */
  int32_t pad0;
  union {
    flyx_embed_block embed;
    flyx_attn_block attn;
    flyx_gdn_block gdn;
    flyx_mlp_block mlp;
    flyx_head_block head;
    flyx_mtp_block mtp;
    flyx_dflash2_block dflash2;
    flyx_dflash2_layer_block dflash2_layer;
    flyx_dv_block dv;                  /* ABI 42 */
    flyx_dv_layer_block dv_layer;      /* ABI 42 */
    flyx_vision_stem_block vision_stem;   /* ABI 44 */
    flyx_vit_attn_block vit_attn;         /* ABI 44 */
    flyx_vit_mlp_block vit_mlp;           /* ABI 44 */
    flyx_vision_merge_block vision_merge; /* ABI 44 */
    flyx_dspine_block dspine;          /* ABI 45 */
    flyx_dspine_layer_block dspine_layer;  /* ABI 45 */
  };
} flyx_block;

typedef struct {
  int32_t hidden, vocab;               /* the residual stream's width, the vocabulary */
  int32_t n_weights, n_states, n_blocks;
  const flyx_weight* weights;
  const flyx_state* states;
  const flyx_block* blocks;
} flyx_model_desc;

/* ---- Program nodes ---------------------------------------------------------- */

typedef enum {
  FLYX_NODE_RUN = 1,     /* run `block` on the pass's rows: embedding, mixer, MLP; head (arg0 = 1: last row only) */
  FLYX_NODE_TAP,         /* residual rows -> feature slot arg0 of drafter `block` */
  FLYX_NODE_DRAFT,       /* drafter `block` proposes arg0 tokens after the anchor; arg1 = FLYX_DRAFT_TREE: a
                             * tree of them, which the round verifies as a tree (ABI 24) */
  FLYX_NODE_CONTEXT,     /* drafter `block` takes the pass's rows (a round's: the accepted ones) as context; DV's
                             * (ABI 42): its canvas pass in a round, starting its canvas over in a prefill */
  FLYX_NODE_ACCEPT,      /* compare drafts with the target's choices, advance, emit */
  FLYX_NODE_COMMIT,      /* fold recurrent state to the accepted rows */
  FLYX_NODE_GIVEN        /* the launch's drafts (gen.drafts) are the round's arg0 drafts; no drafter runs (block -1);
                             * arg1 = FLYX_GIVEN_TREE: they form a tree (gen.parents) */
} flyx_node_kind;

/* FLYX_NODE_GIVEN's arg1: the round verifies a tree of drafts (docs/design/tree-verification.md). */
#define FLYX_GIVEN_TREE 1
/* FLYX_NODE_DRAFT's arg1: the drafter proposes its arg0 drafts as a tree, best first by its own scores (each node
 * the most likely continuation of one already in the tree), and the round verifies the tree.  A backend refuses a
 * drafter that cannot (ABI 24).  DV drafts nothing but trees, its own (flyx_dv_block, ABI 42). */
#define FLYX_DRAFT_TREE 1

typedef struct { int32_t kind, block, arg0, arg1; } flyx_node;

/* ---- Program: what a backend is asked to run ---------------------------- */

typedef enum { FLYX_PHASE_PREFILL = 0, FLYX_PHASE_ROUND = 1, FLYX_PHASE_VISION = 2 } flyx_phase;

typedef struct {
  int32_t phase;                /* flyx_phase */
  int32_t width;                /* round: drafts verified per round (0 = plain decode) */
  int32_t max_rows;             /* prefill: largest chunk; round: width + 1 */
  int32_t n_nodes;
  const flyx_node* nodes;
  /* Optional intra-round stopping checkpoints (ABI 41; 0 = fixed width): n_checkpoints sorted verify widths
   * w_0 < ... < w_{n-1} == width.  Backends that do not lower adaptive plans may refuse n_checkpoints > 0. */
  int32_t n_checkpoints;
  const int32_t* checkpoints;
} flyx_program;

/* Drafter-agnostic online confidence stopping parameters (ABI 41): computed by core/speculation/ from observed
 * per-draft (prob, margin) and measured bucket costs, evaluated on device inside an adaptive round plan. */
typedef struct {
  int32_t enabled;                          /* 0 = run full plan width; 1 = dynamic intra-round stopping */
  int32_t n_buckets;                        /* matches the adaptive plan's n_checkpoints (<= FLYX_MAX_STOP_BUCKETS) */
  int32_t widths[FLYX_MAX_STOP_BUCKETS];    /* sorted verify widths w_0 < ... < w_{n-1} */
  float costs[FLYX_MAX_STOP_BUCKETS];       /* measured solo round cost (ms) for each bucket width */
  float p_pos[FLYX_MAX_ROWS];               /* online per-position conditional accept rate P(acc_k | acc_<k) */
  float w_bias;                             /* online logistic bias */
  float w_margin;                           /* online weight on clamped top-1/top-2 margin */
  float w_logit_p;                          /* online weight on clamped logit(p_max) */
  float lambda;                             /* online throughput estimate (tokens / ms) */
} flyx_stop_policy;

/* ---- Generation parameters and results ------------------------------------- */

/* Token choice and stop conditions, see ops/sampling/sampling.h for the exact meaning. */
typedef struct {
  float temperature;            /* 0 = greedy */
  float top_p;                  /* 1 = off */
  int32_t top_k;                /* 0 = off */
  /* What a token already chosen in this reply loses from its logit: presence
   * once, frequency per time it was chosen (0 = off).  The backend counts the
   * reply's tokens as it emits them, from set_reply() on. */
  float presence_penalty;
  float frequency_penalty;
  int32_t pad0;
  uint64_t seed;                /* sampling noise is keyed on (seed, position, token) */
  int32_t n_stop_ids;           /* stop_ids[0, min(n_stop_ids, FLYX_MAX_STOP_IDS)) are read */
  int32_t stop_ids[FLYX_MAX_STOP_IDS];
  int32_t limit;                /* stop once the position reaches this */
  /* The tokens the first token this launch (or prefill) chooses may be: bit
   * v % 32 of word v / 32 set = allowed; ceil(vocab / 32) words; NULL = any.
   * Read during the call (copied); the later rows of a round are not masked.
   * See ops/sampling/sampling.h for how a mask meets top-k and top-p. */
  const uint32_t* mask;
  /* A round's drafts, in place of the drafter's (drafts looked up in the
   * sequence itself), and the drafts of a round with a FLYX_NODE_GIVEN
   * node: drafts[i] is the draft for position drafts_at + i, drafts_at
   * being the position after the round's anchor.  Rows past n_drafts get
   * token 0.  Read during launch() only (copied); NULL = the drafter's own.
   * The engine passes them only with no other round in flight, so drafts_at
   * is exact. */
  const int32_t* drafts;
  int32_t n_drafts;
  int32_t drafts_at;
  /* The tree of a round whose plan's GIVEN node has arg1 = FLYX_GIVEN_TREE: draft i is a child of draft
   * parents[i] (-1: of the round's anchor), parents[i] < i.  NULL: a chain (parents[i] = i - 1); rows past
   * n_drafts continue the chain of the row before them.  Read during launch() only (copied). */
  const int32_t* parents;
  /* Relaxed acceptance (lossy; both 0: off, a draft is accepted only when it is the target's own choice).  At a row,
   * a child draft also qualifies when the target gives its token a probability (softmax of the logits over the
   * temperature, 1 when greedy, after penalties) of at least accept_min_p times the row's highest, or above
   * min(accept_typical, sqrt(accept_typical) * exp(-entropy)); of the qualifying children the most probable is taken,
   * and the round ends with the target's own choice at the last row taken.  See ops/sampling/sampling.h. */
  float accept_min_p;
  float accept_typical;
  /* Optional intra-round confidence stopping policy (ABI 41): read during launch() on adaptive plans (copied);
   * NULL or !enabled runs the plan's full width. */
  const flyx_stop_policy* stop_policy;
} flyx_gen_params;

/* One entry of the token ring: what a round (or the last prefill chunk) produced. */
typedef struct {
  uint64_t seq_no;              /* 1, 2, 3, ... per sequence */
  int32_t n_tokens;             /* tokens emitted (accepted drafts + the target's own) */
  int32_t n_accepted;           /* drafts accepted */
  int32_t stopped;              /* the round saw a stop id or hit the limit (or was already stopped) */
  int32_t position;             /* tokens in the state after this entry */
  int32_t tokens[FLYX_MAX_WIDE_ROWS];
  int32_t n_drafts;
  int32_t drafts[FLYX_MAX_WIDE_ROWS];
  /* The round's tree (ABI 24): draft i continues draft parents[i] (-1: the anchor); in a chain, i - 1.  With the
   * accepted tokens it names the branch that was accepted (siblings are distinct tokens). */
  int32_t parents[FLYX_MAX_WIDE_ROWS];
  /* Per-draft confidence from the drafter's head (ABI 40; 0 when unavailable): truncated softmax probability of the
   * top candidate and logit margin over the runner-up, independent of drafter architecture. */
  float draft_prob[FLYX_MAX_ROWS];
  float draft_margin[FLYX_MAX_ROWS];
} flyx_round_result;

typedef struct {
  char name[96];
  int32_t sm_major, sm_minor, sm_count;
  /* Whether the backend computes NVFP4 inputs (FLYX_ACT_NVFP4) on this device (ABI 25): the engine opens a model
   * whose file says NVFP4 inputs with bf16 ones where it does not. */
  int32_t fp4_inputs;
  uint64_t mem_total, mem_free;
} flyx_device_info;

/* Why the clocks are below their maximum (flyx_device_stats.clock_limits). */
enum {
  FLYX_CLOCK_IDLE = 1u << 0,         /* nothing to run */
  FLYX_CLOCK_POWER_CAP = 1u << 1,    /* the power limit (or a power brake) */
  FLYX_CLOCK_THERMAL = 1u << 2,      /* temperature */
  FLYX_CLOCK_HW_SLOWDOWN = 1u << 3,  /* a hardware slowdown signal */
  FLYX_CLOCK_APP_SETTING = 1u << 4,  /* application clocks set by someone */
  FLYX_CLOCK_SYNC_BOOST = 1u << 5,
  FLYX_CLOCK_DISPLAY = 1u << 6       /* the display's clock setting */
};

/* The device's live figures, for monitoring (device_stats); -1 = unknown. */
typedef struct {
  int32_t util;                 /* busy, percent of the driver's last sample period */
  int32_t mem_util;             /* memory controller busy, percent */
  int32_t temp_c;
  int32_t sm_clock_mhz;
  int32_t pcie_gen, pcie_width; /* the link as it runs now */
  int32_t fan;                  /* percent */
  int32_t mem_clock_mhz;
  double power_w, power_limit_w;
  uint64_t mem_used, mem_total; /* the whole device, every process (0 = unknown) */
  uint32_t clock_limits;        /* FLYX_CLOCK_* bits; 0 = running at full clocks or unknown */
  uint32_t pad0;
  /* Host memory this process holds page-locked for the device, all of it, as the system counts it (ABI 39): Windows'
   * non-local video memory of the process (DXGI; Task Manager's "Shared GPU memory").  It need not be in the
   * process's working set, which Windows trims under pressure.  0 where the system does not tell it. */
  uint64_t process_host;
} flyx_device_stats;

/* Where vision encoder weights live and how they run (ABI 44). */
typedef enum {
  FLYX_VISION_DEV_DISK = 0,     /* default: auto-deleted temp file on disk (0 VRAM, 0 locked host RAM), 3-stage streamed */
  FLYX_VISION_DEV_HOST = 1,     /* pinned host RAM (0 VRAM), double-buffered DMA streamed to GPU */
  FLYX_VISION_DEV_GPU  = 2,     /* vision weights resident in GPU VRAM */
  FLYX_VISION_DEV_AUTO = 3      /* GPU resident when >= 8 GB VRAM remain free after weights, else disk */
} flyx_vision_device;

/* On-the-fly quantization of floating-point vision linear weights at load time (ABI 44).
 * Pre-quantized GGML block weights in mmproj files are always kept packed as-is. */
typedef enum {
  FLYX_VISION_QUANT_Q8    = 0,  /* default: quantize float vision linears to Q8_0 (8.5 bpw, ~46% smaller) */
  FLYX_VISION_QUANT_EXACT = 1,  /* keep file format (BF16/F16/F32 or pre-quantized GGML blocks) */
  FLYX_VISION_QUANT_NVFP4 = 2   /* quantize float vision linears to NVFP4 (4.5 bpw, ~70% smaller) */
} flyx_vision_quant;

typedef struct {
  int32_t max_ctx;              /* tokens per sequence */
  int32_t verbose;
  int32_t kv_fp8;               /* K/V caches the target reads (state.draft == 0) in FP8 (e4m3, a scale per position and
                                   head); a drafter's as the backend chooses (CUDA: a one-layer one too) */
  /* Backend options, "key=value,key=value" (NULL or "" = none).  Each backend
   * documents its keys and fails open() on one it does not know; a key given
   * twice takes the later value.  The engine passes them through unread. */
  const char* options;
  int32_t vision_device;        /* flyx_vision_device (ABI 44; 0 = disk) */
  int32_t vision_quant;         /* flyx_vision_quant (ABI 44; 0 = q8) */
} flyx_open_options;

/* Device memory as the backend sees it (0 = unknown, e.g. a CPU backend). */
typedef struct {
  uint64_t weights;     /* taken by the model (weights, norms, tables) */
  uint64_t workspace;   /* shared scratch of the kernels */
  uint64_t free, total; /* of the device, now */
  /* What this process holds of the device as the system counts it, all of it: its models, their sequences and the
   * runtime's own (ABI 37).  0 where the system does not tell a process's share. */
  uint64_t process;
} flyx_memory_info;

typedef struct flyx_bmodel flyx_bmodel;
typedef struct flyx_plan flyx_plan;
typedef struct flyx_seq flyx_seq;

/* Where a sequence's K/V caches live, and how its attention reads them (ABI 27). The engine
 * decides the tiers (GPU memory first, then host memory: EngineOptions::kv_gpu / kv_ram); the backend places the
 * bytes.  Positions [0, device_positions) of every K/V cache the target reads are in device memory, the rest in
 * page-locked host memory the GPU reads over the bus; a kernel sees one cache either way, and the output is the same
 * bits wherever a position is.  device_positions >= max_ctx: all on the device. */
#define FLYX_KV_READ_ALL 0 /* every cached position at every step (exact attention) */
/* Rounds read a low-bit copy of every position on the device and refine the positions each row attends to most
 * from the K/V cache (after StrataFlux); prefill passes read the cache. method_arg: the copy's bits,
 * 2 to 4 (0: 2). */
#define FLYX_KV_STRATA 1
/* Exact where it costs nothing, the copy where exact would be slower (ABI 29): a round's row at a
 * position below the switch reads every position, as FLYX_KV_READ_ALL; at the switch or past it, as FLYX_KV_STRATA.
 * The backend picks the switch: where its exact attention, reading the positions past the device's part over the
 * bus, would get slower than the copy's; a session whose every position is below it keeps no copy.  The switch is a
 * position, never a time, so drafts do not change the output.  method_arg: the copy's bits (0: 2). */
#define FLYX_KV_AUTO 2
/* copy_host (ABI 30): of the low-bit copies of the caches in host memory (FLYX_KV_STRATA, FLYX_KV_AUTO),
 * how many are in host memory too, each read through a device staging that a round fills while the layers before
 * the copy's attention run; 0: every copy on the device. */
typedef struct {
  int32_t max_ctx;
  int32_t device_positions;
  int32_t method; /* FLYX_KV_READ_ALL, FLYX_KV_STRATA (ABI 28) or FLYX_KV_AUTO (ABI 29) */
  int32_t method_arg;
  int32_t copy_host;
} flyx_seq_layout;

/* What a sequence of a layout takes: device and host bytes (device: its round graphs among them, as seq_bytes's, ABI
 * 35), and the device positions the backend would use (its placement has a granularity: device_positions rounds down
 * to it, never to fewer than asked when all fit). */
typedef struct {
  uint64_t device, host;
  int32_t device_positions;
  int32_t strata_from; /* rounds read the low-bit copy from this position on; max_ctx or more: never (ABI 29) */
  int32_t copies;      /* the caches with a low-bit copy: copy_host at most (ABI 30) */
  /* Of host: the part mapped into the device's address range and nowhere in the process's (the K/V positions past
   * the device's part, ABI 39).  No process memory figure counts it (Linux's resident set, Windows' working set and
   * private bytes); the rest of host is page-locked memory the process maps, which they do. */
  uint64_t host_mapped;
} flyx_seq_size;

/* A sequence's round graphs (ABI 35; measurement and tests): the device bytes counted for the graphs it holds (its
 * own and its shares of rounds of several sequences') and for its room, which seq_bytes counts; its rounds launched
 * from a graph and without one, and the graphs captured for it (its own and the batch graphs it is in); the largest
 * kernel parameters a graph of it held, and its graphs' kernels whose parameters the driver did not say. */
typedef struct {
  uint64_t held, room;
  uint64_t from_graph, without_graph, captured;
  uint64_t params_most, params_unknown;
} flyx_graph_stats;

/* Vision encoder input (ABI 44): flat patch rows and spatial-temporal grid metadata. */
typedef struct {
  const float* patches;        /* [n_patches, patch_dim] row-major float32 */
  int32_t n_patches;
  int32_t patch_dim;
  const int32_t* grid_thw;     /* [n_grids * 3]: (T, H_patches, W_patches) per image/tile */
  int32_t n_grids;
  int32_t n_segs;              /* number of bidirectional attention segments */
  const int32_t* cu_seqlens;   /* [n_segs + 1]: cumulative patch offsets (0 .. n_patches) */
} flyx_vision_input;

/* Special prefill_snapshot() argument (ABI 44): suppress intra-pass snapshot capture for the next pass. */
#define FLYX_PREFILL_SNAP_NONE (-2)

/* Extended prefill pass input (ABI 44): supports external embedding injection (vision/audio)
 * and explicit 3D MRoPE coordinates with a decode-time scalar rope_delta. */
typedef struct {
  const int32_t* tokens;       /* [n] token IDs */
  int32_t n;
  int32_t n_embeds;            /* number of rows overridden by external embeddings (0 = text-only) */
  const float* embeds;         /* [n_embeds, hidden] float32 embeddings (NULL if n_embeds == 0) */
  const int32_t* embed_rows;   /* [n_embeds] sorted row indices in [0, n) receiving embeds[i] */
  const int32_t* pos_3d;       /* [n * 3] (t, h, w) coordinates for 3D MRoPE (NULL = standard 1D) */
  int32_t set_rope_delta;      /* 1: set sequence's decode rope_delta to `rope_delta` after this pass */
  int32_t rope_delta;          /* offset added to 1D decode positions: pos_1d = seq_pos + rope_delta */
  int32_t has_snap_rope_delta; /* 1: use snap_rope_delta for an intra-pass checkpoint captured in this pass */
  int32_t snap_rope_delta;     /* model-provided decode rope_delta at the intra-pass checkpoint position */
} flyx_prefill_input;

/* All functions return 0 on success unless stated otherwise; on failure
 * last_error() describes what happened. */
typedef struct flyx_backend_api {
  uint32_t abi_version;
  uint32_t struct_size;
  const char* name;

  int (*probe)(flyx_device_info* out);
  int (*open)(const flyx_model_desc* model, const flyx_open_options* opts, flyx_bmodel** out);
  void (*close)(flyx_bmodel* m);
  int (*lower)(flyx_bmodel* m, const flyx_program* program, flyx_plan** out);
  void (*plan_free)(flyx_plan* p);

  int (*seq_new)(flyx_bmodel* m, int32_t max_ctx, flyx_seq** out);
  void (*seq_free)(flyx_seq* s);

  /* Run tokens at the current position.  sample != NULL: also choose the next
   * token from the last row with these parameters, make it the anchor and
   * emit it as a ring entry. */
  int (*prefill)(flyx_plan* p, flyx_seq* s, const int32_t* tokens, int32_t n, const flyx_gen_params* sample);
  /* Enqueue one round.  Returns without waiting. */
  int (*launch)(flyx_plan* p, flyx_seq* s, const flyx_gen_params* g);
  /* Wait for and return the oldest ring entry not yet read. */
  int (*read)(flyx_seq* s, flyx_round_result* out);
  /* Move the state back to `pos` tokens with `anchor` as the next input.
   * Returns 1 when done, 0 when `pos` is outside the rollback window (the
   * state is then reset to empty), negative on error.  Waits for the device. */
  int (*rewind)(flyx_seq* s, int32_t pos, int32_t anchor);
  /* Save the state at the latest position a fresh prefill reproduces bit for
   * bit: the current one, or the multiple of prefill_align() the last prefill
   * pass crossed and kept (the last one it crossed, or prefill_snapshot's).
   * Returns that position, negative on error. */
  int (*checkpoint)(flyx_seq* s, int32_t slot);
  /* Restore a checkpoint; returns its position, or negative if the slot is empty. */
  int (*restore)(flyx_seq* s, int32_t slot, int32_t anchor);
  /* Testing: replace drafts with tokens[pos] (0 past the end). n = 0 turns it off. */
  int (*set_oracle)(flyx_seq* s, const int32_t* tokens, int32_t n);

  const char* (*last_error)(void);

  int (*memory)(flyx_bmodel* m, flyx_memory_info* out);
  /* Device bytes one sequence of max_ctx tokens takes: what seq_new allocates and (ABI 35) what the device's driver
   * takes for the rounds the backend keeps ready to launch again (the CUDA backend's graphs), at most, for the round
   * plans lowered so far; the backend keeps a sequence's inside it.  To size a plan no sequence has run yet, it may
   * capture a round of it on a sequence of its own (nothing runs) that it frees before it returns: an error where it
   * cannot make that sequence. */
  int (*seq_bytes)(flyx_bmodel* m, int32_t max_ctx, uint64_t* out);
  /* Prefill passes give a row the bits of one longer pass over the same
   * positions when they start at a multiple of this (1: anywhere). */
  int32_t (*prefill_align)(flyx_bmodel* m);
  /* Live figures of the device probe() chose.  Any thread, at any time after
   * probe(); touches no model or sequence and leaves last_error() alone.
   * Returns non-zero when the backend has none (e.g. a CPU backend). */
  int (*device_stats)(flyx_device_stats* out);
  /* Whether the next call that waits for the device would return at once
   * (ABI 13).  With a ring entry not yet read: 1 when the oldest one is
   * there, so read() would not wait.  With nothing to read: 1 when all that
   * was launched on the sequence has finished, so prefill(), rewind() and
   * checkpoint() would not wait.  0 while the device still works for this
   * sequence; negative on error.  Never waits: an engine that runs several
   * sequences asks before it turns to one of them. */
  int (*ready)(flyx_seq* s);
  /* The reply so far, for the penalties (ABI 14): these n tokens replace what
   * the sequence has counted; n = 0 starts a new reply.  The engine calls it
   * before the pass that chooses a reply's first token and after a rewind
   * inside a reply, and only for requests with a penalty; the tokens emitted
   * after it are counted by the backend.  Waits for the device. */
  int (*set_reply)(flyx_seq* s, const int32_t* tokens, int32_t n);
  /* A checkpoint as bytes any sequence of the same model and options can load
   * (ABI 20): the checkpoint's recurrent part and every
   * positional cache (the target's K/V, the drafters') of the positions
   * before it.  snapshot_bytes: how many bytes snapshot_save writes for
   * checkpoint `slot`.  snapshot_save: writes them to the host pieces `dst`
   * (ABI 22: in order, piece after piece; at least `bytes` in all); returns
   * the checkpoint's position, negative on error.  snapshot_load: makes the
   * bytes snapshot_save wrote (`src`, `bytes` of them) the sequence's state,
   * with `anchor` as the next input, as restore() does; returns the position,
   * negative on error or for bytes of another model or layout.  The
   * sequence's own checkpoints are left as they are.  All wait for the device.
   * ABI 32: begin = 0 saves a complete checkpoint. For a delta the caller
   * already holds the unchanged [0, begin) state of this sequence. Save the
   * rows after it, plus the checkpoint's entire recurrent and ring state.
   * Restore that exact parent before applying the delta; token equality
   * alone does not establish that two independently computed states match. */
  int (*snapshot_bytes)(flyx_seq* s, int32_t slot, int32_t begin, uint64_t* out);
  int (*snapshot_save)(flyx_seq* s, int32_t slot, int32_t begin, const flyx_host_pieces* dst, uint64_t bytes);
  int (*snapshot_load)(flyx_seq* s, const flyx_host_pieces* src, uint64_t bytes, int32_t anchor);
  /* One round of several sequences (ABI 21): sequence i runs
   * round plan plans[i] with gens[i], its rows after sequence i - 1's, at most
   * FLYX_MAX_ROWS rows in all; the plans may differ in their drafter and
   * width only.  Each sequence gets what launch(plans[i], seqs[i], gens[i])
   * gives it, read with read(); the weights are read once for all of them.
   * Returns without waiting. */
  int (*launch_batch)(flyx_plan* const* plans, flyx_seq* const* seqs, const flyx_gen_params* const* gens,
                      int32_t n);
  /* Host memory the device copies to and from at the bus's speed (ABI 22):
   * snapshots of long conversations (gigabytes) that would otherwise go
   * through pageable memory at a fraction of it.  host_alloc: `bytes` of it,
   * or NULL when the backend has none to give (the caller uses its own then);
   * host_free: gives back what host_alloc gave.  Getting it costs about what
   * filling it once does (every page faulted and pinned): a caller keeps it. */
  void* (*host_alloc)(uint64_t bytes);
  void (*host_free)(void* p);
  /* Measurement (ABI 26): row `row` of the head's logits in the last round
   * read (the anchor's row is 0), `vocab` floats into `out`, before the token
   * choice.  For rounds without a drafter, which may reuse the rows as scratch;
   * the CUDA backend scores every row only with backend option head_cert=0.
   * Waits for the device; 0, or negative on error. */
  int (*logits)(flyx_seq* s, int32_t row, float* out);
  /* seq_new and seq_bytes for a layout of the K/V caches (ABI 27); seq_new(m, n) is the layout
   * {n, n, FLYX_KV_READ_ALL}. */
  int (*seq_new_in)(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq** out);
  int (*seq_size)(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq_size* out);
  /* Give back device memory taken only to run faster, never anything that
   * changes a result (a second copy of a weight in a faster layout): `bytes` or
   * more if there is that much (ABI 34).  Returns the bytes freed, 0 when there
   * is nothing to give.  Plans lowered before may use what was freed: the engine
   * frees them and lowers again.  NULL: nothing to give. */
  uint64_t (*trim)(flyx_bmodel* m, uint64_t bytes);
  /* A sequence's round graphs (ABI 35, flyx_graph_stats); NULL, or non-zero: the backend keeps none. */
  int (*graph_stats)(flyx_seq* s, flyx_graph_stats* out);
  /* Where the next prefill pass keeps the state checkpoint() saves (ABI 38): `pos`, a multiple of prefill_align()
   * the pass crosses, in place of the last one it crosses (a message boundary inside a pass then needs no pass of
   * its own); -1: the last one.  For the next prefill call only.  NULL: always the last one. */
  int (*prefill_snapshot)(flyx_seq* s, int32_t pos);
  /* Stateless vision encoder execution (ABI 44): runs a FLYX_PHASE_VISION plan on `in` and writes
   * [n_out_rows, out_hidden] float32 embeddings into `out_embeds` (capacity `out_cap_rows` rows).
   * Writes actual output row count into `*out_rows` if non-NULL.  Waits for the device.
   * NULL when the backend does not implement vision blocks. */
  int (*encode_vision)(flyx_plan* vision_plan, const flyx_vision_input* in, float* out_embeds,
                       int32_t out_cap_rows, int32_t* out_rows);
  /* Extended prefill pass (ABI 44) supporting external embeddings (vision/audio) and 3D MRoPE positions.
   * NULL: text-only backend (engine falls back to prefill() when inputs are text-only). */
  int (*prefill_ex)(flyx_plan* p, flyx_seq* s, const flyx_prefill_input* in, const flyx_gen_params* sample);
} flyx_backend_api;

typedef const flyx_backend_api* (*flyx_backend_entry_fn)(void);

#ifdef __cplusplus
}
#endif
#endif /* FLYX_BACKEND_ABI_H */
