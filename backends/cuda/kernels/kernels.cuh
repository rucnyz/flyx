// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// CUDA kernels for the generic (untuned) path, as host-callable launchers.
//
// Conventions
//   * activations are float32 row-major [rows, n]; weights bf16 [N, K] row-major
//   * anything that changes between rounds (positions, anchor, ring slot, the
//     recurrent-state slot) is read on the device from DevParams, so a round
//     can be captured once in a CUDA graph and replayed
//   * every launcher takes the stream to enqueue on and never synchronizes
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <string>

#include "flyx/backend_abi.h"

namespace flyx::cuda {

using bf16 = __nv_bfloat16;
// The K/V caches' element: binary16, llama.cpp's default. bf16 would keep 8 significant bits of
// each key and value where binary16 keeps 11; keys and values stay far inside binary16's range.
using kv16 = __half;

// Per-sequence values that rounds read and advance on the device.
struct DevParams {
  int position;      // tokens in the state
  int anchor;        // next input token (emitted, not yet processed)
  int round;         // rounds committed; the next round starts from slot flyx_round_slot(round)
  int stopped;
  int last_n;        // drafts accepted by the last round
  int last_slot;     // slot the last round started from
  int start;         // position at the start of the last round
  int noop;          // the last launched round did nothing (already stopped)
  unsigned long long seq_no;
  int rope_delta;    // scalar MRoPE offset for decode / text turns after multimodal prefill
  int oracle_len;
  int last_round;    // round index that started at last_slot (for replay slot indexing)
};

struct DevGen {
  float temperature;  // 0 = greedy (see ops/sampling/sampling.h)
  float top_p;
  int top_k;
  int mask;  // 1: row 0's choice is restricted to the bitmask passed to choose_rows
  unsigned long long seed;
  int n_stop;
  int stop[FLYX_MAX_STOP_IDS];
  int limit;         // stop once position >= limit (0 = none)
  int max_ctx;       // the sequence's length: stop when no round fits after (flyx_room)
  float presence;    // penalties on tokens the reply already has (ops/sampling/sampling.h); 0 = off
  float frequency;
  float accept_min_p;    // relaxed acceptance (flyx_gen_params; ops::relaxed_qualifies); 0 = off
  float accept_typical;
};

// The reply so far as the penalties count it: how often each token was
// chosen, and the tokens chosen at least once (so a pass touches those only).
struct Reply {
  int* count;   // [vocab], one count per occurrence in the reply
  int* seen;    // the distinct tokens, in the order they first appeared
  int* n_seen;  // [1]
};

constexpr int kRing = 8;  // device ring entries (more than rounds in flight)
static_assert(kRing > FLYX_MAX_IN_FLIGHT, "a round in flight writes an entry not yet read");

// Which recurrent-state slot a kernel uses.  kSlotNext: the slot the round in
// progress commits into (only for rounds that always accept all their rows).
enum SlotSel : int { kSlotCur = 0, kSlotLast = 1, kSlotAfterLast = 2, kSlotNext = 3 };
// When a round-path kernel does nothing: never; when P->noop (commit after a
// stopped round); when P->stopped (the pass of a round that will be a no-op:
// it must not write a ring slot the rewind window still holds).
enum SkipWhen : int { kSkipNever = 0, kSkipIfNoop = 1, kSkipIfStopped = 2 };
// Where a kernel takes its base position from.
enum PosSel : int { kPosPosition = 0, kPosStart = 1 };

// ---- what a launch needs besides its operands --------------------------------------------
// Choices the shapes do not settle: one set per loaded model (the backend's
// options), the defaults in tests and benchmarks.
enum AttnImpl : int { kAttnAuto = 0, kAttnSplit = 1, kAttnPerHead = 2 };
// The selection's chunks: about this many pairs each, at most kStrataSelChunks; never more than
// kStrataSelChunk pairs (a thread of the selection holds 16 of them in registers), so a session's context is at most
// kStrataSelChunk kStrataSelChunks pairs (4M positions).
constexpr int kStrataSelChunk = 16384, kStrataSelChunks = 128;
struct KernelOptions {
  int tuning = -1;            // GEMM tuning table (gemm_tuning_for), -1 = the defaults
  bool row_gemms = false;     // every GEMM on the CUDA-core row path (debugging)
  int attention = kAttnAuto;  // kAttnSplit: no tensor-core prefill; kAttnPerHead: no KV-head grouping (and no FP8 K/V)
  bool pdl = false;           // programmatic dependent launch (launch_dependent); the backend: the GPU table's
                              // default (gemm_tuning_pdl) unless option pdl says
  bool head_cert = true;      // greedy rounds choose with the certified head
  bool gdn_round = true;      // a GDN layer's round in one kernel (up to 4 rows)
  bool stacks = true;         // a stack's run and the next of another block type in one round launch where compiled
                              // (gemm_blocks_staged_runs)
  bool round_glu = true;      // a round's SwiGLU in its dp4a gate|up GEMM (gemm_blocks_round_glu_takes)
  bool side_stream = true;    // a round's narrow GEMMs of a stack on the launch's side stream (gemm_blocks_staged_runs)
  bool log_shapes = false;    // the backend prints every GEMM shape once
  bool prefill_fp4 = false;   // NVFP4 weights in prefill GEMMs take NVFP4 activations (W4A4, sm_120a; gemm_w4a4):
                              // set from the model's weights (FLYX_ACT_NVFP4), not an option
  bool head_fp8 = false;      // the target's output head in FP8 (e4m3 per row; gemm_fp8_skinny)
  bool head_w12 = true;       // the target's output head in W12 for 1..16 rows (lossless; gemm_w12_skinny)
  bool rows_mma = true;       // decode and verification attention on tensor cores, binary16 caches (attention_rows_mma)
  int rows_mma_from = 8192;   // ... for rows of this many keys or more (by the row's own position; fewer: float32)
  int rows_mma_parts = 0;     // ... q and p as this many binary16 parts (1 to 3; 0: one where llama.cpp rounds them
                              // to one binary16 itself, three elsewhere)
  bool rows_mma_q8 = false;   // ... over FP8 caches, q as e4m3 and the scores on FP8 tensor cores
  bool fp8_mma = false;       // FP8 tensor cores summing in float32 (sm_90 and later; not sm_89): set at load
  int act_group = 32;         // measurement: the staging kernels' 8-bit scale covers this many values (64, 128, 256:
                              // coarser scales, written into each group of 32; producers that round their own
                              // outputs keep 32, so run with fusion=0 and round_glu=0)
  int act_fp4 = 0;            // measurement: the staging kernels first round each value as an NVFP4 (1) or
                              // MXFP8 (2) activation
  int strata_denom = -1;      // strata attention refines one in this many pairs of positions a row (-1: 16 with a
                              // 2-bit copy, 64 with 3 or 4 bits), at least
  int strata_least = 32;      // this many (StrataSel)
  int strata_recent = 0;      // ... and the last this many positions
  int strata_most = 256;      // ... and at most this many pairs (0: no bound), so the refinement stops growing with
                              // the context
  int strata_chunk = kStrataSelChunk;  // the selection's pairs a chunk (tests: small, for many chunks)
  int strata_hot = 2048;      // pairs a KV head strata attention keeps on the device of those it re-reads from host
                              // memory (a power of two; 0: none)
  int strata_from = -1;       // FLYX_KV_AUTO: rounds read the copy from this position on (-1: kStrataHostExact
                              // past the device's part of the K/V)
  int strata_grid = 0;        // the low-bit copy's grid (KvRef::grid)
  bool strata_mma = true;     // the copy's scan on tensor cores (false: the float32 kernel, for measurement)
  bool stage_ahead = true;    // K/V in tiers: a cache's positions in host memory copied to its staging beside the
                              // layers before its attention, two stagings taking turns (flyx_seq::stage_begin)
  int stage_blocks = 4;       // ... by this many blocks (kv_stage): fewer leave the bus idle, more take SMs from the
                              // layers (63K tokens on the 27B, 30K in host memory, 5070 Ti: 2 blocks 41.2 tok/s,
                              // 3 45.6, 4 45.5, 8 42.6, 16 33.3, 32 28.2)
  int mtp_window = 4096;      // the MTP layer's draft steps read its last N positions exactly (0: its
                              // every position, the whole cache on the device), when the context is past 8 N
  int mtp_sink = 64;          // ... and its first N exactly, without a copy
  int mtp_copy = 2;           // ... and every position through a copy of this many bits (2 to 4; strata attention over
                              // the MTP layer's cache; 0: none, only the positions above)
  bool mtp_ring = true;       // ... which a ring of slots holds (false: a slot a position, for tests)
  int stage_window = 16384;   // FLYX_KV_STRATA / AUTO: a staging's positions, a multiple of kFlashWindowAlign; a
                              // prefill whose keys go past them reads them a window at a time
};
// Device scratch for GEMM inputs: the split activations of the 1..16-row
// GEMMs (three bf16 parts, split_cap elements each) and the bf16 rows of the
// prefill GEMMs.  The backend owns it: each session one for its rounds, the
// model one for prefill passes; the kernels of one stream use it in turn.
constexpr int kQ8Tiles = 4096;  // tiles of a prefill GEMM in 8 bits whose runs of K can go side by side
struct Scratch {
  bf16* split = nullptr;
  size_t split_cap = 0;
  bf16* abuf = nullptr;
  size_t abuf_cap = 0;
  // GEMMs on GGML block weights (gemm_blocks.cu): 1..16 rows as binary16 in pairs of rows (pairs_cap values),
  // and every row's scale; their prefill rows are binary16 in abuf, with the sums of each row's runs of 16
  // values [rows][K / 16].
  uint16_t* pairs = nullptr;
  size_t pairs_cap = 0;
  float* row_scale = nullptr;
  float* run_sums = nullptr;
  // ... or, with inputs in 8 bits (W.q8), the rows as int8 [rows][K] and a scale per 32 [rows][K / 32]; a
  // round's (up to 16 rows, q8_round_cap values) laid out for its kernel, with the scale times the sum of each
  // 16 integers [rows][K / 16].  Between GEMMs on the same stream, draft_pick's partitioned selector may reuse
  // q8 (up to q8_round_cap bytes) as temporary per-row candidate pool keys/indices.
  int8_t* q8 = nullptr;
  float* q8_scale = nullptr;
  float* q8_sums = nullptr;
  size_t q8_round_cap = 0;
  // A prefill GEMM in 8 bits whose runs of K go side by side (gemm_blocks.cu): each run's sums [runs][rows][N]
  // (q8_parts_cap floats), and a counter per tile of the output (kQ8Tiles), zero between GEMMs.
  float* q8_parts = nullptr;
  size_t q8_parts_cap = 0;
  int* q8_count = nullptr;
  // A round's GEMM in 8 bits whose streams are split among blocks of threads (k_round_blocks_split): each set's sums
  // a tile of 8 columns (round_parts_cap floats), and a counter per tile (round_count_cap), zero between GEMMs.
  float* round_parts = nullptr;
  size_t round_parts_cap = 0;
  int* round_count = nullptr;
  size_t round_count_cap = 0;
  // The same for a launch on a launch's side stream (side_scratch; kSideColumns a launch): two split launches at once
  // take sums and counters of their own.
  float* side_parts = nullptr;
  size_t side_parts_cap = 0;
  int* side_count = nullptr;
  size_t side_count_cap = 0;
  // A round's SwiGLU written by its gate|up GEMM (Launch::glu): 8-bit rows laid out as a round's q8 (q8_round_cap
  // values, their scales and sums), which the down GEMM reads (glu_inputs) while q8 is still the gate|up GEMM's input.
  int8_t* q8_glu = nullptr;
  float* q8_glu_scale = nullptr;
  float* q8_glu_sums = nullptr;
  // The prefill rows in NVFP4 (gemm_w4a4): codes [rows][K/2], block scales [rows][K/16], row factors [rows].
  uint8_t *q4 = nullptr, *q4_scale = nullptr;
  float* q4_inv = nullptr;
  // The SwiGLU output of gemm_glu_staged [rows][I], the input of gemm_on_glu (abuf holds the GEMM's input then).
  bf16* glu = nullptr;
  size_t glu_cap = 0;
  // With NVFP4 inputs (scratch_*'s f32_rows; KernelOptions::prefill_fp4): a prefill pass's rows in float32, where
  // fused producers leave them (split_out), abuf_cap values, which the W4A4 kernels round once; and
  // each SwiGLU row's largest |value| [rows], which the W4A4 gate|up GEMM leaves beside its bf16 rows in glu.
  float* af32 = nullptr;
  float* glu_amax = nullptr;
  // Whether the last gemm_glu_staged ran W4A4 (glu_amax holds its rows' maxima; host state of the launches).
  mutable bool glu_w4a4 = false;
};
// Scratch for passes of up to `rows` rows whose GEMMs have K <= k_max: its
// bytes, and its buffers carved from `bytes` of memory at a 256-byte boundary
// (one layout does both; memory too small is a bug and aborts).  q8_parts:
// room for the runs of K of prefill GEMMs in 8 bits side by side (else they
// take them in turn).  scratch_carve() does no device work; scratch_in()
// also sets the tiles' counters to zero, which memory that starts zeroed
// already has.
size_t scratch_bytes(int rows, int k_max, bool q8_parts = true, bool f32_rows = false, bool need_glu = true,
                     bool need_q4 = true);
Scratch scratch_carve(void* mem, size_t bytes, int rows, int k_max, bool q8_parts = true, bool f32_rows = false,
                      bool need_glu = true, bool need_q4 = true);
Scratch scratch_in(void* mem, size_t bytes, int rows, int k_max, bool q8_parts = true, bool f32_rows = false);
// A round's gate|up rows gu [rows][ld] (gate columns 0..I-1, up I..2I-1): the GEMM that writes them also leaves
// silu(gate) * up in 8 bits in its scratch (Scratch::q8_glu), the bits swiglu_split(..., q8) gives.
struct RoundGlu {
  const float* gu;
  int ld, I;
};
// A second stream a launch may fork work onto and join back from (a round's narrow GEMMs of a stack beside its wide
// one, gemm_blocks_staged_runs; a captured round's graph runs them as parallel branches), with its two events.
struct SideStream {
  cudaStream_t st = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
};
// The most columns a launch on the side stream takes (the split kernel's sums and counters there: Scratch::side_*).
constexpr int kSideColumns = 2048;
// Where a launch runs and what it may use: stream, scratch (only launchers
// that stage GEMM inputs need it), options (null: the defaults); glu: a round's
// gate|up GEMM on block weights that also writes the SwiGLU (only where
// gemm_blocks_round_glu_takes says it can); side: a stream to fork onto (none: null).
struct Launch {
  cudaStream_t st = nullptr;
  const Scratch* scratch = nullptr;
  const KernelOptions* opt = nullptr;
  const RoundGlu* glu = nullptr;
  const SideStream* side = nullptr;
  // A round's rows (a sequence's own buffers): every row the bits of the 1..16-row kernels however many rows there
  // are (a wide lookup round's up to FLYX_MAX_WIDE_ROWS), never the prefill kernels'.
  bool round_rows = false;
  // A pass only drafting reads (DV's canvas): a bf16 GEMM of 17..64 rows takes bf16 inputs on k_gemm_skinny_mt,
  // deterministic, not a row's bits alone.
  bool draft = false;
};
// The scratch whose 8-bit rows are the SwiGLU a gate|up GEMM left (Launch::glu): the down GEMM's input.
Scratch glu_inputs(const Scratch& s);
// The scratch of a launch on the side stream: the same rows, the split kernel's sums and counters of its own.
Scratch side_scratch(const Scratch& s);

// y[r, n] (=|+=) sum_k x[r, k] * W[n, k].  K % 8 == 0.  Chooses CUDA-core or tensor-core path by rows: 1..16 the
// skinny kernel (gemm_rows where it does not fit, or with option row_gemms); more, the prefill kernels, or a round's
// rows (Launch::round_rows: a linear's bf16 weights, regions/linear.cpp, and an adapter's) k_gemm_skinny_mt in three
// parts (up to 64 rows; else the 1..16-row choice 16 rows at a time), a drafting pass's (Launch::draft)
// k_gemm_skinny_mt in one part (up to 64 rows).
void gemm(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
          const Launch& L);
// Explicit paths (for tests).
void gemm_rows(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
               cudaStream_t st);
void gemm_mma(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
              cudaStream_t st);
// rows <= 16, K % 32 == 0, x and W 16-byte aligned, ldx % 4 == 0.
void gemm_skinny(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                 const Launch& L);
// Prefill GEMMs, pipelined tensor cores (bf16 activations); used by gemm() /
// gemm_fp4() for more than 16 rows.  Explicit entry points for tests and
// benchmarks: cfg picks one of kTcConfigs compiled configurations (all give
// the same bits), -1 the default.
constexpr int kTcConfigs = 4;  // block tiles 64 x 128, 128 x 128, 64 x 64, 64 x 32 (gemm_prefill.cu)
void gemm_tc(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int M, bool accumulate,
             const Launch& L, int cfg = -1);
void gemm_fp4_tc(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
                 int ldy, int M, bool accumulate, const Launch& L, int cfg = -1);

// Benchmarks: one of the compiled skinny configurations (tile, warps, unroll),
// listed in backends/cuda/variants/candidates.toml.
void gemm_skinny_cfg(int cfg, const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows,
                     bool accumulate, const Launch& L);
struct Bf16Config {
  int nt, warps, unroll;
};
extern const Bf16Config kBf16Configs[];
extern const int kSkinnyConfigs;
// 17..64 rows of bf16 weights in one launch (backends/cuda/variants/candidates.toml, bf16_mt): each span of weights
// read once for every 16-row tile.  parts 3: every row the bits the 1..16-row kernel gives it (chunks 16; the kernel
// splits the float32 rows x itself); parts 1: bf16 inputs (in abuf), deterministic, those of no other kernel, and
// x == nullptr the bf16 rows a producer left in abuf.
struct MtConfig {
  int nt, chunks, unroll, parts;
};
extern const MtConfig kMtConfigs[];
extern const int kMtConfigCount;
void gemm_skinny_mt_cfg(int cfg, const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows,
                        bool accumulate, const Launch& L);
// Weights a launch of k_gemm_skinny_mt takes on the same rows, each its own N columns (y [rows][ldy], += with
// accumulate; or bf16 into y16), from columns x_col .. x_col + K of the input; first_block: the launcher's.
struct MtOut {
  const bf16* W = nullptr;
  float* y = nullptr;
  bf16* y16 = nullptr;
  int N = 0, ldy = 0, x_col = 0, accumulate = 0, first_block = 0;
};
constexpr int kMtGroups = 4;
struct MtGroups {
  MtOut g[kMtGroups];
  int n = 0;
};
// A drafting pass's GEMMs on bf16 rows x [rows][ldx] (1..64 rows, K % 32 == 0, ldx % 8 == 0): every weight of gs in
// one launch, one part (a configuration whose parts are 1); deterministic.
void gemm_draft_cfg(int cfg, const bf16* x, int ldx, int K, const MtGroups& gs, int rows, const Launch& L);
// The same in the configuration this GPU's table gives the first weight's shape (bf16_draft): its tile and unroll, the
// chunks K's (gemm_draft_chunks), so each weight gets the bits gemm() of a drafting pass gives it alone.
void gemm_draft(const bf16* x, int ldx, int K, const MtGroups& gs, int rows, const Launch& L);
// The chunks of a drafting pass's GEMM over K (its K partition, which decides its sums): by K alone, in every table and
// at every row count (tools/codegen/gen_gemm_tables.py, draft_chunks).
int gemm_draft_chunks(int K);
// float32 rows x [rows][ldx] as bf16 (to nearest) in the launch's abuf, stride K: gemm_draft's input.
const bf16* stage_bf16(const float* x, int ldx, int rows, int K, const Launch& L);

// W12 (lossless 12-bit bf16 weights; gemm_decode.cu): a row's weights in an
// exponent window e0 .. e0 + 15 as a byte (sign, offset, 3 mantissa bits) in
// hi [N][K] and a nibble (4 mantissa bits) in lo [N][K / 2]; the others in
// the row's escape list esc_start[n] .. esc_start[n + 1] (k, bf16 bits).
struct W12 {
  uint8_t* hi = nullptr;
  uint8_t* lo = nullptr;
  uint8_t* e0 = nullptr;       // [N]
  int* esc_start = nullptr;    // [N + 1]
  uint16_t* esc_k = nullptr;   // [escapes]
  uint16_t* esc_v = nullptr;   // [escapes]
};
size_t w12_plane_bytes(int N, int K);
// Encodes W [N][K] (K % 64 == 0, K < 65536) into w's planes, e0 and
// esc_start (all allocated by the caller); esc_start_host: N + 1 ints of host
// scratch.  Returns the number of escapes (allocate esc_k, esc_v for them).
int w12_encode(const bf16* W, int N, int K, W12* w, int* esc_start_host, cudaStream_t st);
// Fills the escape lists and decodes every weight as the GEMM does: returns
// the number of 8-weight groups that differ from W (0: lossless; -1: a CUDA error).
int w12_finish(const bf16* W, int N, int K, W12* w, cudaStream_t st);
// The columns cols[0 .. *count) of gemm_skinny(x, W) for 1..16 rows, each with
// gemm_skinny's bits, written to y's column cols[j] (K % 32 == 0).
void gemm_skinny_cols(const float* x, int ldx, const bf16* W, int K, const int* cols, const int* count, float* y,
                      int ldy, int rows, const Launch& L);
// The certified output head, greedy, 1..16 rows. logits holds the scores from gemm_fp8_skinny and coef
// holds the two planes from head_cert_coef. Afterwards each surviving column holds gemm_skinny's bits
// in every row and the others hold -inf. Each row's largest entry (lowest index on ties) matches
// gemm_skinny(x, W) over the whole vocabulary. Scratch holds
// norms [2 * rows] (input norms then input rounding-error norms), key [rows], cols [V], count [1].
void head_certify(const float* x, int ldx, const bf16* W, const float* coef, int V, int K, float* logits, int ldl,
                  int rows, float* norms, int* key, int* cols, int* count, const Launch& L);
// The GEMMs' arithmetic allowance. coef[n] bounds ||w_n - q_n|| + kHeadCertRounding (||w_n|| + ||q_n||),
// and coef[V + n] bounds ||q_n|| for the e4m3 row scaled by RS[n]. The score error is bounded by
// ||x|| coef[n] + ||x - bf16(x)|| coef[V + n] on the sliced path and by ||x|| coef[n] on the three-part path.
constexpr float kHeadCertRounding = 1e-3f;
void head_cert_coef(const bf16* W, const uint8_t* Q, const float* RS, int V, int K, float* coef, cudaStream_t st);
// 1..16 rows of x times W12 weights: the bits of gemm_skinny on the bf16 weights.
void gemm_w12_skinny(const float* x, int ldx, const W12& W, int N, int K, float* y, int ldy, int rows,
                     bool accumulate, const Launch& L);
void gemm_w12_skinny_cfg(int cfg, const float* x, int ldx, const W12& W, int N, int K, float* y, int ldy, int rows,
                         bool accumulate, const Launch& L);

// Tuning tables (backends/cuda/variants/<arch>/tables): the one for this
// device (KernelOptions::tuning): its own by name, else the one that serves its
// compute capability cc (major * 10 + minor; *by_arch then true),
// -1 if none; the table of that sku name (-1: none); every table's name; a table's name.
int gemm_tuning_for(const char* device_name, int cc, bool* by_arch = nullptr);
int gemm_tuning_named(const char* name);
std::string gemm_tuning_names();
const char* gemm_tuning_name(int tuning);
// Whether tuning table `tuning` has its kernels compiled for compute capability `cc` (always true with FLYX_TUNE_CANDIDATES).
bool gemm_tuning_compiled_for(int tuning, int cc);
// Whether kernels launch as programmatic dependents on the GPU of this table unless the options say.
bool gemm_tuning_pdl(int tuning);
// The configuration index a GEMM gets (fmt 0 = bf16, 1 = nvfp4, 2 = w12; 17..64 rows on k_gemm_skinny_mt, a
// kMtConfigs index: 3 = a round's, 4 = a drafting pass's) with a tuning table.
int gemm_config(int tuning, int fmt, int N, int K, int rows);
bool gemm_skinny_cfg_compiled(int cfg, int rows, bool w12 = false);
bool gemm_fp4_cfg_compiled(int cfg, int rows);
bool gemm_mt_cfg_compiled(int cfg);  // a kMtConfigs index

// NVFP4 weights (W4A16): Q [N, K/2] two e2m1 per byte (low nibble first),
// S [N, K/16] e4m3 block scales, RS [N] per-output-column scale (the tensor's
// scale2; stacked matrices may differ per row).  K % 32 == 0 (K % 64 for the
// 1..16-row path, which is bit-identical across row counts and configurations).
// NVFP4 copy (codes Q [N][K/2], e4m3 block scales S [N][K/16], row scales RS [N],
// the layout gemm_fp4 reads) of a bf16 matrix W [N, K], K % 16 == 0; values
// round to nearest.
void quantize_nvfp4(const bf16* W, int N, int K, uint8_t* Q, uint8_t* S, float* RS, cudaStream_t st);
// Q8_0 copy of a bf16 matrix W [N, K], K % 32 == 0, as the block kernels hold a GGUF's Q8_0 weights (payload [N][K /
// 32][32] codes, scales [N][K / 32] binary16): d = binary16(max |w| / 127) a block of 32, q = round(w / d).
void quantize_q8_0(const bf16* W, int N, int K, uint8_t* payload, uint16_t* scales, cudaStream_t st);
// Whether this build has code the current device can run (its architecture or PTX it can compile).
bool device_supported();
// The compute capability (major * 10 + minor, e.g. 80, 89, 120) the loaded device image was compiled for, or 0.
int compiled_arch();
void gemm_fp4(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
              int ldy, int rows, bool accumulate, const Launch& L);
void gemm_fp4_skinny_cfg(int cfg, const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N,
                         int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
// A round's NVFP4 GEMM of any number of rows, every row's bits those of the 1..16-row kernels (the shape's K
// partition): 17..64 rows (staged in the scratch's prefill rows) decode each span of weights once for up to four
// tiles of 16 rows (k_gemm_skinny_fp4_mt);
// more, or a configuration it is not compiled for, 16 rows at a time.  For rounds of large trees (phase 2).  cfg:
// a 1..16-row configuration (tests), -1 the shape's for 9..16 rows.
void gemm_fp4_rows(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K,
                   float* y, int ldy, int rows, bool accumulate, const Launch& L, int cfg = -1);
void gemm_fp4_mma(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
                  int ldy, int rows, bool accumulate, cudaStream_t st);
// The compiled NVFP4 1..16-row configurations: (span, chunks) decide the bits,
// (tile, warps, unroll) only the speed.
struct Fp4Config {
  int span, chunks, nt, warps, unroll;
};
extern const Fp4Config kFp4Configs[];
extern const int kFp4ConfigCount;
// FP8 weights for the output head (opt-in, KernelOptions::head_fp8): Q [N, K]
// e4m3 codes, RS [N] = max|W[n]| / 448 (quantize_fp8_rows rounds to nearest).
void quantize_fp8_rows(const bf16* W, int N, int K, uint8_t* Q, float* RS, cudaStream_t st);
// gemm_fp8_skinny: 1..16 rows, K % 32 == 0. The sliced kernel reads only the bf16 high part of x.
__host__ __device__ constexpr bool gemm_fp8_skinny_rounds_input(int K) { return K % 256 == 0; }
void gemm_fp8_skinny(const float* x, int ldx, const uint8_t* W, const float* RS, int N, int K, float* y, int ldy,
                     int rows, bool accumulate, const Launch& L);
// ---- GGML block weights (GGUF files) packed on the device (gemm_blocks.cu) ----
// A [rows, K] weight of a GGML block type (flyx_dtype), K / values blocks a
// row: per block its payload (the file's block without the binary16 scales
// that lead or trail it, so its fields are aligned) and those scales as bits
// in a plane of their own.  blocks_pack lays the file's blocks out that way.
struct BlockMat {
  const uint8_t* data = nullptr;    // [rows * K / values] payloads
  const uint16_t* scales = nullptr; // their binary16 scales
  int dtype = 0;
  int rows = 0;
  bool q8 = false;                  // every pass takes its inputs in 8 bits (FLYX_ACT_Q8)
};
struct BlockLayout {
  int values, bytes, stride, scales;  // per block: weights, bytes in the file, payload bytes and scales on the device
};
BlockLayout blocks_layout(int dtype);  // values 0: not a type the kernels take
// Builds the device tables the decoders read (ggml's codebooks as selectors
// of their rows' magnitudes, bit fields and scales as binary16), once per
// process.  The launchers below build them on first use; a backend calls
// this when it loads block weights, so that the first use is not inside a
// graph capture.
void blocks_init();
// Whether the kernels take a [N, K] weight of this type (a GGML block type, rows of whole blocks).
bool blocks_supported(int dtype, int K);
// Host: n blocks as the file stores them -> their payloads and scales.
void blocks_pack(int dtype, const uint8_t* file, size_t n, uint8_t* payload, uint16_t* scales);
// y[r, n] (=|+=) sum_k a(x)[r, k] W[n, k] with W's exact weights, a(x) the
// rows as the kernels hold them: rounded to bf16, then binary16 over a power
// of two per row (ops/linear/ref.h: linear_a16h).  Any number of rows (1..16:
// one kernel whose bits depend neither on the row count nor on the
// configuration; more: tiles whose bits do not depend on the pass's length).
// K % 32 == 0, x 8-byte aligned, ldx even.  The rows are staged once per
// input, from float32 (blocks_stage) or from the bf16 rows a fused producer
// left in the launch's scratch (blocks_stage_presplit); then each weight
// that shares the input runs on them (gemm_blocks_staged).  gemm_blocks is
// blocks_stage and gemm_blocks_staged.
void gemm_blocks(const float* x, int ldx, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                 const Launch& L);
void blocks_stage(const float* x, int ldx, int K, int rows, const Launch& L, bool q8 = false);
void blocks_stage_presplit(int K, int rows, const Launch& L, bool q8 = false);
void gemm_blocks_staged(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
// A stack's n runs on the same staged rows, their columns side by side in y: a launch a run, or in rounds two or three
// neighbouring runs in one where their types are compiled together (gemm_blocks_ops.h) and the first run's rules run
// it on dp4a (KernelOptions::stacks): the fewest launches.  In a round with a side stream (Launch::side,
// KernelOptions::side_stream), launches of up to kSideColumns columns go there, beside the wider ones.  The same bits
// either way.
void gemm_blocks_staged_runs(const BlockMat* W, int n, int K, float* y, int ldy, int rows, bool accumulate,
                             const Launch& L);
// Whether a round of `rows` rows on a gate|up stack (runs W[0..n-1], 2 I columns) writes its SwiGLU as well
// (Launch::glu): one row (decoding), 8-bit inputs, every run in a launch of its own whose rules run it on the dp4a
// kernel in a configuration compiled with a SwiGLU (one part of the rows, at most kRoundGluRows of them, not in
// windows), where a block takes a group of 32 outputs' columns at a time.  KernelOptions::round_glu.
bool gemm_blocks_round_glu_takes(const BlockMat* W, int n, int rows, int I, const Launch& L);
// Benchmarks and tests: one of the compiled 1..16-row configurations
// (backends/cuda/variants/candidates.toml), on staged rows.  All give the same bits.
// False, and nothing run: the configuration has no room for this many rows on this device.
// Passes of more than 16 rows with inputs in 8 bits (W.q8, staged by blocks_stage(..., true)): every
// compiled configuration (bn columns and bm rows a block of threads, tiles of rows qk values of K long) gives the
// same bits, and so does every spread of a shape's runs of K: -1 as the launch chooses, 0 one block a tile taking
// the runs in turn, 1 a block a run (their sums through the scratch where it holds them, else folded onto y in order).
// runs: 0 the shape's (gemm_blocks_q8_runs); more,
// that many (at most a run a window): other bits, for choosing the table (cuda_bench q8runs).
struct BlocksQ8Config {
  int bn, qk, bm;
};
extern const BlocksQ8Config kBlocksQ8Configs[];
extern const int kBlocksQ8ConfigCount;
void gemm_blocks_q8_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                        const Launch& L, int spread = -1, int runs = 0);
bool gemm_blocks_q8_cfg_compiled(int dtype, int cfg);
bool gemm_blocks_cfg_compiled(int cfg);
// On a GPU of tuning table `tuning` (gemm_tuning_for; -1 the defaults): the runs of K a pass with inputs in 8 bits
// adds a layer of N columns up in (fixed per shape), and the configuration a pass of `rows` rows runs by default on a
// layer of block type `dtype`.
int gemm_blocks_q8_runs(int tuning, int N, int K);
int gemm_blocks_q8_default(int tuning, int dtype, int N, int rows);
// Rounds (1..16 rows) with inputs in 8 bits on rows staged by blocks_stage(..., true): kernel 0 dp4a, 1 the integer
// tensor cores with the rows in the MMA's first operand, 2 with the columns there (8 rows; more: kernel 1), 3 and 4
// the streams split among blocks of threads (rows, columns there), 5 among a block's warps (the types
// gemm_blocks_round_warps_takes; others run kernel 3); the same bits (gemm_blocks_staged picks by the rules).  The dp4a kernel's compiled configurations (variants/candidates.toml,
// blocks_round: the rows it holds, columns a warp takes at once, groups a lane loads ahead per column, 0 by the
// type's unit, warps a block, the rows staged in shared memory or not) give the same bits too; _cfg runs one (rows: at most its own); _default_kernel names the kernel a round
// of `rows` rows on a layer of type `dtype` and N columns runs by default on a GPU of tuning table `tuning`,
// _default its dp4a configuration (-1 when the kernel is another).
void gemm_blocks_round_q8_kernel(int kernel, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                                 const Launch& L);
bool gemm_blocks_round_warps_takes(int dtype);
struct BlocksRoundConfig {
  int rows, columns, ahead, warps, staged, parts;
};
extern const BlocksRoundConfig kBlocksRoundConfigs[];
extern const int kBlocksRoundConfigCount;
void gemm_blocks_round_q8_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                              const Launch& L);
bool gemm_blocks_round_cfg_compiled(int dtype, int cfg);
bool gemm_blocks_round_kernel_compiled(int dtype, int kernel, int rows);
bool gemm_blocks_round_glu_compiled(int dtype, int cfg);
// A stack's n runs (2 or 3; their columns side by side in y) in one launch, the first run in configuration cfg;
// false, and nothing run, where their types are not compiled together (gemm_blocks_round_stack_compiled) or cfg not
// for stacks (only the configurations a rule gives the first run's type are).
bool gemm_blocks_round_stack_compiled(const int* dtype, int n);
bool gemm_blocks_round_q8_stack_cfg(int cfg, const BlockMat* W, int n, int K, float* y, int ldy, int rows,
                                    bool accumulate, const Launch& L);
int gemm_blocks_round_default(int tuning, int dtype, int N, int rows);
int gemm_blocks_round_default_kernel(int tuning, int dtype, int N, int rows);
struct BlocksConfig {
  int nt, nw, kw;  // tiles of 8 columns a warp; warps that share a window; runs of K at once
};
extern const BlocksConfig kBlocksConfigs[];
extern const int kBlocksConfigCount;
bool gemm_blocks_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
// The configuration `rows` rows of a layer of N columns run by default on a GPU of tuning table `tuning`.
int gemm_blocks_default(int tuning, int N, int rows);
// x[r] = W[tokens[r]] for r < rows: rows of a table of block weights (W: [vocab, H]), as float32.
void embed_blocks(const int* tokens, const BlockMat& W, int H, float* x, int rows, cudaStream_t st);
// out[r ld + c] = W[idx[r ld + c]] . x[r] for r < rows, c < count[r ld] (<= cap): dots of rows of a table of block
// weights (W: [vocab, K], its values as embed_blocks gives them) with rows of x, lanes over K 4 values at a time in
// order, then a fixed tree; idx, count and out at the same stride ld (elements).
void dots_blocks(const BlockMat& W, int K, const int* idx, const int* count, float* out, size_t ld, int cap,
                 const float* x, int ldx, int rows, cudaStream_t st);

// Draft tokens from a two-stage head (draft_head.cu): a 1-bit copy of the
// head W [N][K] (K % 128 == 0, K <= 4096) gives rough scores of every token, the
// kDraftCands best of a row are scored exactly against the bf16 rows of W,
// and their top k (high to low, the lowest index on ties) come out as idx
// [rows][k] and val [rows][k] (val may be null).  rough: scratch of rows x
// max(N, kDraftPickFloats) floats.  For drafts only: the target verifies every one.
// kDraftCandCap: the most candidates scored exactly (the kDraftCands best and their ties).
constexpr int kDraftCands = 256, kDraftPickFloats = 1025, kDraftCandCap = 512;
struct SignHead {
  uint32_t* bits = nullptr;  // word w of row n at sign_word_at: bit j set when weight 32 w + j >= +0
  bf16* scales = nullptr;    // scale g of row n at sign_scale_at: mean |w| of weights 64 g .. 64 g + 63
};
// Tiles of 256 rows x 128 weights, each a tile's rows one after another
// (whole tiles: allocate for N rounded up to 256 rows).
__host__ __device__ inline size_t sign_word_at(int n, int w, int K) {
  return ((size_t(n >> 8) * (K / 128) + (w >> 2)) * 256 + (n & 255)) * 4 + (w & 3);
}
__host__ __device__ inline size_t sign_scale_at(int n, int g, int K) {
  return ((size_t(n >> 8) * (K / 128) + (g >> 1)) * 256 + (n & 255)) * 2 + (g & 1);
}
size_t sign_head_bytes(int N, int K);  // bits and scales
void make_sign_head(const bf16* W, int N, int K, SignHead* h, cudaStream_t st);
void gemm_sign_rough(const float* x, int ldx, const SignHead& h, int N, int K, float* y, int ldy, int rows,
                     const Launch& L, bool exact_1row = false);
// y[n] += rough(x, h)[n] + (bias ? bias[n] : 0) for 1 row, L2-caching h for small tables reused across steps.
void gemm_sign_rough_add(const float* x, int ldx, const SignHead& h, const float* bias, int N, int K, float* y,
                         int ldy, const Launch& L);
// Shared two-stage candidate-set primitives over rough [rows][ld] (ld >= max(V, kDraftPickFloats)):
// select ~kDraftCands survivors per row, score survivors against any bf16 matrix (overwrite or accumulate + bias +
// softcap), combine a packed dots_blocks candidate vector + bias + softcap, and pick top-k with prob/margin.
void draft_select_cands(float* rough, int ld, int V, int rows, const Launch& L);
void draft_score_bf16(const float* x, int ldx, const bf16* W, const float* bias, int K, int rows, float* rough, int ld,
                      bool accumulate, float softcap, cudaStream_t st);
void draft_combine_cands(const float* add, int ld_add, const float* bias, float softcap, int rows, float* rough, int ld,
                         cudaStream_t st);
void draft_pick_topk(float* rough, int ld, int rows, int k, int* idx, float* val, cudaStream_t st,
                     float* prob = nullptr, float* margin = nullptr);
void draft_pick(const float* x, int ldx, const SignHead& h, const bf16* W, int V, int K, int rows, int k, int* idx,
                float* val, float* rough, const Launch& L, bool exact_1row = false, float* prob = nullptr,
                float* margin = nullptr);
// The same for a head of packed GGML blocks W [V][K] (any K % 128 == 0): its 1-bit copy made from its rows as
// float32 (embed_blocks, a run of rows n0 .. n0 + n - 1 at a time: W [n][K]), and the candidates of every row
// scored in float32 from the packed rows in one launch (dots_blocks).
void make_sign_head_rows(const float* W, int n0, int n, int K, SignHead* h, cudaStream_t st);
void draft_pick_blocks(const float* x, int ldx, const SignHead& h, const BlockMat& W, int V, int K, int rows, int k,
                       int* idx, float* val, float* rough, const Launch& L, bool exact_1row = false,
                       float* prob = nullptr, float* margin = nullptr);
// Tests: decode n words of 8 e2m1 codes into 4 bf16x2 each.
void fp4_decode_test(const uint32_t* in, uint32_t* out, int n, cudaStream_t st);

// y = x * rsqrt(mean(x^2) + eps) * (offset + w) per row.
void rmsnorm(const float* x, int ldx, const float* w, float offset, float eps, float* y, int ldy, int rows, int n,
             cudaStream_t st);
// The same of bf16 rows (a prefill pass's bf16 residual): the values of the float32 rows they convert to.
void rmsnorm(const bf16* x, int ldx, const float* w, float offset, float eps, float* y, int ldy, int rows, int n,
             cudaStream_t st);

// x[r] = E[tokens[r]] for r < rows.
void embed(const int* tokens, const bf16* E, int H, float* x, int rows, cudaStream_t st);
void embed(const int* tokens, const bf16* E, int H, bf16* x, int rows, cudaStream_t st);  // bf16 rows, as they are
// tokens = [P.anchor, fill...] when fill >= 0, else [P.anchor, drafts[0..rows-1)]
void build_rows(const DevParams* P, const int* drafts, int fill, int* tokens, int rows, cudaStream_t st);

// A cache in tiers keeps its first positions on the device and maps the rest from host memory, in pieces of the
// mapping's granularity `gran`: the device's part of a cache asked for `positions` rows of `row` bytes is the bytes
// below the last boundary at or below them (`even`: at an even count of whole rows, strata's pairs of positions
// staying on one side).  Where `row` does not divide `gran` that part can end inside a row.
inline size_t tiered_device_bytes(int positions, size_t row, size_t gran, bool even) {
  size_t dev = size_t(positions) * row / gran * gran;
  if (even)
    while (dev >= gran && ((dev / row) & 1)) dev -= gran;
  return dev;
}
// The most positions, `positions` at most, at which the device's part of every cache of these row sizes ends on a
// whole row (tiered_device_bytes == positions * row for each).  Asked for these, a layer's K and V and every layer
// split at the same position; asked for more, each cache would round down by itself, a granule apart from the one
// allocated before it (rows of 768 bytes, 13,000 positions: 10,922 for the first cache, 8,192 for the next), though
// a layer's scales, pageable buffers and tail ring are split at one position for both.
inline int tiered_device_positions(int positions, const size_t* rows, int n, size_t gran, bool even) {
  for (bool moved = true; moved && positions > 0;) {
    moved = false;
    for (int i = 0; i < n; ++i) {
      const int held = int(tiered_device_bytes(positions, rows[i], gran, even) / rows[i]);
      if (held < positions) positions = held, moved = true;
    }
  }
  return positions;
}

// A layer's K/V cache, [cap][nkv][hd]: binary16 (kv16), or e4m3 with one float scale per
// (position, kv head) in ks / vs (amax / 448 of that head's row).  FP8 when ks is set.
// Per-pair gather request written by k_strata_page_list and serviced by the host callback from pageable RAM into a
// compact mapped pinned bounce buffer (zero WDDM non-local footprint for cold K/V).
struct PageGatherReq {
  int32_t kvh;
  int32_t q;         // pair index (positions 2*q, 2*q + 1)
  int32_t hot_slot;  // >= 0: install into kv.hot[kvh, hot_slot]; < 0: hash-collision bounce-only slot
  int32_t pad;
};

struct PageGatherCtx {
  // Cold host K/V in ordinary pageable memory (VirtualAlloc / mmap; not page-locked).
  const uint8_t* k_page = nullptr;
  const uint8_t* v_page = nullptr;
  const float* ks_page = nullptr;
  const float* vs_page = nullptr;
  // Shared per-session mapped pinned staging (host and device views of the same allocation).
  PageGatherReq* req_host = nullptr;
  PageGatherReq* req_dev = nullptr;
  int32_t* miss_count_host = nullptr;
  int32_t* miss_count_dev = nullptr;
  int32_t* miss_map = nullptr;       // device buffer [FLYX_MAX_ROWS][nkv][sel_stride]: bounce index or -1
  uint8_t* bounce_kv_host = nullptr;
  uint8_t* bounce_kv_dev = nullptr;
  float* bounce_scale_host = nullptr;
  float* bounce_scale_dev = nullptr;
  int nkv = 0, hd = 0, elem = 1;     // elem: 1 (FP8 e4m3) or 2 (binary16)
  int max_misses = 0;
  int cap = 0;
  int device_positions = 0;          // k_page/v_page/ks_page/vs_page start at position device_positions
};

// Descriptor for draining a pageable layer's newly committed rows from its GPU tail ring into a per-round-slot mapped
// pinned drain buffer at the end of a round (read_result copies them into pageable RAM on the host).
struct PageDrainLayer {
  const uint8_t *k = nullptr, *v = nullptr;
  const float *ks = nullptr, *vs = nullptr;
  uint8_t *drain_k = nullptr, *drain_v = nullptr;  // device pointer to [kRing][max_rows][row_bytes]
  float *drain_ks = nullptr, *drain_vs = nullptr;  // device pointer to [kRing][max_rows][nkv] (FP8 only)
  uint8_t *drain_k_host = nullptr, *drain_v_host = nullptr;
  float *drain_ks_host = nullptr, *drain_vs_host = nullptr;
  size_t row_bytes = 0;
  int nkv = 0, device_positions = 0, ring_base = 0, ring = 0;
  __host__ __device__ int slot_of(int pos) const {
    return ring <= 0 || pos < device_positions ? pos
                                               : ring_base + (pos - device_positions) % (ring - ring_base);
  }
};

struct KvRef {
  void* k = nullptr;
  void* v = nullptr;
  float* ks = nullptr;
  float* vs = nullptr;
  // The low-bit copy strata attention reads (null: none): every position's K and V rotated by the
  // normalized Walsh-Hadamard transform and quantized uniformly to `bits` bits per position and head, `words` 32-bit
  // words of codes each, laid out as StrataLayout says, and its (lo, step) in binary16
  // ([cap][nkv]; the codes chosen on that grid): a code c stands for lo + (c + 1/2) step.
  uint32_t* kq = nullptr;
  uint32_t* vq = nullptr;
  __half2* kqs = nullptr;
  __half2* vqs = nullptr;
  int bits = 0, words = 0;
  // The copy's grid: 0 from the head's smallest to its largest value; 1 a step fitted to a normal distribution of
  // the head's mean and deviation (the rotated values are near one), the codes clamped at its ends.
  int grid = 0;
  // Pairs of positions in host memory strata attention re-read lately, kept on the device (null:
  // none): per KV head 2^hot_bits slots, a pair q in slot hot_slot(q); a slot's tag the pair it holds (-1: none), its
  // bytes [K, V][2 positions][hd] (FP8 caches only), hot_claim the winning pair * rows + row (kHotFree: none).
  // host_from: the first position in host memory.
  int32_t* hot_tag = nullptr;
  int32_t* hot_claim = nullptr;
  uint8_t* hot = nullptr;
  int hot_bits = 0;
  int host_from = INT32_MAX;
  // A cache of `ring` slots (the MTP layer's), its first `sink` positions in the first slots for good,
  // the others in turn in the rest; 0: a slot a position.
  int ring = 0, sink = 0;
  // FP8 scales where the positions are: ks and vs hold those of positions [0, scale_split), ks_far
  // and vs_far the others' (in host memory, or staged with the K/V); both indexed by position.  hot_scale: the scales
  // of the pairs kept on the device, [KV head][slot][K 2q, K 2q + 1, V 2q, V 2q + 1].
  float* ks_far = nullptr;
  float* vs_far = nullptr;
  int scale_split = INT32_MAX;
  float* hot_scale = nullptr;
  // Pageable host K/V layer (null for GPU / pinned host layers): cold positions [host_from, ring_lo) live in pageable
  // RAM and are gathered via a CUDA graph host node; positions >= ring_lo = max(host_from, (P_hot->pos - ring_safe) & ~1)
  // are live in the GPU tail ring (slot_of).
  PageGatherCtx* page_ctx = nullptr;
  int ring_safe = 0;
  // Optional gap slots skipped between `sink` and the tail ring start (1 when VMM `dev` is not a multiple of row
  // bytes so the first tail-ring slot never aliases the device VMM handle shared with the staged view).
  int ring_pad = 0;
  __host__ __device__ bool fp8() const { return ks != nullptr; }
  __host__ __device__ float* kscales(int pos) const { return pos < scale_split ? ks : ks_far; }
  __host__ __device__ float* vscales(int pos) const { return pos < scale_split ? vs : vs_far; }
  __host__ __device__ int slot_of(int pos) const {
    if (ring <= 0 || pos < sink) return pos;
    const int base = sink + ring_pad;
    return base + (pos - sink) % (ring - base);
  }
  // Fibonacci hashing: pairs near one another spread over the slots.
  __host__ __device__ int hot_slot(int q) const { return int((uint32_t(q) * 2654435761u) >> (32 - hot_bits)); }
};
constexpr int32_t kHotFree = 0x7f7f7f7f;  // a slot no pair claims (what a memset of 0x7f bytes leaves)

// The strata copy's layout, made for the tensor-core scan, whose warps decode their own fragments: a
// word holds pairs of codes as two 16-bit lanes (slot s of a word: the pair's first code at bit s bits, its second 16
// bits above), cph pairs a word, and 16 positions of a KV head keep their words together ([cap / 16][nkv][16 words])
// so that every word slot of a warp's 32 lanes is one 128-byte line.  K: the dims (2P, 2P + 1) are pair j = P / 4 of
// lane group t = P % 4 (lanes 4 g + t of an MMA's A fragment hold them, for keys g and g + 8 of 16); V: the dims
// (16 m + g, 16 m + g + 8) pair m of lane group g (the A fragment of V^T, keys 2 t, 2 t + 1, 2 t + 8, 2 t + 9).  A
// group's pairs fill words of its own; those left over share words, kgpw (vgpw) groups a word.
struct StrataLayout {
  int bits = 0, cph = 0, kown = 0, kleft = 0, kgpw = 1, ksh = 0, vown = 0, vleft = 0, vgpw = 1, vsh = 0;
  __host__ __device__ constexpr StrataLayout(int hd, int b) {
    bits = b;
    cph = 16 / b;
    kown = hd / 8 / cph;
    kleft = hd / 8 - kown * cph;
    kgpw = kleft ? cph / kleft : 1;
    ksh = kleft ? (4 + kgpw - 1) / kgpw : 0;
    vown = hd / 16 / cph;
    vleft = hd / 16 - vown * cph;
    vgpw = vleft ? cph / vleft : 1;
    vsh = vleft ? (8 + vgpw - 1) / vgpw : 0;
  }
  __host__ __device__ constexpr int kwords() const { return 4 * kown + ksh; }
  __host__ __device__ constexpr int vwords() const { return 8 * vown + vsh; }
  __host__ __device__ constexpr int words() const { return kwords() > vwords() ? kwords() : vwords(); }
  // K pair j of lane group t (V pair j of lane group g): the position's word w holding it, and its slot.
  __host__ __device__ constexpr void k_at(int t, int j, int* w, int* s) const {
    if (j < kown * cph) *w = t * kown + j / cph, *s = j % cph;
    else *w = 4 * kown + t / kgpw, *s = (t % kgpw) * kleft + (j - kown * cph);
  }
  __host__ __device__ constexpr void v_at(int g, int j, int* w, int* s) const {
    if (j < vown * cph) *w = g * vown + j / cph, *s = j % cph;
    else *w = 8 * vown + g / vgpw, *s = (g % vgpw) * vleft + (j - vown * cph);
  }
  // Word w of the position at p16 (0-15) of its 16, within their 16 words: K's own words in lane order, slot i of
  // keys g and g + 8 for lanes 4 g + t; V's for keys 2 t + e + 8 h.
  __host__ __device__ constexpr int k_index(int p16, int w) const {
    const int g = p16 & 7, h = p16 >> 3;
    if (w < 4 * kown) return ((w % kown) * 2 + h) * 32 + g * 4 + w / kown;
    return kown * 64 + (h * 8 + g) * ksh + (w - 4 * kown);
  }
  __host__ __device__ constexpr int v_index(int p16, int w) const {
    const int h = p16 >> 3, t = (p16 & 7) >> 1, e = p16 & 1;
    if (w < 8 * vown) return (((w % vown) * 2 + h) * 2 + e) * 32 + (w / vown) * 4 + t;
    return vown * 128 + ((h * 2 + e) * 4 + t) * vsh + (w - 8 * vown);
  }
  // Where dim d of the position at p16 is: its word's index among the 16 positions' words, and the code's shift.
  __host__ __device__ constexpr void k_code(int p16, int d, int* index, int* shift) const {
    int w = 0, s = 0;
    k_at((d >> 1) & 3, d >> 3, &w, &s);
    *index = k_index(p16, w);
    *shift = (d & 1) * 16 + s * bits;
  }
  __host__ __device__ constexpr void v_code(int p16, int d, int* index, int* shift) const {
    int w = 0, s = 0;
    v_at(d & 7, d >> 4, &w, &s);
    *index = v_index(p16, w);
    *shift = ((d >> 3) & 1) * 16 + s * bits;
  }
};
// The first word of position pos's 16 for KV head h; the positions a copy of cap positions takes (whole tiles of the
// scan's 32); and the positions whose words hold the first pos positions' (whole 16s).
__host__ __device__ inline size_t strata_words_at(int pos, int h, int nkv, int words) {
  return (size_t(pos >> 4) * nkv + h) * 16 * size_t(words);
}
__host__ __device__ inline int strata_positions(int cap) { return (cap + 31) / 32 * 32; }
__host__ __device__ inline int strata_span(int pos) { return (pos + 15) / 16 * 16; }
// How many pairs of positions a row's strata attention refines: one in `denom` of a row's pairs, at
// least `least`; every pair when that is all of them; none when denom is 0.  The row's own position decides it.
struct StrataSel {
  int denom = 128, least = 32;
  int recent = 0;  // and the pairs of the last `recent` positions, whatever their scores
  int most = 0;    // at most this many pairs (0: no bound), so that the refinement stops growing with the context
  __host__ __device__ static int pairs(int positions) { return positions / 2 + (positions & 1); }
  // A full quota of recent pairs leaves no place for a scored pair (the MTP window).
  bool recent_only() const { return denom > 0 && most > 0 && most == pairs(recent) && least <= most; }
  __host__ __device__ int recent_pairs(int total) const {
    const int ng = pairs(total), r = pairs(recent);
    return r < ng ? r : ng;
  }
  __host__ __device__ int groups(int total) const {
    if (denom == 0 || total == 0) return 0;  // no pairs refined
    const int ng = pairs(total), r = recent_pairs(total), sampled = 1 + (ng - 1) / denom;
    int k = sampled < ng - r ? sampled + r : ng;
    if (most > 0 && k > most) k = most;
    return k > least ? (k < ng ? k : ng) : (least < ng ? least : ng);
  }
};
// A sequence's scratch for strata attention: the scan's log-sum-exp per pair and query head ([rows][nh][lse_stride],
// binary16) and the pairs the selection picked ([rows][nkv][sel_stride], their number [rows][nkv]).
// The correction's slices: a row's selected positions split among this many blocks a KV head, their changes summed in
// order after ([rows][nkv][kStrataSlices][G][2 hd + 1] floats in `delta`).
constexpr int kStrataSlices = 16;
struct StrataScratch {
  __half* lse = nullptr;  // null for recent_only(): the selected pairs need no scores or candidate lists
  int* sel = nullptr;
  int* sel_n = nullptr;
  // The selection in two steps: each chunk's best pairs and their keys ([rows][nkv][chunks][sel_stride],
  // their number [rows][nkv][chunks]), and them in one list ([rows][nkv][chunks][sel_stride]).
  int* cand = nullptr;
  uint32_t* cand_key = nullptr;
  int* cand_n = nullptr;
  int* gathered = nullptr;
  uint32_t* gathered_key = nullptr;
  int chunks = 1;
  float* delta = nullptr;
  float* qr = nullptr;  // the rows' query vectors in the copy's basis ([rows][nh][hd])
  int lse_stride = 0, sel_stride = 0;
  StrataSel how;
};
// A round's draft tree (docs/design/tree-verification.md): row r (0 the anchor, r >= 1 draft r - 1) at depth[r],
// and the row of its ancestor at every depth, path[r][d] for d <= depth[r] (path[r][depth[r]] = r). n = 0: a chain
// (depth r, path d), which every kernel then runs as it runs a chain. accept_tree() writes the accepted path's rows
// taken[0 .. n_taken]; the compactions move them to rows 0 .. n_taken, so what runs after sees a chain.
// Up to FLYX_MAX_WIDE_ROWS rows (a wide round's).
struct TreeRows {
  int n;
  int depth[FLYX_MAX_WIDE_ROWS];
  unsigned char path[FLYX_MAX_WIDE_ROWS][FLYX_MAX_WIDE_ROWS];
  int taken[FLYX_MAX_WIDE_ROWS];
  int n_taken;
};
__host__ __device__ inline bool tree_on(const TreeRows* t) { return t != nullptr && t->n > 0; }
__host__ __device__ inline int tree_depth(const TreeRows* t, int r) { return tree_on(t) ? t->depth[r] : r; }
// The row of row r's ancestor at depth d.
__host__ __device__ inline int tree_row(const TreeRows* t, int r, int d) { return tree_on(t) ? t->path[r][d] : d; }
// Row r's key j of a round whose anchor is at position `base`: the slot it is read from (its ancestor's).
__host__ __device__ inline int tree_slot(const TreeRows* t, int r, int base, int j) {
  return tree_on(t) && j >= base ? base + t->path[r][j - base] : j;
}

// The keys [ks, ke) that split `split` of `splits` gives a row attending to `total` keys: as many splits as its
// keys fill, kAttnMinSplitKeys or more each, starting on 32-key boundaries.  It depends on the row's own position
// only, so a row gets the same splits whatever rows a round has.
constexpr int kAttnMinSplitKeys = 128;
constexpr int kAttnCoresSplits = 32;  // the most of a grid's splits the float32 kernels use (the rest: empty partials)
__host__ __device__ inline int attn_split_used(int total, int splits) {  // the splits [0, used) a row's keys fill
  const int fill = (total + kAttnMinSplitKeys - 1) / kAttnMinSplitKeys;
  return splits < (fill > 1 ? fill : 1) ? splits : (fill > 1 ? fill : 1);
}
__host__ __device__ inline void attn_split_range(int total, int split, int splits, int* ks, int* ke) {
  const int used = attn_split_used(total, splits);
  const int chunk = ((total + used - 1) / used + 31) & ~31;
  *ks = total < split * chunk ? total : split * chunk;
  *ke = total < *ks + chunk ? total : *ks + chunk;
}
// The head dims every attention kernel takes (split, grouped, flash; binary16 or FP8 caches): 32, 64, 128 and 256.
// The backend refuses blocks with others at load (regions/plan.cpp).
bool attention_head_dim(int hd);
// Whether decode, verification and prefill can read an FP8 cache of this shape.
bool kv_fp8_supported(const KernelOptions& opt, int nh, int nkv, int hd);

// DV's canvas pass (ops/canvas/ref.h, plan: the CONTEXT pass after a round): its row w sits at position
// P->start + 1 + w, after the round's anchor, and keeps its K/V in the cache's scratch slot s0 + w past the text.  Its
// keys are the text [0, P->start] (the anchor's included) and the canvas's own slots, every one of them or, when
// causal, those up to its own (attention: rows + `extra`'s keys as for a chain, positions after P->start read from
// s0 on); never a round's tree slots.  s0 < 0: no canvas (every other pass).
struct CanvasKeys {
  int s0 = -1;
};

// 3D Multimodal RoPE (Qwen3.5/3.8 interleaved or contiguous sections): active when any sec[i] > 0.
// During multimodal prefill, pos_3d holds [rows, 3] device ints (t, h, w); during decode or pure-text
// passes, pos_3d is null and every axis uses scalar pos + P->rope_delta.
struct MRoPE {
  int sec[3] = {0, 0, 0};
  int interleaved = 0;
  const int32_t* pos_3d = nullptr;
  __host__ __device__ bool active() const { return sec[0] > 0 || sec[1] > 0 || sec[2] > 0; }
};

// Attention input prep for one projection output row block:
//   qkv rows: [q heads (nh*hd) | other (skip) | k (nkv*hd) | v (nkv*hd)] at the given column offsets.
//   q -> rmsnorm(q_norm) -> rope -> q_out [rows, nh, hd]
//   k -> rmsnorm(k_norm) -> rope -> Kc[pos], v -> Vc[pos]    (binary16 caches [cap, nkv, hd])
// A null q_norm or k_norm: no norm, the heads go to the rotary as projected (Llama).
// pos = base + pos_add + r, base from P (PosSel); rows at a negative position
// write no K/V.  Skipped entirely if skip_if_noop and P->noop.
void attn_prep(const float* qkv, int ld, int q_off, int k_off, int v_off, const float* q_norm, const float* k_norm,
               float norm_offset, float eps, double theta, int rot_dim, int nh, int nkv, int hd, int rows,
               const DevParams* P, int pos_sel, bool skip_if_noop, float* q_out, kv16* Kc, kv16* Vc,
               cudaStream_t st, int pos_add = 0);
// strata_base: 16-aligned base slot subtracted from Strata output offsets (requires kv.ring == 0 and slot >= strata_base)
// so prefill can stage host-pinned Strata writes contiguously from offset 0 before flushing via cudaMemcpyAsync.
// tree: rotary positions by depth, K/V by row.  canvas (no tree): row r rotated to P->start + 1 + pos_add + r (pos_sel
// unused), its K/V written to slot canvas.s0 + pos_add + r.
void attn_prep(const float* qkv, int ld, int q_off, int k_off, int v_off, const float* q_norm, const float* k_norm,
               float norm_offset, float eps, double theta, int rot_dim, int nh, int nkv, int hd, int rows,
               const DevParams* P, int pos_sel, bool skip_if_noop, float* q_out, KvRef kv, cudaStream_t st,
               int pos_add = 0, const TreeRows* tree = nullptr, int strata_base = 0, CanvasKeys canvas = {},
               MRoPE mrope = {});
// The MTP layer's positions a draft step reads: its first src.sink positions and its last `window`,
// [max(0, q - window + 1), q] (q = P->position + pos_add), of src (where src.slot_of says), K, V and their scales,
// copied in order to dst's first rows; *P_out = *P but for the position at which an attention at pos_add reads
// exactly those rows.
void kv_window(const KvRef& src, const KvRef& dst, int nkv, int hd, int window, const DevParams* P, DevParams* P_out,
               int pos_add, cudaStream_t st);
// *P_out = *P with the position pos_add on: a kernel that reads P->position + row then reads a draft step's row.
void params_at(const DevParams* P, DevParams* P_out, int pos_add, cudaStream_t st);

// Attention of rows queries (q [rows, nh, hd]) against cached keys.
//   causal:  row r sees keys [0, base + r]           (base = P->position + pos_add)
//   else:    every row sees keys [0, base + extra)
//   window > 0: of those, row r only the keys j with |base + r - j| < window (a drafter's sliding window).
// out[r, h*hd + d] = softmax(q k^T * scale) v, times sigmoid(gate[r, h*hd + d]) if gate.
void attention(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int rows, bool causal,
               int extra, float scale, const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out,
               float* partial, int splits, const Launch& L, int pos_add = 0, int window = 0);
// tree (rows <= FLYX_MAX_WIDE_ROWS, causal): row r at position base + its depth, its round's keys its ancestors'
// (TreeRows); never the prefill kernels.
// canvas (no tree, no window): a canvas pass's rows (CanvasKeys; base = P->start + 1 + pos_add), on the float32 kernels
// at every length (attention_split's, never the tensor-core ones).
void attention(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra, float scale,
               const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, float* partial, int splits,
               const Launch& L, int pos_add = 0, int window = 0, const TreeRows* tree = nullptr,
               CanvasKeys canvas = {});
size_t attention_partial_floats(int rows, int nh, int hd, int splits);
// The rotary factors theta^(-2 i / rot_dim), i < rot_dim / 2, of a (theta, rot_dim) pair in device memory, made once
// (rope_register, at load: it waits for the device) by the kernels' own pow, so attn_prep and draft_ctx_kv take them
// instead of a float64 pow a launch, with the same bits; rope_table: the pair's table, nullptr if never registered.
void rope_register(double theta, int rot_dim);
const double* rope_table(double theta, int rot_dim);
// The two implementations attention() chooses from: float32 split-K over the
// keys (any shape; decode, verification, drafting), and tensor cores for
// causal prefill of more than 16 rows (head dims 32/64/128/256; q and the
// probabilities rounded to binary16 like the caches, fp32 accumulation).
// attention_flash returns false when the head dim is not supported.
// Windowed split-K attention across staging windows [k_lo, k_hi) (k_lo a multiple of kFlashWindowAlign): carries the
// in-flight split's warp state across window boundaries (at most one split per (row, head) straddles any boundary) so
// pageable short prefills (rows <= FLYX_MAX_ROWS) reproduce single-pass attention_split bit-for-bit.  A launch has two
// boundaries: the split that began before k_lo reads what the launch before left, and the split that goes on past
// k_hi leaves its own.  Where a split ends inside the window these are two blocks of one launch, running in no
// order, so what is read (`carry`) and what is written (`state`) are two buffers, which the caller swaps from one
// launch to the next.
__host__ __device__ constexpr int kSplitStateFloats(int hd) { return 4 * (hd + 40); }
struct SplitWindow {
  int k_lo = 0, k_hi = INT_MAX;
  float* state = nullptr;        // [rows][nh][kSplitStateFloats(hd)]: written by the split that goes on past k_hi
  const float* carry = nullptr;  // the same of the launch before, read by the split that began before k_lo (k_lo > 0)
  bool first = true, last = true;
};
void attention_split(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int rows, bool causal,
                     int extra, float scale, const DevParams* P, const float* gate, int ld_gate, float* out,
                     int ld_out, float* partial, int splits, const Launch& L, int pos_add = 0, int window = 0);
void attention_split(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra, float scale,
                     const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, float* partial,
                     int splits, const Launch& L, int pos_add = 0, int window = 0, const TreeRows* tree = nullptr,
                     CanvasKeys canvas = {}, const SplitWindow& sw = SplitWindow{});
bool attention_flash(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int rows, float scale,
                     const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, const Launch& L,
                     int pos_add = 0);
// attention_split's partials on tensor cores (binary16 and FP8 caches; KernelOptions::rows_mma) for the rows of `from`
// keys or more (the others' partials untouched): q and p as binary16 parts, the same splits; false (nothing launched):
// a shape it does not take.
bool attention_rows_mma(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra,
                        float scale, const DevParams* P, float* partial, int splits, const Launch& L, int pos_add,
                        int from, const TreeRows* tree = nullptr, const SplitWindow& sw = SplitWindow{});
// A prefill's attention a window of keys at a time: the keys [k_lo, k_hi) of each row's (k_lo a multiple
// of kFlashWindowAlign; kv need hold only those), carrying each (row, head)'s softmax state (its maximum, its four
// lanes' sums, its deferred FP8 value-scale exponent bit-cast in slot 5 for grouped FP8 kernels, and its output,
// kFlashStateFloats(hd) floats) in `state` from one window to the next: first, the state
// starts empty; last, the output is written, else the state.  The tiles are the ones a single pass takes, so the
// windows give its bits.  The default: every key, one pass.
constexpr int kFlashWindowAlign = 64;
__host__ __device__ constexpr int kFlashStateFloats(int hd) { return hd + 8; }
struct FlashWindow {
  int k_lo = 0, k_hi = INT_MAX;
  float* state = nullptr;  // [rows][nh][kFlashStateFloats(hd)]
  bool first = true, last = true;
  bool history = false;  // Every key in the window precedes the first query.
};
bool attention_flash(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, float scale, const DevParams* P,
                     const float* gate, int ld_gate, float* out, int ld_out, const Launch& L, int pos_add = 0,
                     const FlashWindow& win = FlashWindow{});
// The FP8 24:4, HD 256 grouped kernel with 2, 3 or 6 query heads per block (tuner and bit tests).
bool attention_flash_gqa_cfg(int hb, const float* q, KvRef kv, int nh, int nkv, int hd, int rows, float scale,
                             const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, const Launch& L,
                             int pos_add = 0, const FlashWindow& win = FlashWindow{});
// Strata attention for a round's causal rows at P->position + r: the low-bit copy's attention for
// every position, each row's StrataScratch::how pairs most attended to replaced by the cache's own values (kv.k / kv.v,
// FP8 or binary16, in device or host memory).  The shapes strata_supported names; `partial` as attention_split's.
// Only the rows at position `from` or past it are written (the others are exact attention's).
void attention_strata(const float* q, KvRef kv, const StrataScratch& scratch, int nh, int nkv, int hd, int rows,
                      float scale, const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out,
                      float* partial, int slots, const Launch& L, int from = 0, cudaEvent_t second_ready = nullptr,
                      const DevParams* P_hot = nullptr);
bool strata_supported(int nh, int nkv, int hd, int bits);
int strata_words(int hd, int bits);  // 32-bit words of codes a position and head (StrataLayout)

// out = silu(a) * b, a and b are column blocks of the same rows.
void swiglu(const float* gu, int ld, int I, float* out, int ld_out, int rows, cudaStream_t st);

// ---- gated delta net ------------------------------------------------------------
struct GdnDims {
  int nk, nv, dk, dv, ck;   // heads, head dims, conv kernel
  int tiled = 0;            // how value heads map to key heads (flyx_gdn_head_order)
  __host__ __device__ int C() const { return 2 * nk * dk + nv * dv; }
  __host__ __device__ int rep() const { return nv / nk; }  // value heads to a key head
  // The key head value head h reads; value head i (0 .. rep() - 1) of key head kh; whether h is a key head's first.
  __host__ __device__ int key_head(int h) const { return tiled ? h % nk : h / rep(); }
  __host__ __device__ int value_head(int kh, int i) const { return tiled ? i * nk + kh : kh * rep() + i; }
  __host__ __device__ bool first_of_key(int h) const { return tiled ? h < nk : h % rep() == 0; }
};

// Selects a recurrent-state slot on the device.  sel >= kSlotExplicit picks slot sel - kSlotExplicit.
constexpr int kSlotExplicit = 16;
// Replay-input slots per GDN layer: one per in-flight round (FLYX_MAX_IN_FLIGHT = 2).
constexpr int kGdnReplaySlots = FLYX_MAX_IN_FLIGHT;

// One layer's per-row inputs of the recurrence, kept so a round can be folded
// or rewound later: pre-conv rows (rounds only; slot_stride > 0), conv value
// output (v), normalised q and k (written into by conv, normalised in place by
// gdn_prep), beta, g.  Prefill uses a single set (slot_stride == 0) where the
// leading region holds gdn_chunk's per-chunk intermediates (T, A, gam, beta).
constexpr int kGdnChunkSize = 32;
struct GdnBufs {
  float* base = nullptr;
  size_t slot_stride = 0;  // floats between slots (0 = prefill)
  int rows_cap = 0;
  GdnDims d{};
  __host__ __device__ size_t o_v() const {
    if (slot_stride) return size_t(rows_cap) * d.C();
    const size_t max_chunks = size_t(rows_cap + 2 * kGdnChunkSize - 2) / kGdnChunkSize;
    return max_chunks * size_t(d.nv) * (2 * kGdnChunkSize * kGdnChunkSize + 2 * kGdnChunkSize);
  }
  __host__ __device__ size_t o_qn() const { return o_v() + size_t(rows_cap) * d.nv * d.dv; }
  __host__ __device__ size_t o_kn() const { return o_qn() + size_t(rows_cap) * d.nk * d.dk; }
  __host__ __device__ size_t o_beta() const { return o_kn() + size_t(rows_cap) * d.nk * d.dk; }
  __host__ __device__ size_t o_g() const { return o_beta() + size_t(rows_cap) * d.nv; }
  __host__ __device__ size_t floats() const { return o_g() + size_t(rows_cap) * d.nv; }
  static GdnBufs prefill(int rows, GdnDims dims) { return {nullptr, 0, rows, dims}; }
  static GdnBufs round(int rows, GdnDims dims) {
    GdnBufs b{nullptr, 1, rows, dims};
    b.slot_stride = (b.floats() + 3) / 4 * 4;
    return b;
  }
};

// Causal conv (+SiLU) of `rows` rows of `mixed` against the window of slot win_sel.
// Writes the conv output to bufs.conv (slot bufs_sel) and, for rounds (slot_stride > 0),
// copies the rows to bufs.mixed for a later fold.
// win_out_sel >= 0: write the advanced window to that slot (win_sel itself in
// prefill; kSlotNext in plain decoding), not if skip (a SkipWhen) says so.
// win: [4][ck-1][C].
// snap_sel >= 0: the window before row snap_row (0 <= snap_row <= rows) also goes to that slot.
void gdn_conv(const float* mixed, int ld_mixed, const float* conv_w, float* win, size_t win_stride, int win_sel,
              int win_out_sel, int skip, GdnBufs bufs, int bufs_sel, const DevParams* P, int rows, cudaStream_t st,
              int snap_sel = -1, int snap_row = -1, const TreeRows* tree = nullptr);  // tree: a row's last inputs
                                                                                       // are its ancestors' 
// The same on a prefill pass's bf16 in-projection rows (gemm_staged_bf16): no tree, the window not skipped.
void gdn_conv(const bf16* mixed, int ld_mixed, const float* conv_w, float* win, size_t win_stride, int win_sel,
              int win_out_sel, GdnBufs bufs, int bufs_sel, const DevParams* P, int rows, cudaStream_t st,
              int snap_sel = -1, int snap_row = -1);
// q,k l2-normalised (q also * 1/sqrt(dk)); beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
void gdn_prep(GdnBufs bufs, int bufs_sel, const float* b, const float* a, int ld_ba, const float* A_log,
              const float* dt_bias, const DevParams* P, int rows, cudaStream_t st);
void gdn_prep(GdnBufs bufs, int bufs_sel, const bf16* b, const bf16* a, int ld_ba, const float* A_log,
              const float* dt_bias, const DevParams* P, int rows, cudaStream_t st);  // bf16 rows (gdn_conv's)
// Recurrence over rows.  S: [4][nv][dv][dk].  Reads slot in_sel, writes slot out_sel (skip if < 0).
// out [rows, nv*dv] may be null (fold).  rows = rows_dev ? *rows_dev + 1 : rows.
// skip: a SkipWhen (true = kSkipIfNoop).
void gdn_recur(float* S, size_t s_stride, int in_sel, int out_sel, GdnBufs bufs, int bufs_sel, const DevParams* P,
               float* out, int rows, const int* rows_dev, int skip, cudaStream_t st);
// A tree round's recurrence: row r from its parent's state, every row's arithmetic gdn_recur's (where a row's parent
// is not the row before it, the slot's state again and the parent's path replayed); no state written.  False: shapes
// it does not take (gdn_recur's lanes kernel's: key dims 16 to 128, an aligned state; up to 16 rows).
bool gdn_recur_tree(const float* S, size_t s_stride, int in_sel, GdnBufs bufs, int bufs_sel, const DevParams* P,
                    float* out, int rows, const TreeRows* tree, cudaStream_t st);
// Prefill: the recurrence over `rows` rows in chunks of gdn_chunk_align()
// positions aligned to absolute positions (pos0: the position of row 0), on
// bf16 tensor cores (each product's inputs split into a high part and its
// remainder, three products) with a float32 state: reads and writes slot `sel` of S,
// writes out [rows, nv*dv].  A row's results do not depend on where the pass
// ends, and a pass that starts at a multiple of the alignment gives the bits
// of a longer pass over the same positions.  bufs: the prefill set
// (slot_stride 0); its pre-conv area holds the chunks' intermediates.  false:
// shapes it does not take (then nothing ran).
// snap_sel >= 0: the state at row snap_row (a chunk boundary inside the pass)
// also goes to that slot.
bool gdn_chunk(float* S, size_t s_stride, int sel, GdnBufs bufs, const DevParams* P, int pos0, float* out, int rows,
               cudaStream_t st, int snap_sel = -1, int snap_row = -1);
int gdn_chunk_align();
// Whether gdn_chunk takes a GDN block of these dimensions (on this device).
bool gdn_chunk_takes(const GdnDims& d);
// Whether this GPU prefills the GDN in chunks at all: the GPUs llama.cpp runs its own chunked kernel on (sm_86, 89,
// 120, 121).  Elsewhere (A100, H100, B200) llama.cpp keeps the float32 recurrence, and so does this backend: never
// coarser than llama.cpp on the same GPU.
bool gdn_chunk_device();
// The rows a prefill set needs at least, so that the intermediates fit for every pass (Qwen3.5 shapes).
int gdn_chunk_rows();
// Window after `rows` (or *rows_dev + 1) rows of bufs.mixed: slot in_sel -> slot out_sel.
void gdn_conv_fold(float* win, size_t win_stride, int in_sel, int out_sel, GdnBufs bufs, int bufs_sel,
                   const DevParams* P, int rows, const int* rows_dev, bool skip_if_noop, cudaStream_t st);
// A round's commit for every gated delta net of a sequence in two launches, not two a layer: gdn_recur (no output,
// kSkipIfNoop) and gdn_conv_fold (skip if noop) of each, the same arithmetic, with each layer's state, window and
// replay buffers from `layers` (device memory, n of them, alike: the same dims, slot strides and rows) and bufs' dims
// and slot stride. False (nothing launched): key dims the lanes kernels do not take.
struct GdnLayer {
  float* S;
  float* win;
  float* replay;
};
bool gdn_fold_layers(const GdnLayer* layers, int n, size_t s_stride, size_t w_stride, GdnBufs bufs, int in_sel,
                     int out_sel, int bufs_sel, const DevParams* P, const int* rows_dev, cudaStream_t st);
// After a tree round's accept: its replay rows (slot kSlotLast) of the accepted path to rows 0 .. n_taken, for the
// commit and rewinds.  layers non-null: all n of them (bufs: their dims and strides), else bufs' own.
void gdn_compact(const GdnLayer* layers, int n, GdnBufs bufs, const DevParams* P, const TreeRows* tree,
                 cudaStream_t st);
// y = rmsnorm(o per head, w) * silu(z)   (o: [rows, nv*dv])
void gdn_gated_norm(const float* o, const float* z, int ld_z, const float* w, float eps, float* y, int rows,
                    GdnDims d, cudaStream_t st);
// A round's whole mixer between the two projections in one kernel (rows <=
// 16): gdn_conv, gdn_prep, gdn_recur with outputs and gdn_gated_norm on the
// proj rows [mixed | z | b | a], with the same bits.  Reads window, state and
// buffers of slot kSlotCur; next_sel >= 0 (plain decoding, one row): writes the
// advanced window and state to that slot unless P->stopped.  y null: the
// output goes to the split buffer (for gemm_presplit; q8: in 8 bits, for block
// weights that take 8-bit inputs).  false: shapes it does not take (then
// nothing ran).
// True when gdn_round() supports this block's dimensions, row count, alignment, and input rounding.
bool gdn_round_fits(const GdnDims& d, int rows, const float* S, size_t s_stride, const Launch& L, bool q8 = false);
// Fused GDN round (conv + prep + recur + gated norm + staging). When y is null, writes into
// split_out_slice(L, total_rows > 0 ? total_rows : rows, row_offset, vd, q8) for gemm_presplit.
bool gdn_round(const float* proj, int ld, const float* conv_w, float* win, size_t win_stride, float* S,
               size_t s_stride, int next_sel, const float* A_log, const float* dt_bias, const float* gw, float eps,
               GdnBufs bufs, const DevParams* P, int rows, float* y, const Launch& L, bool q8 = false,
               int total_rows = 0, int row_offset = 0);

// ---- head, drafter, round bookkeeping -----------------------------------------------
// out[r] = argmax_v logits[r, v] (lowest index on ties).
void argmax_rows(const float* logits, int ld, int V, int* out, int rows, float* scratch, int* iscratch,
                 cudaStream_t st);
// Token choice per row as defined in ops/sampling/sampling.h, with the parameters in G
// (greedy when G->temperature <= 0).  Row r's token goes to position
// P->position + base_add + r.  When G->mask, row 0 chooses among the tokens set
// in mask (ceil(V / 32) words).  cz/ci: scratch of rows * V floats / ints.
void choose_rows(const float* logits, int ld, int V, int rows, const DevParams* P, int base_add, const DevGen* G,
                 const uint32_t* mask, float* cz, int* ci, float* scratch, int* iscratch, int* out, cudaStream_t st,
                 const TreeRows* tree = nullptr, int tree_from = 0);
// tree: row r samples position P->position + base_add + depth - tree_from, its depth the tree's row tree_from + r's
// (a wide round's 16 rows at a time from row tree_from, base_add = tree_from + 1).
// The penalties of ops/sampling/sampling.h, in place and before choose_rows:
// logits[r][v] -= penalty(count of v), for the tokens the reply has.  drafts
// (a round; null for the pass that chooses a reply's first token): row r also
// counts drafts[0..r), the tokens it was given after the reply's.  Does
// nothing when G has no penalty.
void penalize_rows(float* logits, int ld, int V, int rows, const DevGen* G, Reply reply, const int* drafts,
                   cudaStream_t st, const TreeRows* tree = nullptr, int tree_from = 0);
// tree: a row's drafts before it: its ancestors.  tree_from: logits row r is the round's row tree_from + r (a wide
// round's 16 rows at a time).
// Top-k per row, sorted high to low (lowest index on ties).  k <= 16.  With part buffers of kTopkParts * 16 entries a
// row for up to part_rows rows: each row's slices in parallel, then their candidates merged (the same top-k: the
// order is total); else one block a row.
constexpr int kTopkParts = 24;
void topk_rows(const float* logits, int ld, int V, int k, int* idx, float* val, int rows, cudaStream_t st,
               float* part_v = nullptr, int* part_i = nullptr, int part_rows = 0, float* prob = nullptr,
               float* margin = nullptr);
// DFlash2 grouped dynamic causal conv; rows are one block (position = row index).
//   y (=|+=) conv(x) with coef[t][c] = base[t][c] + delta[r][t][c / group]
void grouped_conv(const float* x, int ldx, const float* delta, int ld_delta, const float* base, int H, int taps,
                  int group, float* y, int ldy, int rows, bool accumulate, cudaStream_t st);
// Greedy candidate walk; writes `width` drafts.  Uses oracle tokens when P->oracle_len > 0.
void selector_walk(const DevParams* P, const int* cand, const float* unary, int k, const float* hid, int rank,
                   const bf16* pred, const bf16* succ, int width, const int* oracle, int* drafts, cudaStream_t st);
// The candidates as a tree of `width` drafts, best first by the selector's scores (ops/draft/ref.h), candidates at
// `depths` positions: drafts and the round's TreeRows.  The oracle tokens as a chain when P->oracle_len > 0.
void selector_tree(const DevParams* P, const int* cand, const float* unary, int k, const float* hid, int rank,
                   const bf16* pred, const bf16* succ, int width, int depths, const int* oracle, int* drafts,
                   TreeRows* tree, cudaStream_t st);
// DSpine (kernels/dspine.cu, ops/dspine/ref.h; regions/layers.cpp's dspine_block runs them between its GEMMs).
// The bridge's write after a layer: per row rms = sqrt(mean(x^2)) and xn = x / (rms + eps) (the read GEMM's input);
// the read rows' codes in place (row 0: sqrt(C) codebook[P->anchor]; else sqrt(C) u / (|u| + eps)); the messages
// from the gate | pred | prev rows [rows][3R] (m_0 = 0, m_r = sigmoid(q_r + s_{r-1} + bias) * v_{r-1}; g, when not
// null, each row's q); x_r += rms_r d_r for r >= 1 (before, when not null, x as it was).
void dspine_normalize(const float* x, int H, float eps, int rows, float* xn, float* rms, cudaStream_t st);
void dspine_codes(float* codes, int C, const bf16* codebook, const DevParams* P, float eps, int rows,
                  cudaStream_t st);
void dspine_message(const float* gates, int R, const float* bias, int rows, float* msg, float* g, cudaStream_t st);
void dspine_apply(float* x, const float* d, const float* rms, int H, int rows, float* before, cudaStream_t st);
// The walk.  For the n = rows * k candidates cand (row t's at t k): heads[i] = head[cand[i]] * nw (W null: heads holds
// the rows already, embed_blocks') and base[i] = <before_{t+1}, heads[i]>; the senders' codes (0: the anchor's, i >= 1
// candidate i - 1's); for each sender the best candidate of the row after it (pick; sv the codes' pred | prev
// [senders][2R], tg the candidates' heads through the last write [n][R], g and rms the rows' before it); the chain of
// `width` drafts from the picks (the test's oracle tokens when P->oracle_len > 0).
void dspine_heads(const int* cand, const bf16* W, const float* nw, const float* before, int H, int k, int n,
                  float* heads, float* base, cudaStream_t st);
void dspine_senders(const DevParams* P, const int* cand, const bf16* codebook, int C, int senders, float* codes,
                    cudaStream_t st);
void dspine_scores(const float* g, const float* sv, const float* bias, const float* tg, const float* base,
                   const float* rms, int R, int k, int senders, int* pick, cudaStream_t st);
void dspine_walk(const DevParams* P, const int* cand, const int* pick, int k, int width, const int* oracle,
                 int* drafts, cudaStream_t st);
// Copy rows of x into feats at column block `slot` (taps).
void tap_copy(const float* x, int H, float* feats, int ld_feats, int slot, int rows, cudaStream_t st);
void tap_copy(const bf16* x, int H, float* feats, int ld_feats, int slot, int rows, cudaStream_t st);
// Draft context K/V: kv rows [rows, layers * 2 * nkv * hd] = per layer [k | v]; k -> rmsnorm -> rope -> dK[layer][pos].
void draft_ctx_kv(const float* kv, int ld, const float* k_norms, float eps, double theta, int layers, int nkv, int hd,
                  int rows, const DevParams* P, int pos_sel, bool skip_if_noop, kv16* dK, kv16* dV,
                  size_t layer_stride, cudaStream_t st);
// Compare drafts with choices, advance P, write the ring entry.  With a
// penalty in G the tokens emitted are counted in `reply`.
void accept(DevParams* P, const DevGen* G, const int* drafts, const int* choice, int width,
            flyx_round_result* ring, Reply reply, cudaStream_t st, const float* draft_prob = nullptr,
            const float* draft_margin = nullptr);
// A round's accept by ops::accept_tree: the rows of a tree (tree non-null; writes tree->taken), else a chain; with
// the request's relaxed rules when logits and stats (row_stats) are given, else exact matching.  The same ring entry
// and position bookkeeping as accept().  ld 0: logits holds each draft's logit in its parent's row (child_logits: a
// wide round's, whose head keeps 16 rows of logits at a time).  The tokens emitted stop at the context's end
// (G->max_ctx): a round past 16 rows started where flyx_room holds leaves the sequence there at most, so the round in
// flight behind it writes inside the K/V cap.
void accept_tree(DevParams* P, const DevGen* G, const int* drafts, const int* choice, int width,
                 flyx_round_result* ring, Reply reply, TreeRows* tree, const float* logits, int ld,
                 const float* stats, cudaStream_t st, const float* draft_prob = nullptr,
                 const float* draft_margin = nullptr);
// Each row's distribution for the relaxed acceptance rules: stats[3 r ..] = {max logit, Z, entropy} (k_row_stats);
// nothing when the request has none.
void row_stats(const float* logits, int ld, int V, int rows, const DevGen* G, float* stats, cudaStream_t st);
// z[c - 1] = logits[(parent(c) - from) * ld + drafts[c - 1]] for every draft c (1..width) whose parent row (tree,
// else c - 1) is one of the rows from .. from + rows - 1 the logits hold: what accept_tree's relaxed rules read of a
// wide round's logits, gathered as its head goes 16 rows at a time.
void child_logits(const float* logits, int ld, int from, int rows, const int* drafts, int width, const TreeRows* tree,
                  float* z, cudaStream_t st);
// After a tree round's accept, the accepted path's rows to rows 0 .. n_taken, so what runs after sees a chain: the
// round's K/V slots P->start + taken[d] -> P->start + d of every cache in the table (bytes a position: row_bytes;
// e4m3 caches' scales too), and per-row buffers rows taken[d] -> d (`words` 32-bit words a row, stride ld words).
// Nothing for a round that was a no-op.
struct KvRows {
  int row_bytes, nkv;
  KvRef where;     // where a position's row and scales are (KvRef::slot_of, kscales)
};
void compact_kv(const KvRows* table, int n, int most_row_bytes, const DevParams* P, const TreeRows* tree,
                cudaStream_t st);
void compact_rows(void* rows, size_t ld_words, int words, const DevParams* P, const TreeRows* tree, cudaStream_t st);

// ---- DV drafter (bifill-dv's: docs/prior-art.md; ops/canvas/ref.h, ops/draft/ref.h) --------------------------------
// A round's tree (DRAFT, before the round's pass) grows from what the canvas pass after the round before left (a pass
// of N rows of its own, CanvasKeys), slid by the tokens that round committed.  Every slot keeps its kDvTop best.
constexpr int kDvTop = 16;
constexpr int kDvMaxDepths = 63, kDvMaxBeam = 32, kDvMaxWidth = 63;  // the trees' limits (depths <= N - 1)
// What a canvas pass leaves for its N slots (canvas_stats): ids [N][16], the slot's 16 best tokens high to low (ties
// to the lower id); p [N][16], their probabilities under the whole vocabulary; lse [N], the log-sum-exp of its logits;
// mass [N], the 16's sum of p; zn [N][hidden], its expected embedding sum_c p_c E[ids_c]; hid [N][rank], what the
// selector reads of its final-normed hidden state h: H h / sqrt(rank).  part_i, part_v [N][kDvParts][16]:
// canvas_stats' scratch (each part of the vocabulary's best 16).
constexpr int kDvParts = 16;
struct DvStats {
  int* ids;
  float *p, *lse, *mass, *zn, *hid;
  int* part_i = nullptr;
  float* part_v = nullptr;
};
// The canvas as the next round reads it (dv_slide writes it, ops::slide): cv [N] each slot's next input (the mask
// token: its guess was not kept) and z [N][hidden] its residual context; cand [N][16], logp [N][16] = log max(p, 1e-9),
// cov [N] = log max(sum_c exp(logp_c), 1e-6) (DYN's cov term, ops::slot_coverage), live [N] (0: past the pass's slots,
// no candidates) and hid [N][rank] for the tree; next_start [1], the position of the round they are for.  A round at
// another position (-1 after a prefill, rewind or restore) drafts DV's first tree, every draft the mask token under
// the anchor, and its canvas pass starts from mask inputs and z = 0.
struct DvCanvas {
  int* cv;
  float* z;
  int* cand;
  float *logp, *cov;
  int* live;
  float* hid;
  int* next_start;
};
// The selector's token tables, once at load: TA[t] = A E[t] and TB[t] = B E[t], [vocab][rank] float32, from its
// float32 A, B [rank][hidden] and the target's bf16 embedding E [vocab][hidden]; each value one fma after another in
// the order of hidden, from 0 (the same whatever the tiling).
void dv_selector_tables(const float* A, const float* B, int rank, const bf16* E, int vocab, int hidden, float* TA,
                        float* TB, cudaStream_t st);
// A canvas pass's input rows (ops::canvas_input), slots rows of x [slots][hidden]:
//   x_w = (cv_w == mask ? mask_emb : E[cv_w]) + res_out (silu(g) * u),   [u | g] = res_in z_w,
// from the canvas's cv and z; in a round not at c.next_start (P->start: the pass runs after the round) every slot
// mask_emb.  The products are a round's gemm() (the 1..16-row kernels' arithmetic: a slot's bits whatever the slots);
// silu(g) * u and the sum are this file's kernels.  ug: scratch [slots][2 hidden]; res_in [2 hidden][hidden] bf16,
// res_out [hidden][hidden] bf16, mask_emb [hidden] bf16.
void canvas_input(const DevParams* P, const DvCanvas& c, int slots, int mask, const bf16* E, const bf16* mask_emb,
                  const bf16* res_in, const bf16* res_out, int hidden, float* ug, float* x, const Launch& L);
// What a canvas pass leaves (ops::canvas_stats, DvStats) from its slots' logits [slots][ld] (vocab of them), their
// final-normed rows h [slots][ldh] and the selector's H [rank][hidden] (float32): the top 16 the best of the best 16 of
// kDvParts parts of the vocabulary (the row's own); sums in float32 in a fixed order (p = exp((l - max) - log sum
// exp(l - max)), zn's and hid's terms in the order of c and of hidden).
void canvas_stats(const float* logits, int ld, int vocab, const bf16* E, const float* h, int ldh, int hidden,
                  const float* H, int rank, int slots, const DvStats& s, cudaStream_t st);
// The canvas after a round (ops::slide): slot w takes the pass's slot w + n, n = P->position - P->start the tokens the
// round committed; its guess becomes its next input from probability `fill` on; next_start = P->position.  Nothing
// after a no-op round (P->noop).
void dv_slide(const DevParams* P, const DvStats& s, int slots, int hidden, int rank, float fill, int mask,
              const DvCanvas& c, cudaStream_t st);
// A round graph's conditional node around its canvas pass: set to 1, or 0 after a no-op round (P->noop), whose pass
// would leave the canvas as it was (dv_slide).
void dv_live(cudaGraphConditionalHandle h, const DevParams* P, cudaStream_t st);
// A round's selector table (ops::selector_table, float32) over the first `depths` slots: table[(d * 16 + i) * 16 + b],
// candidate b of slot d after candidate i of slot d - 1 (slot 0: after P->anchor, its 16 rows the same) from the
// selector's tables TA, TB [vocab][rank]: S = (sum_j TA[prev]_j hid_j TB[cand_b]_j + alpha logp_b) / temp, table =
// S - log sum_b exp S.  Nothing in a round not at c.next_start.
void dv_selector_table(const DevParams* P, const DvCanvas& c, const float* TA, const float* TB, int rank, float alpha,
                       float temp, int depths, float* table, cudaStream_t st);
// DV's trees of `width` drafts on that table, depths <= kDvMaxDepths: dv_tree_beam is ops::beam_tree<float> (DYN
// width:beam[:cov], beam <= kDvMaxBeam, with c.cov as the bonus when cov), dv_tree_pool ops::pool_tree<float> (ST
// width/ranks/1, ranks < 16), bit for bit, with c.cand and c.live.  They write drafts [width], parents [width] (null:
// not written; -1 the anchor) and the round's TreeRows (null: not written; width + 1 rows) as selector_tree does.
// A round not at c.next_start drafts DV's first tree (width mask tokens under the anchor); with the test's drafts
// (P->oracle_len), a chain of them.  One block each; a pool that does not fit a block's shared memory (ST 31/7/1:
// 26 KB) is refused, as are shapes past the limits.
// Whether they take a tree of `kind` (flyx_dv_tree) of `width` drafts over `depths` slots, `fan` its beam or its
// pool's ranks (no: why): lowering asks, so that a round they would refuse mid-generation is no plan.
bool dv_tree_takes(int kind, int width, int fan, int depths, std::string* why);
void dv_tree_beam(const DevParams* P, const DvCanvas& c, const float* table, int width, int beam, bool cov, int depths,
                  int mask, const int* oracle, int* drafts, int* parents, TreeRows* tree, cudaStream_t st);
void dv_tree_pool(const DevParams* P, const DvCanvas& c, const float* table, int width, int ranks, int depths,
                  int mask, const int* oracle, int* drafts, int* parents, TreeRows* tree, cudaStream_t st);

// ---- MTP drafter ------------------------------------------------------------------
constexpr int kHring = 64;  // target hidden kept for the last kHring positions (MTP input after a rollback)

// MTP input rows: cat[r] = [rmsnorm(E[tokens[r]]) * (offset + we) | rmsnorm(h_r) * (offset + wh)] with
// h_0 = h0 if given, else hring[base - 1]; h_r = hn[r - 1] for r > 0 (base from P, PosSel).
// emb (E null): the rows' embeddings already looked up, [rows, H] (embed_blocks).
void mtp_gather(const DevParams* P, int pos_sel, bool skip_if_noop, const float* h0, const float* hring,
                const float* hn, int ldh, const int* tokens, const bf16* E, const float* emb, const float* we,
                const float* wh, float offset, float eps, int H, int rows, float* cat, cudaStream_t st);
// hring[(base + r) % kHring] = hn[r] for the last kHring of `rows` rows.
void hring_put(const DevParams* P, int pos_sel, bool skip_if_noop, const float* hn, int ldh, int rows, float* hring,
               int H, cudaStream_t st);
// Test hook: mtok[step + 1] = oracle[P->position + 1 + step] when P->oracle_len > 0.
void mtp_oracle(const DevParams* P, const int* oracle, int step, int* mtok, cudaStream_t st);
// A round's given drafts: drafts[i] = oracle[P->position + 1 + i] (0 past P->oracle_len), i < width.
// dst = *P with its position add further on: a wide round's rows a..a+15 run as the round's rows 0..15 of a pass
// at P->position + a (attention_strata takes up to FLYX_MAX_ROWS rows).
void shift_params(const DevParams* P, DevParams* dst, int add, cudaStream_t st);
void given_drafts(const DevParams* P, const int* oracle, int width, int* drafts, cudaStream_t st);
void copy_ints(const int* src, int* dst, int n, cudaStream_t st);
// K/V in tiers: copies positions of a layer's K and V caches (`row` bytes a position, a multiple of
// 16) from host memory to their views, the staging an attention then reads.  With at = P->position + pos_add, the
// attention of rows [at, at + span) reads positions [from, min(cap, at + span)); kv_stage copies them all
// (kStageAll), those written before the pass (kStageBefore: [from, at), ahead of the attention on a stream of their
// own), or the pass's rows (kStagePass: [max(from, at), min(cap, at + span))).  `blocks` blocks of 256 threads: the
// bus, not the GPU, sets the speed, and a few leave the rest of it to the layers beside.  A row of `per_row`
// positions (strata's copy: 16) is copied whole when it holds one of them. Scales (FP8):
// the same positions' nkv floats from ks to vks and vs to vvs (indexed by position), when given.
enum KvStagePart { kStageAll, kStageBefore, kStagePass };
struct KvStageScales {
  const float *ks = nullptr, *vs = nullptr;
  float *vks = nullptr, *vvs = nullptr;
  int nkv = 0;
};
void kv_stage(const void* k, void* vk, const void* v, void* vv, size_t row, int from, int cap, const DevParams* P,
              int pos_add, int span, KvStagePart part, int blocks, cudaStream_t st, int per_row = 1,
              const KvStageScales& scales = KvStageScales{});
// Copy positions written by the current pass from a pageable layer's GPU tail ring (slot_of) into a linear staging
// window (vk/vv, scales).
void kv_stage_ring(const void* k, void* vk, const void* v, void* vv, size_t row, int sink, int ring_base, int ring,
                   int from, int cap, const DevParams* P, int pos_add, int span, KvStagePart part, int blocks,
                   cudaStream_t st, const KvStageScales& scales = KvStageScales{});
// At round end (before copy_ring): copy each pageable layer's newly committed rows [pos - n_tokens, pos) from its GPU
// tail ring into its mapped pinned drain slot (P->seq_no % kRing).
void kv_drain_pageable(const PageDrainLayer* layers, int n_layers, const DevParams* P, const flyx_round_result* ring_dev,
                       int max_rows, cudaStream_t st);

// After a prefill chunk: advance the position; optionally set the anchor from
// choice[0] and emit it (counted in `reply` when G has a penalty).
void prefill_done(DevParams* P, int n, const int* choice, int sample, flyx_round_result* ring, const DevGen* G,
                  Reply reply, cudaStream_t st);
// The W4A4 prefill GEMM on the float32 rows a fused producer left (Scratch::af32) into bf16 rows y16 [M][ldy], each
// value the float32 product (accumulate: its sum with y16's value) rounded once (prefill rows vLLM and SGLang hold in
// bf16): false where these weights do not take NVFP4 inputs here (the caller runs gemm_tc_staged into float32).
bool gemm_staged_bf16(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, bf16* y16, int ldy, int M,
                      const Launch& L, bool accumulate = false);
// gemm_on_glu into bf16 rows y16 (accumulate: the float32 sum with y16's value, rounded once): false where the last
// gemm_glu_staged did not run W4A4 or these weights do not take NVFP4 inputs here.
bool gemm_on_glu_bf16(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, bf16* y16, int ldy, int M,
                      bool accumulate, const Launch& L);

// Fused producers (rows <= 16): write their output already split into the
// skinny GEMMs' scratch (stride = the output width), so the next GEMM skips
// its split kernel; gemm_presplit / gemm_fp4_presplit then read it.  Same
// bits as the unfused pair.  attention() with out == nullptr does the same.
// q8: for GGML block weights whose inputs are 8 bits, the output in 8 bits
// where the block kernels read it (blocks_stage's work, the same bits; the
// width a multiple of 32, of 64 for SwiGLU); gemm_presplit then stages nothing.
void rmsnorm_split(const float* x, int ldx, const float* w, float offset, float eps, int rows, int n, const Launch& L,
                   bool q8 = false);
// The same of bf16 rows (a prefill pass's bf16 residual), more than 16 rows.
void rmsnorm_split(const bf16* x, int ldx, const float* w, float offset, float eps, int rows, int n, const Launch& L,
                   bool q8 = false);
void swiglu_split(const float* gu, int ld, int I, int rows, const Launch& L, bool q8 = false);
void gdn_gated_norm_split(const float* o, const float* z, int ld_z, const float* w, float eps, int rows, GdnDims d,
                          const Launch& L, bool q8 = false);
void gdn_gated_norm_split(const float* o, const bf16* z, int ld_z, const float* w, float eps, int rows, GdnDims d,
                          const Launch& L, bool q8 = false);  // z: bf16 rows (gdn_conv's bf16 overload)
void gemm_presplit(const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
// A low-rank adapter's first product beside a linear (backend_abi.h flyx_lora): u [rows][ldu] = scale (x a^T), a
// [n, K] bf16 (the a's of the linears that read x, stacked), on the rows the linear read: x through gemm(), or, x
// null, the input a fused producer left in the scratch through gemm_presplit(), the same bits (F).  The second
// product, y += u b^T, is gemm() on u.  The scale is one float32 product after the sums (none for scale 1).
void lora_down(const float* x, int ldx, const bf16* a, int n, int K, float scale, float* u, int ldu, int rows,
               const Launch& L);
void gemm_fp4_presplit(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy, int rows,
                       bool accumulate, const Launch& L);
// A prefill MLP's gate|up GEMM with SwiGLU in its epilogue (rows > 16, I % 8 == 0): the
// bf16 rows a fused producer left in the scratch times W [2 I, K] (gate rows, then up;
// bf16 W, or NVFP4 Q/S/RS) -> scratch glu [rows][I] = bf16(silu(g) u).  Same bits as the
// GEMM into float32 followed by swiglu_split; W4A4 under KernelOptions::prefill_fp4 as
// gemm_fp4_presplit.  cfg: a k_gemm_tc configuration (tests), -1 the default.  Then
// gemm_on_glu: the next GEMM (the down projection) on those rows.
void gemm_glu_staged(const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int I, int K, int rows,
                     const Launch& L, int cfg = -1);
void gemm_on_glu(const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy,
                 int rows, bool accumulate, const Launch& L);

// ---- W4A4 prefill GEMM (sm_120a block-scaled FP4 tensor cores; KernelOptions::prefill_fp4) ----
// Whether this build has it and the device runs it (compute capability 12.0).
bool gemm_w4a4_supported();
// float32 rows x [M][ldx] (K % 128 == 0) -> NVFP4, one rounding: q [M][K/2], block scales s [M][K/16], inv [M] =
// 1 / g (device.cuh: kNvfp4RowRange, nvfp4_block_scale, e2m1_nearest).
void quantize_rows_nvfp4(const float* x, int ldx, int M, int K, uint8_t* q, uint8_t* s, float* inv, cudaStream_t st);
// The same from bf16 rows whose largest |value| per row is amax [M] (the SwiGLU rows gemm_w4a4 leaves): one read,
// the codes quantize_rows_nvfp4 gives on the rows as float32.
void quantize_rows_nvfp4_bf16(const bf16* x, int ldx, int M, int K, const float* amax, uint8_t* q, uint8_t* s,
                              float* inv, cudaStream_t st);
// y [M, N] (=|+=) dequant(A) dequant(W)^T on NVFP4 rows A and NVFP4 weights (Q, S, RS).
// cfg: a tile configuration (0: 128 x 128, 4 stages; 1: 128 x 256, 3; 2: 128 x 128, 2; 3: 128 x 128, 2 stages of 256
// values of K, where K % 256 == 0, else 2; 4: 256 x 128 on a warp-specialized TMA pipeline, where K % 256 == 0, else
// 3's rule), all the same bits; -1: 4 where K % 256 == 0, else 2.
// glu non-null: W is a gate|up weight [N = 2 I, K], glu [M][I] = bf16(silu(g) u) and glu_amax [M] each row's
// largest |stored value| (gemm_glu_staged; quantize_rows_nvfp4_bf16 rounds them for the down projection, as vLLM's
// and SGLang's bf16 SwiGLU rows are rounded), y unused.  y16 non-null (a plain product): the product, or with
// accumulate the float32 sum of y16's value and the product, rounded once to bf16 into y16 [M][ldy] instead of y
// (prefill rows vLLM and SGLang hold in bf16: projections, the residual).
void gemm_w4a4(const uint8_t* Aq, const uint8_t* As, const float* Ar, const uint8_t* Q, const uint8_t* S,
               const float* RS, int N, int K, float* y, int ldy, int M, bool accumulate, cudaStream_t st,
               int cfg = -1, bf16* glu = nullptr, float* glu_amax = nullptr, bf16* y16 = nullptr);

// ---- Device-side intra-round confidence stopping (CUDA conditional graphs) ----
struct DevStopState {
  float survival;
  int32_t decision;
};
struct StopCondHandles {
  unsigned long long h[FLYX_MAX_STOP_BUCKETS];
};
void stop_decide_step(const float* prob, const float* margin, const flyx_stop_policy* pol, DevStopState* state,
                      int bucket, int prev_w, int cur_w, unsigned long long h_next_draft, unsigned long long h_ver,
                      cudaStream_t st);
void stop_decide_block(const float* prob, const float* margin, const flyx_stop_policy* pol, DevStopState* state,
                       StopCondHandles h_ver, cudaStream_t st);

// ---- Vision encoder & multimodal prefill kernels (vision.cu) ----
// Standard LayerNorm: y = (x - mean) * rsqrt(var + eps) * (w ? w : 1) + (b ? b : 0) per row.
void layernorm(const float* x, int ldx, const float* w, const float* b, float eps, float* y, int ldy, int rows, int n,
               cudaStream_t st);
// y[r, c] += b[c] for r < rows, c < cols.
void add_bias(float* y, int ldy, const float* b, int rows, int cols, cudaStream_t st);
// x[r, c] += scale ? y[r, c] * scale[c] : y[r, c] (plain residual or LayerScale).
void vision_residual_add(float* x, const float* y, const float* scale, int rows, int cols, cudaStream_t st);
// In-place activation (flyx_vact_kind: GELU_TANH, GELU, SILU, QUICK_GELU) on n elements.
void vact_apply(float* x, size_t n, int act, cudaStream_t st);
// Gated activation: out[i] = vact(gate[i], act) * up[i] on n elements.
void vact_gated(const float* gate, const float* up, float* out, size_t n, int act, cudaStream_t st);
// Zero columns [valid_cols, ld) in every row of x [rows, ld] (for padded quantized K dimensions).
void vision_zero_tail(float* x, int rows, int valid_cols, int ld, cudaStream_t st);
// Float32 GEMM: y[r, n] (=|+=) sum_k x[r, k] * W[n, k], arbitrary K (patch Conv2d/3d stem with F32 weights).
void gemm_f32(const float* x, int ldx, const float* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
              cudaStream_t st);
// Add fixed position embedding table pos [pos_rows, H] modulo pos_rows to x [rows, H].
void vision_pos_embed_add(float* x, const float* pos, int pos_rows, int H, int rows, cudaStream_t st);
// Add bilinearly interpolated 2D position embedding from [pos_h, pos_w, H] using per-row coords and grid dims.
void vision_pos_embed_bilinear_add(float* x, const float* pos, int pos_h, int pos_w, int H, const int32_t* coords_h,
                                   const int32_t* coords_w, const int32_t* row_gh, const int32_t* row_gw, int rows,
                                   cudaStream_t st);
// 2D axial Vision RoPE in-place on Q and K heads inside qkv [rows, ld] at offsets q_off, k_off.
void vision_rope_2d(float* qkv, int ld, int q_off, int k_off, int nh, int hd, double theta, const int32_t* coords_h,
                    const int32_t* coords_w, int rows, cudaStream_t st);
// Bidirectional block-diagonal vision attention (arbitrary hd <= 256, float32 online softmax) over cu_seqlens segments.
void vision_attention(const float* qkv, int ld, int q_off, int k_off, int v_off, int nh, int hd,
                      const int32_t* cu_seqlens, int n_segs, int rows, float scale, float* out, int ld_out,
                      cudaStream_t st);
// Average-pool contiguous groups of `group` rows: out[b, c] = mean_{g < group} x[b * group + g, c].
void vision_avg_pool_blocks(const float* x, int n_blocks, int group, int hidden, float* out, cudaStream_t st);
// Overwrite selected rows embed_rows[i] of x (float32) or x16 (bf16) with embeds[i * H .. (i + 1) * H).
void embed_scatter(float* x, bf16* x16, int H, const float* embeds, const int32_t* embed_rows, int n_embeds,
                   cudaStream_t st);

}  // namespace flyx::cuda
