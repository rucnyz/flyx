// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The CUDA backend's internals, shared by its host sources:
//   backend.cpp         the ABI table, probe, open (options), close
//   blocks.h the model as blocks and the state table
//   loader/bind.cpp     the description's tables -> blocks and state, weights laid out for the kernels
//   loader/weights.cpp  weights onto the device, the NVFP4 copies drafting reads
//   runtime/state.cpp   sequence state and row buffers (sizes, layout), the shared prefill buffers
//   runtime/runner.cpp  prefill passes, rounds (graphs, the token ring), waiting, rollback, checkpoints
//   runtime/graphs.cpp  what the round graphs take of the device's memory, a session's room for them
//   regions/            what each node runs on its block (layers.cpp), a GEMM on a weight (linear.cpp),
//                       plans: which blocks the kernels take, lower() and running a plan (plan.cpp)
// The kernels they launch are declared in kernels/kernels.cuh (built by nvcc into a static library).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backends/cuda/blocks.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/runtime/vision_disk.h"
#include "flyx/backend_abi.h"
#include "ops/dtype/ref.h"

namespace flyx::cuda {

// ---- errors ----------------------------------------------------------------------------------
// The message last_error() returns (per thread).
extern thread_local std::string g_error;
struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void ck(cudaError_t e, const char* what);
#define CK(x) ::flyx::cuda::ck((x), #x)
// The model's device memory (flyx_bmodel::alloc) and the prefill buffers: cudaMalloc that never takes more than the
// GPU has free.  Windows' driver lets an allocation past the free memory succeed in host memory, read over the bus
// every round (Qwen3.5-4B NVFP4 beside other programs holding 4.7 of 8 GB: 12.5 tok/s plain against 132.4); this
// throws OutOfDeviceMemory instead (and so does a refusal by the driver).  device_room: the same check alone, for
// memory the driver maps (a VMM slab), throwing where device_malloc would.
struct OutOfDeviceMemory : CudaError {
  using CudaError::CudaError;
};
void device_room(size_t bytes);
void* device_malloc(size_t bytes);
// Records m for last_error(); returns -1.
int fail(const std::string& m);
// fail() for a call given no sequence (a session whose seq_new failed or was never made): an error, not a crash.
int no_sequence(const char* call);

// A side stream and its events (Launch::side), made beside the stream it forks from and gone with it.
inline void create_side(SideStream* s) {
  CK(cudaStreamCreateWithFlags(&s->st, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&s->fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&s->join, cudaEventDisableTiming));
}
inline void destroy_side(SideStream& s) {
  if (s.fork) cudaEventDestroy(s.fork);
  if (s.join) cudaEventDestroy(s.join);
  if (s.st) cudaStreamDestroy(s.st);
}

// ---- options ---------------------------------------------------------------------------------
// The CUDA backend's options (flyx_open_options::options, "key=value,..."):
//   profile=1             time every node of every pass, printed when the session ends (no graphs)
//   graphs=0              launch a round's kernels one by one instead of replaying its graph
//   dv_conditional=0      a DV round's canvas pass inline in the round's graph, not the body of a conditional node that
//                         skips it after a no-op round (measurement and tests: what the conditional node takes)
//   fusion=0              the generic implementation of every region, no fused overrides
//   gdn_round=0           a GDN layer's round as four kernels, not one
//   stacks=0              every run of one block type in a stack its own round launch (gemm_blocks_staged_runs)
//   round_glu=0           a round's SwiGLU in a kernel of its own, not in the gate|up GEMM (block weights, 8 bits)
//   side_stream=0         a round's narrow GEMMs of a stack after its wide ones, not beside them on a second stream
//   attention=split|head  prefill attention in float32 / no KV-head grouping (and no FP8 K/V)
//   gemm=rows             every GEMM on the CUDA-core row path
//   pdl=0|1 programmatic dependent launch off / on (default: the GPU's table)
//   head_cert=0 greedy choices read the whole head, not the certified head
//   wait=spin|poll|auto   wait for the device by spinning (lowest latency, one core busy), by polling with short sleeps
//                         (the default on Linux), or (auto, the default on Windows) spinning while the last waits for a
//                         round's result took under 1.5 ms (a sleep ends ~0.5 ms late), polling past that (runner.cpp)
//   draft_weights=exact   drafts read the bf16 head, MTP layer and DFlash2 drafter, not NVFP4 or Q8_0 copies
//   draft_kv=16           the MTP layer's K/V cache in binary16, not FP8 with the target's (when they are FP8)
//   blocks=bf16           GGML block weights (GGUF) decoded to bf16 at load instead of kept packed
//   embed=gpu an embedding table no head reads on the GPU, not in host memory
//   tuning=defaults every kernel by the defaults' rules, not this GPU's tuning table
//   tuning=NAME the tuning table of that name (rtx4090...), whatever the GPU
//   log_shapes=1          print every GEMM shape once
//   stage_ahead=0         K/V in host memory copied to the device right before each attention, not beside the
//                         layers before it; stage_blocks=N: the blocks that copy beside them;
//                         stage_window=N: with strata or auto, a staging's positions (16384)
//   stage_dma=0           use staging kernels for host Strata prefixes and historical prefill KV windows
//   strata_copy_slots=2|3 device staging slots for host Strata copies with stage_ahead=1 (default 2)
//   strata_compact=1       remove the structural padding from hosted 256d three-bit Strata copies
//   mtp_window=N          past 8 N positions of context, the MTP layer's draft steps read its last N positions
//                         exactly (4096; 0: its whole cache on the device, every position exact) and the others
//                         through a copy of mtp_copy=BITS bits (2; 0, or heads strata attention is not compiled
//                         for: none, then its first mtp_sink=N exactly, 64);
//                         mtp_ring=0: its cache a slot a position, not a ring (tests)
//   strata_denom=N strata attention refines one pair of positions in N a row (default 16 with
//                         a 2-bit copy, 64 with more; 0: none, the copy alone), at least strata_least=N pairs (32),
//                         plus the last strata_recent=N positions, and at most strata_most=N pairs (256; 0: no
//                         bound); strata_scan=cores, strata_grid=normal: measurement
//   strata_from=N         --kv-method auto: rounds read the copy from position N on (default: the GPU's part of
//                         the K/V, i.e. 0 tokens of exact host overhang)
//   act_group=64|128|256  measurement: 8-bit inputs with one scale this many values (bench/quality/kld.cpp; with
//                         fusion=0 and round_glu=0, whose producers round their own outputs per 32)
//   act_fp4=1|2           measurement: inputs first rounded as NVFP4 (1) or MXFP8 (2) activations would be
enum class WaitMode { poll, spin, automatic };  // option wait=poll|spin|auto
struct BackendOptions {
  KernelOptions kernels;  // the ones launches consult
  bool profile = false, graphs = true, fusion = true, exact_drafts = false;
  bool dv_conditional = true;  // option dv_conditional=0 clears it: DV's canvas pass inline in a round's graph
  bool stage_dma = true;
  // How a thread waits for the device (runner.cpp; on WSL2 every CUDA wait spins a core, so the default polls).  On
  // Windows a sleep ends on a high-resolution timer, about 0.5 ms after it is asked to, so a round's result that the
  // last waits say comes within 1.5 ms (kSpinBelowUs) is spun for (auto).
#if defined(_WIN32)
  WaitMode wait = WaitMode::automatic;
#else
  WaitMode wait = WaitMode::poll;
#endif
  bool tuning_defaults = false;  // tuning=defaults
  std::string tuning_table;      // tuning=NAME
  bool pdl_given = false;        // option pdl was given: the GPU table's default does not apply
  bool full_draft_head = false;  // draft_head=full: draft logits from the whole (NVFP4) head, not draft_pick
  bool draft_kv16 = false;       // draft_kv=16: the MTP layer's cache in binary16 beside the target's FP8
  bool packed_blocks = true;     // blocks=bf16 clears it
  int embed = 0;                 // embed=auto (0: host memory unless the GPU has room, open_model), host (1), gpu (2)
  int vision = FLYX_VISION_DEV_DISK;         // vision=disk (0: temp file, 0 VRAM/RAM), host (1), gpu (2), auto (3)
  int vision_quant = FLYX_VISION_QUANT_Q8;   // vision_quant=q8 (0), exact (1), nvfp4 (2)
  int wide = 0;                  // wide=auto (0: wide lookup rounds where the GPU has room, open_model), 0 (1), 1 (2)
  int strata_copy_slots = 2;
  bool strata_compact = false;  // pack the two shared words into the 24 private words of a hosted 256d 3-bit copy
};
bool parse_options(const char* text, BackendOptions* o, std::string* err);

// runtime/device_stats.cpp: NVML figures of the device probe() chose.
void remember_device(int device);
int device_stats(flyx_device_stats* out);
// The device's free and total memory, as the budget (memory()) and device_malloc read them.  On Windows NVML's free
// memory (nvidia-smi's): cudaMemGetInfo's there is the video memory manager's budget for the process, 0.65 GB or more
// below it once 5 of 8 GB are held, and 0 while 0.65 GB are free (device_stats.cpp).  false: no figure.
bool gpu_memory(size_t* free, size_t* total);
// What this process holds of that device as the system counts it (flyx_memory_info.process): the models, their
// sequences, and the CUDA context with its kernels' modules.  0 where the system does not tell a process's share.
size_t gpu_process_memory();
// Windows: device memory device_malloc keeps free.  Past what its video memory manager keeps free (250 MiB for 32 MiB
// allocations, up to 560 MiB for 256 MiB to 1 GiB ones: device_stats.cpp), allocations land in host memory.
constexpr size_t kPagedFloor = 768ull << 20;
std::string describe(const BackendOptions& o);

// ---- weights ---------------------------------------------------------------------------------
uint16_t float_to_bf16(float f);
// A GGML NVFP4 tensor ([N, K], K % 64 == 0) as the NVFP4 planes: q [N][K / 2] (two e2m1 codes a byte, low nibble
// first), s [N][K / 16] (an e4m3 scale per 16); its row scale is 1. order (optional): the stored row of row r.
void ggml_nvfp4_planes(const flyx_tensor_view* v, const std::function<size_t(size_t)>& order, uint8_t* q,
                       uint8_t* s);
// y (+)= x W^T for a weight in either format; gemm_presplit: x is in the launch's scratch already.
void gemm(const float* x, int ldx, const Lin& W, int N, int K, float* y, int ldy, int rows, bool accumulate,
          const Launch& L);
void gemm_presplit(const Lin& W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
// A prefill MLP (rows > 16): gate|up with SwiGLU in the GEMM's epilogue (the
// scratch's bf16 rows in, its glu rows out), then a GEMM on those rows.
void gemm_glu_presplit(const Lin& W, int I, int K, int rows, const Launch& L);
void gemm_on_glu(const Lin& W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);

// Buffers sized by the rows of a pass.  Each session has its own for rounds
// (FLYX_MAX_ROWS rows); the model has one set for prefill chunks, shared by
// all sessions (one pass runs at a time), so a large chunk costs memory once.
struct RowBufs {
  int *tokens = nullptr, *choice = nullptr;
  float *x = nullptr, *h = nullptr, *proj = nullptr, *act = nullptr, *gdn_o = nullptr, *qbuf = nullptr,
        *part = nullptr, *attn_state = nullptr, *feats = nullptr, *cx = nullptr, *ckv = nullptr, *mcat = nullptr,
        *mx = nullptr, *mhn = nullptr;
  float* lora = nullptr;  // the target's adapters' first products (RowNeeds::lora a row; none without adapters)
  GdnBufs pre{};  // GDN prefill scratch (prefill set only)
  Scratch scratch{};  // the GEMM inputs the kernels stage
};

// Widths of the row buffers (floats per row), from the blocks (runtime/state.cpp).
struct RowNeeds {
  int proj = 0, act = 0, gdn_o = 0, q = 0;  // projections, activations, GDN output, attention queries
  int attn_state = 0;                       // prefill windowed flash attention softmax state (floats per row)
  int lora = 0;                             // the target's adapters' first products: q|k|v's (3 rank), gate|up's
  int feats = 0, cx = 0, ckv = 0;           // DFlash2 context: tapped features, fc output, K/V projections
  bool mtp = false;                         // MTP context rows (2 x hidden, hidden, hidden)
  bool glu = false;                         // non-packed MLP weights (prefill gemm_glu_staged)
  int k_max = 0;                            // largest GEMM K: the scratch staging GEMM inputs is sized by it
  bool q8 = false;  // weights with inputs in 8 bits: a prefill scratch holds their GEMMs' runs of K side by side
};

// option head_cert: head_certify's scratch (row norms, bound keys, surviving columns and their count), shared in shape
// by each sequence and by batch rounds.
struct CertScratch {
  float* hn = nullptr;
  int *key = nullptr, *cols = nullptr, *count = nullptr;
};

// Where the load's time went (open_model reports it with verbose): GGML blocks read from the file and laid out
// on the host, and copied to the device.
struct LoadTimes {
  double pack_s = 0, copy_s = 0;
  uint64_t file_bytes = 0;
};

// ---- what a round graph takes of the device's memory (runtime/graphs.cpp) -------------------------------------------
// The driver keeps every node's launch and every graph's own in pieces of 2 MiB that the process's graphs share, and
// keeps the pieces once the graphs are destroyed, for its next graphs.  What an instantiated graph takes at most, by
// what it holds (`cuda_bench graphs` measures each kind of node; RTX 5060 Ti, driver 610.88; a node's figure there has
// the graph's own 96 KiB in it): a kernel node kGraphKernelBytes (chains of 100, 400 and 1000: 3.5, 3.3 and 2.7 KiB a
// node; launched with programmatic dependent launch the same) while its parameters take kGraphParamBytes at most, and
// kGraphParamCost times each byte past them (4 KB of parameters: 11.5 KiB a node); a copy node kGraphCopyBytes (100
// between device buffers: 5.8 KiB a node; 100 to or from pinned or host-mapped memory take no more than the graph's
// own 96 KiB, as do 100 memsets); a memset node kGraphSetBytes; any other kGraphOtherBytes, and the graphs it holds
// (a conditional node's bodies: DV's canvas pass, regions/layers.cpp; a child graph) each a graph of its own, its nodes
// priced the same way (a conditional node with a body of 1, 100 and 400 kernels: 192, 256 and 1440 KiB a graph, 96 and
// 128 KiB more than a chain of 1 and 400); the graph itself kGraphBytes (96 KiB for a graph of one node).
constexpr size_t kGraphKernelBytes = 3584, kGraphParamBytes = 1024, kGraphParamCost = 3;
constexpr size_t kGraphCopyBytes = 7 << 10, kGraphSetBytes = 1 << 10, kGraphOtherBytes = 16 << 10;
constexpr size_t kGraphBytes = 128 << 10;
// A graph's nodes by kind (a conditional node's bodies' among them), its largest kernel parameters, and what it takes
// at most (graph_size).
struct GraphSize {
  size_t kernels = 0, copies = 0, sets = 0, others = 0, params_most = 0, bytes = 0;
  size_t params_unknown = 0;  // kernels whose parameters the driver did not say (priced at kGraphParamBytes)
  size_t bodies = 0;          // graphs its nodes hold (a conditional node's bodies), each priced as a graph
  size_t bodies_unknown = 0;  // conditional nodes whose bodies the driver did not give (priced as the node alone)
};
// A session's room holds a graph of every round plan lowered: what a greedy round of it takes on a session whose
// K/V are all on the GPU and read exactly (its first graph, the load's, or one captured on a session of
// kGraphCountCtx tokens), and what the session's other rounds of it take more, in kernel and copy nodes (measured on
// the tiny test model's rounds, -v):
//   kGraphHeadNodes      a plan counted at another head's graph than the one a round runs (heads differ by a few
//                        kernels: Qwen3-0.6B's greedy round 322, sampled with a penalty 318);
//   kMtpWindowNodes      an MTP draft step where the context holds more than 8 MTP windows (the window's gather: +3);
//   kTierNodes           a cache in tiers (its stagings: +2);
//   kStrataNodes         a low-bit copy the session's rounds read (+3; +6 with the copies in host memory or about
//                        auto's switch; +10 with both: 12, a fifth more), kStrataCopyNodes more copy nodes a copy in
//                        host memory (its stagings: +4).
// And a share of a batch graph (graph_keep: what the session's graphs may hold), and a kGraphPoolPart-th of that more,
// which they may not hold, for the driver's pool (graph_room: what seq_bytes counts): it keeps what the graphs a
// session gave up held and gives that to later graphs only where they fit.  In the memory tests, where sessions give up
// graphs for others again and again, the device lost 0.72-0.77 (K/V on the GPU) to 0.96-0.99 (K/V and copies in host
// memory) of what their graphs held at these prices: the part leaves room for a driver that takes a quarter more.
constexpr int kGraphCountCtx = 64;
constexpr size_t kGraphHeadNodes = 8, kMtpWindowNodes = 3, kTierNodes = 2, kStrataNodes = 12, kStrataCopyNodes = 4;
constexpr size_t kGraphPoolPart = 4;
struct PlanGraph {
  size_t bytes = 0;       // its counted graph's (graph_size); 0 where its round's capture failed
  int mtp_steps = 0;      // its MTP draft steps
  bool counted = false;
};
// A graph the session launched in its last kGraphRecent rounds stays when another needs room: that round runs without
// a graph instead, and the session tries no other capture for kGraphRecent rounds (rounds of kinds that alternate
// never take each other's place, round after round).
constexpr unsigned long long kGraphRecent = 8;

// Fixed-capacity graph cache key for a batch round (no heap allocation per launch).
struct BatchKey {
  int n = 0;
  bool cert = false;
  struct Entry {
    uintptr_t seq = 0;
    uintptr_t plan = 0;
    int flags = 0;
    bool operator!=(const Entry& o) const { return seq != o.seq || plan != o.plan || flags != o.flags; }
    bool operator<(const Entry& o) const {
      if (seq != o.seq) return seq < o.seq;
      if (plan != o.plan) return plan < o.plan;
      return flags < o.flags;
    }
  };
  Entry seqs[FLYX_MAX_ROWS]{};
  bool operator<(const BatchKey& o) const {
    if (n != o.n) return n < o.n;
    if (cert != o.cert) return cert < o.cert;
    for (int i = 0; i < n; ++i)
      if (seqs[i] != o.seqs[i]) return seqs[i] < o.seqs[i];
    return false;
  }
};

// Scratch buffers for batched DFlash2 drafting (2+ sequences sharing DFlash2 weight reads, up to FLYX_MAX_ROWS).
struct BatchDFlash {
  int* dtok = nullptr;
  float *dx = nullptr, *dh = nullptr, *dhc = nullptr, *dtmp = nullptr;
  float *dproj = nullptr, *dact = nullptr, *ddelta = nullptr, *dq = nullptr;
  int* cand = nullptr;
  float *unary = nullptr, *hid = nullptr;
  int ld_dproj = 0, ld_dact = 0;
};

// Rounds of several sequences (runtime/runner.cpp launch_batch, regions/batch.cpp): each sequence's
// rows after the one before it, at most FLYX_MAX_ROWS in all, borrowing the lead sequence's row/logits/cert buffers on
// the batch stream (every participating sequence is synchronised on ready_ev / done).  Its graphs are kept by the
// sequences and plans they ran, each in its sequences' rooms for graphs: an equal share of what it takes in each
// one's graph_held.
struct BatchRounds {
  static constexpr size_t kMaxGraphs = 64;        // LRU cap on cached batch graphs
  static constexpr uint32_t kGraphWarmHits = 8;   // capture only once a batch combo repeats (avoids capture stalls)
  static constexpr size_t kMaxHitEntries = 256;
  struct CachedGraph {
    cudaGraphExec_t exec = nullptr;
    uint64_t used = 0;   // `launched` at its last launch
    size_t share = 0;    // in each of its sequences' graph_held
    unsigned long long ran[FLYX_MAX_ROWS]{};  // each sequence's seq_no at its last launch (BatchKey::seqs' order)
  };
  cudaStream_t st = nullptr;
  SideStream side{};  // the round's narrow GEMMs of a stack beside its wide ones (Launch::side)
  BatchDFlash dflash{};     // allocated only when the model has a DFlashBlock with 2 * block <= FLYX_MAX_ROWS
  cudaEvent_t done = nullptr;  // the last batch round: the sequences' streams wait for it
  std::map<BatchKey, CachedGraph> graphs;
  std::map<BatchKey, uint32_t> hits;  // launches seen before graph capture
  bool graphs_ok = true;
  long launched = 0, captured = 0, rows = 0, sequences = 0;  // batches of two sequences and more (verbose report)
  std::mutex mu;
  ~BatchRounds() {
    for (auto& g : graphs) cudaGraphExecDestroy(g.second.exec);
    if (done) cudaEventDestroy(done);
    destroy_side(side);
    if (st) cudaStreamDestroy(st);
  }
};

// Seconds since an arbitrary start, for the load's report.
inline double load_clock() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// fn(a, b) for the rows [a, b) of n, the rows split among up to 16 threads (this one among them): laying out a
// tensor's blocks at load is the CPU's work, and the threads' reads of the file keep the disk busy.
void parallel_rows(size_t n, const std::function<void(size_t, size_t)>& fn);

// Physical VMM pieces backing the model's prefill workspace, shared with sessions' strata copy pools.
// Ref-counted so existing sessions keep valid mappings if a wider plan grows the slab.
struct VmmSlab {
  std::vector<unsigned long long> handles;
  size_t gran = 0, bytes = 0;
  ~VmmSlab();  // runtime/tiers.cpp
};

void* prefill_vmm_alloc(flyx_bmodel* m, size_t bytes, size_t* actual_bytes);
void prefill_vmm_free(flyx_bmodel* m);

}  // namespace flyx::cuda

struct flyx_bmodel {
  std::vector<void*> allocs;
  size_t bytes = 0;
  std::vector<void*> host_allocs;  // pinned host memory (host_alloc / host_pinned)
  size_t host_bytes = 0;
  size_t vision_dev_bytes = 0, vision_host_bytes = 0, vision_disk_bytes = 0;
  flyx::cuda::VisionDiskStore vision_disk;  // auto-deleted temp file when vision=disk (0 VRAM, 0 locked host RAM)
  // Tiny 2-slot pinned host ring (<= 16 MiB total) allocated lazily on first disk encode_vision() under prefill_mu.
  // Keeps 0 bytes locked at load/plan_memory/text-only time while avoiding per-image WDDM pin/unpin latency.
  void* vision_disk_stage = nullptr;
  size_t vision_disk_stage_bytes = 0;
  int verbose = 0;

  // The model as blocks (blocks.h), built by bind() at open: blocks[i] is the
  // description's block i, so program nodes name them by the same index.
  std::vector<flyx::cuda::Block> blocks;
  flyx::cuda::StateTable states;
  int hidden = 0, vocab = 0;  // the residual stream's width, the vocabulary
  int mtp_block = -1, dflash_block = -1;  // the drafters (at most one of each), -1: none
  int dv_block = -1;                      // DV (docs/design/dv.md), -1: none
  int head_block = -1;                    // the head they read
  flyx::cuda::RowNeeds need;

  int prefill_rows = 0;  // rows of the widest prefill plan lowered: the shared prefill buffers hold them
  int prefill_align = 1;  // gdn_chunk_align() when every GDN block takes the chunked prefill recurrence
  int res16 = -1;         // residual16(): whether the model's prefill passes may keep the residual in bf16 (-1: unknown)
  // The most rows a round may have: FLYX_MAX_WIDE_ROWS where lookup rounds may be wide (open_model), else
  // FLYX_MAX_ROWS.  Each sequence's round buffers hold this many rows.
  int wide_rows = FLYX_MAX_ROWS;
  flyx::cuda::RowBufs prefill_bufs;  // shared by every session's prefill
  std::vector<void*> prefill_allocs;
  std::shared_ptr<flyx::cuda::VmmSlab> prefill_slab;  // VMM backing when available (shared with strata pools)
  unsigned long long prefill_va = 0;
  size_t prefill_va_bytes = 0;
  void* prefill_zero_ptr = nullptr;     // scratch counter header re-zeroed after a strata session borrowed the slab
  size_t prefill_zero_bytes = 0;
  flyx_seq* prefill_owner = nullptr;    // session whose strata pool currently occupies prefill_slab (under prefill_mu)
  bool prefill_dirty = false;           // true when prefill_slab was written by a strata session since last prefill
  size_t prefill_bytes = 0;
  // Reusable device staging for multimodal prefill_ex (guarded by prefill_mu; allocated lazily on first VLM prefill).
  float* prefill_embeds = nullptr;
  size_t prefill_embeds_cap = 0;
  int32_t* prefill_embed_rows = nullptr;
  size_t prefill_embed_rows_cap = 0;
  int32_t* prefill_pos3d = nullptr;
  size_t prefill_pos3d_cap = 0;
  cudaEvent_t prefill_ev = nullptr;  // the last prefill pass that used prefill_bufs
  // Held from a pass's wait for prefill_ev to its record, and while the buffers are replaced: sessions on two
  // threads take them one pass after the other.
  std::mutex prefill_mu;                // lock order: prefill_mu before batch->mu
  bool kv_fp8 = false;   // the target's K/V caches in e4m3 (open option)
  flyx::cuda::BackendOptions opt;    // open()'s options; opt.kernels is what every launch for this model consults
  std::unique_ptr<flyx::cuda::BatchRounds> batch;  // rounds of several sequences
  // Round graphs: every round plan lowered (plan_free forgets it) and its graph's size, counted when a
  // session's room for graphs is first asked after it was lowered (graph_room, which seq_bytes counts).
  std::mutex graph_mu;
  std::map<const flyx_plan*, flyx::cuda::PlanGraph> round_graphs;

  ~flyx_bmodel() {
    if (batch && verbose && batch->launched)
      std::fprintf(stderr, "flyx-cuda: %ld rounds of several sequences (%.2f sequences, %.1f rows each), %ld graphs "
                   "captured\n", batch->launched, double(batch->sequences) / batch->launched,
                   double(batch->rows) / batch->launched, batch->captured);
    batch.reset();  // its graphs and stream before the memory they use
    if (prefill_ev) cudaEventDestroy(prefill_ev);
    if (prefill_embeds) cudaFree(prefill_embeds);
    if (prefill_embed_rows) cudaFree(prefill_embed_rows);
    if (prefill_pos3d) cudaFree(prefill_pos3d);
    flyx::cuda::prefill_vmm_free(this);
    for (void* p : prefill_allocs) cudaFree(p);
    for (void* p : allocs) cudaFree(p);
    for (void* p : host_allocs) cudaFreeHost(p);
    if (vision_disk_stage) cudaFreeHost(vision_disk_stage);
  }
  template <class T>
  T* alloc(size_t n) {
    void* p = flyx::cuda::device_malloc(n * sizeof(T));
    allocs.push_back(p);
    bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }
  // Host memory the kernels read in place over the bus (pinned, mapped), for a table the kernels only gather rows
  // of. The pointer returned is the kernels' (it can differ from the host's: write-combined memory
  // under WSL, which reads no faster here); uploads to it go through cudaMemcpyDefault.
  template <class T>
  T* host_alloc(size_t n) {
    void* p = nullptr;
    CK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped));
    host_allocs.push_back(p);
    host_bytes += n * sizeof(T);
    void* d = nullptr;
    CK(cudaHostGetDevicePointer(&d, p, 0));
    return static_cast<T*>(d);
  }
  // Page-locked host memory for async H2D DMA streaming (e.g. offloaded vision weights): raw host pointer,
  // 0 bytes of GPU VRAM, freed with host_allocs on model close.
  template <class T>
  T* host_pinned(size_t n) {
    if (n == 0) return nullptr;
    void* p = nullptr;
    CK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault));
    host_allocs.push_back(p);
    host_bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }
  // Gives back an allocation of n bytes made by alloc().
  void release(void* p, size_t n) {
    auto it = std::find(allocs.begin(), allocs.end(), p);
    if (it == allocs.end()) return;
    allocs.erase(it);
    bytes -= n;
    CK(cudaFree(p));
  }
};

namespace flyx::cuda {
struct Pass;

// ---- regions: the generic implementation of every node, and overrides --------------------
// lower() resolves each node to the most specific implementation that fits
// the plan (rows) and the model (shapes); option fusion=0 keeps the
// generic ones.  An override must give the same result as the generic
// implementation (here: bit for bit; the golden tests run both).
// A node runs on its block (null for nodes without one: tap, accept, commit).
using RegionFn = void (*)(const Pass&, const flyx_node&, const Block*);
struct RegionImpl {
  int kind;                               // flyx_node_kind, 0 = any
  const char* name;
  int max_rows;                           // applies to plans of at most this many rows
  bool (*fits)(const flyx_bmodel*, const Block&);  // shape conditions, may be null
  RegionFn fn;
};
}  // namespace flyx::cuda

// The execution plan: every node of the program resolved to an implementation.
struct flyx_plan {
  flyx_bmodel* m;
  int phase, width, max_rows;
  std::vector<flyx_node> nodes;
  std::vector<const flyx::cuda::Block*> block;      // per node: the block it runs (null: none)
  std::vector<const flyx::cuda::RegionImpl*> impl;  // per node
  // A round of a tree of given drafts (FLYX_GIVEN_TREE) or of its drafter's (FLYX_DRAFT_TREE): its rows by the
  // session's TreeRows, the accepted path compacted after the accept; tree_kv: the K/V caches its attention layers
  // write (flyx_seq::kv indices).  drafted: the drafter writes the TreeRows in the round, the launch does not.
  bool tree = false, drafted_tree = false, has_draft = false;
  std::vector<int> tree_kv;
  std::vector<int> checkpoints;  // sorted verify checkpoints for adaptive conditional round graph (empty: fixed width)
  bool adaptive() const { return checkpoints.size() > 1; }
};

namespace flyx::cuda {

// The first node a round runs after its drafting (its drafter's, or the drafts it is given).
inline size_t target_start(const flyx_plan* p) {
  size_t i = 0;
  while (i < p->nodes.size() && (p->nodes[i].kind == FLYX_NODE_DRAFT || p->nodes[i].kind == FLYX_NODE_GIVEN)) ++i;
  return i;
}

// Key splits of a round's attention (the partials' slots): the tensor-core kernel uses up to 64, the float32 ones up
// to kAttnCoresSplits (fewer at short contexts).
constexpr int kRoundSplits = 64;
// What a round's attention reads of a cache with a low-bit copy: the cache (every position), the
// copy (strata attention), or both, the copy's output replacing the rows at the switch or past it.
enum StrataPhase { kStrataNone, kStrataAll, kStrataBoth };
// FLYX_KV_AUTO's switch (KernelOptions::strata_from): this many positions past the device's part of the K/V.
// Exact attention stays on for every position that fits on the GPU (where it outran strata on RTX 5090 at every
// length); once a session spills into host memory, switching immediately (0 host overhang) avoids sizing the device
// staging window for a 32K exact host suffix and keeps long-context budgeting compact on tight 16 GB cards.
constexpr int kStrataHostExact = 0;
constexpr int kMaxPrefillRows = 4096;
constexpr int kPrefillSplits = 8;

}  // namespace flyx::cuda

struct flyx_seq {
  flyx_bmodel* m;
  int max_ctx = 0, cap = 0, rows_cap = 0;
  cudaStream_t st = nullptr;
  flyx::cuda::SideStream side{};  // a round's narrow GEMMs of a stack beside its wide ones (Launch::side)
  std::vector<void*> allocs;

  flyx::cuda::DevParams* P = nullptr;
  flyx::cuda::DevParams* params_host = nullptr;    // pinned staging for set_params
  flyx::cuda::DevGen* G = nullptr;
  flyx::cuda::DevGen* gen_host = nullptr;          // pinned
  flyx::cuda::DevGen gen_last{};
  bool gen_valid = false;
  uint32_t* mask = nullptr;            // the first chosen token's allowed set (G->mask), ceil(vocab / 32) words
  uint32_t* mask_host = nullptr;       // pinned staging
  int mask_words = 0;
  flyx::cuda::Reply reply{};                       // the reply's tokens as the penalties count them (set_reply, accept)
  flyx_round_result* ring_dev = nullptr;
  flyx_round_result* ring_host = nullptr;  // pinned mirror
  int* tok_host = nullptr;             // pinned staging for prefill tokens
  cudaEvent_t ev[flyx::cuda::kRing]{};

  // state, as the model's state table declares it
  std::vector<flyx::cuda::KvBuf> kv;
  std::vector<flyx::cuda::GdnBuf> gdn;
  // Every gated delta net's state, window and replay buffers on the device, for a round's commit in two launches
  // (gdn_fold_layers); null where the layers are not alike or their states not 16-byte aligned.
  flyx::cuda::GdnLayer* gdn_layers = nullptr;
  float* ring = nullptr;               // the hidden ring (MTP input), kHring x ring_width
  float* ring_snap = nullptr;          // 1-row hidden snapshot at pass_snap - 1 (preserved past kHring overwrites)
  float* ring_ckp[FLYX_CHECKPOINTS]{};  // host memory, as the GDN checkpoints (host_alloc)
  flyx::cuda::RowBufs own;              // this session's row buffers (rounds)
  struct Ckpt {
    bool valid = false;
    int pos = 0;
    int rope_delta = 0;
  } ckp[FLYX_CHECKPOINTS];

  // drafter scratch
  float* mg = nullptr;                 // MTP normalized draft hidden
  int* mtok = nullptr;                 // MTP: [anchor, d1, ..., dK]
  int* oracle = nullptr;
  int oracle_cap = 0;
  // gen.drafts: written where the oracle is read, through a ring of pinned
  // slots (FLYX_MAX_ROWS drafts + the oracle length each), so a launch
  // never waits for a copy still pending; with at most two rounds in flight a
  // slot is free again by the time the ring comes back to it.
  static constexpr int kDraftSlots = 4;
  int* drafts_host = nullptr;
  flyx::cuda::DevParams* P_shift = nullptr;  // a wide round's strata attention 16 rows at a time (shift_params)
  int drafts_slot = 0;
  int test_oracle_len = 0;             // set_oracle's length: restored after a round with gen.drafts
  bool drafts_given = false;           // the last launch set the oracle length for its drafts
  // Tree rounds: the round's tree (written by the launch through a ring of pinned slots, as the drafts; the accept
  // writes the path it took), the K/V caches the compaction moves (built at the first tree launch), and the rows'
  // distributions for relaxed acceptance.
  flyx::cuda::TreeRows* tree = nullptr;
  flyx::cuda::TreeRows* tree_host = nullptr;       // kDraftSlots slots
  int tree_slot = 0;
  flyx::cuda::KvRows* tree_kv = nullptr;
  int n_tree_kv = -1, tree_kv_bytes = 0;  // -1: not built
  float* accept_stats = nullptr;       // 3 floats a row
  float* draft_prob = nullptr;         // per-draft confidence p_max (FLYX_MAX_ROWS)
  float* draft_margin = nullptr;       // per-draft logit margin z1 - z2 (FLYX_MAX_ROWS)
  flyx_stop_policy* stop_host = nullptr;  // kDraftSlots pinned slots for async policy upload
  flyx_stop_policy* stop_dev = nullptr;   // device copy read by adaptive conditional graph
  flyx::cuda::DevStopState* stop_state = nullptr;
  int stop_slot = 0;
  // A wide round's relaxed acceptance (its logits held 16 rows at a time): each draft's logit in its parent's row
  // (child_logits), FLYX_MAX_WIDE_ROWS - 1 of them; null without wide rows.
  float* child_z = nullptr;

  // scratch (head, sampling, DFlash2 drafter)
  int *drafts = nullptr, *dtok = nullptr, *cand = nullptr, *argi = nullptr;
  float *logits = nullptr, *argv = nullptr, *unary = nullptr, *hid = nullptr;
  float* topv = nullptr;  // topk_rows' slices' candidates: top_rows x kTopkParts x 16
  int* topi = nullptr;
  int top_rows = 0;
  float* cz = nullptr;  // sampling scratch: head rows x vocab
  int* ci = nullptr;
  // DV (docs/design/dv.md): the canvas the next round's tree grows from, what the canvas pass after a round leaves,
  // the round's selector table [depths][16][16], and that pass's rows (`slots` of them, a pass of its own: their
  // buffers, logits, attention partials and GEMM scratch), whose K/V go to the caches' scratch slots [s0, s0 + slots)
  // past the text's.  All on the device: a round's graph reads and writes them, two rounds in flight.
  struct DvRows {
    flyx::cuda::DvCanvas canvas{};
    flyx::cuda::DvStats stats{};
    float* table = nullptr;
    float *x = nullptr, *h = nullptr, *proj = nullptr, *act = nullptr, *q = nullptr, *lora = nullptr, *ug = nullptr,
          *logits = nullptr, *part = nullptr;
    // The adapters' products in bf16, as the drafting GEMMs write them for their b (ld_lora).
    flyx::cuda::bf16* lora16 = nullptr;
    // A round graph's canvas pass, captured on a stream of its own as the body of a conditional node that skips it
    // after a no-op round (dv_context_unless_noop); off where the driver refused such a graph.
    cudaStream_t body = nullptr;
    bool skip_noop = true;
    int ld_lora = 0, splits = 0, s0 = -1;
    flyx::cuda::Scratch scratch{};
  } dv;
  flyx::cuda::CertScratch cert{};  // option head_cert: head_certify's scratch
  float *dx = nullptr, *dh = nullptr, *dhc = nullptr, *dtmp = nullptr, *dproj = nullptr, *dact = nullptr,
        *ddelta = nullptr, *dq = nullptr;
  int ld_proj = 0, ld_act = 0, ld_feats = 0, ld_dproj = 0, ld_dact = 0, ld_ckv = 0;
  // DSpine's bridge and walk (regions/layers.cpp dspine_block); null without a DSpine drafter.  n = the block's rows,
  // c = topk of the walk's candidates a row, senders = 1 + (n - 2) c (the anchor, then each candidate of a row but
  // the last's).
  struct DSpineRows {
    float* codes = nullptr;   // the rows' codes [n][code]; the walk's senders' [senders][code]
    float* gates = nullptr;   // their gate | pred | prev [n][3 rank]; the senders' pred | prev [senders][2 rank]
    float* msg = nullptr;     // the rows' messages [n][rank]
    float* rms = nullptr;     // the rows' rms before the write [n] (the last layer's after the block)
    float* before = nullptr;  // the last layer's rows before its write [n][hidden]
    float* g = nullptr;       // and their gate projections [n][rank]
    float* heads = nullptr;   // the candidates' head rows times the final norm's weight [(n - 1) c][hidden]
    float* tg = nullptr;      // those through the last write [(n - 1) c][rank]
    float* base = nullptr;    // each candidate's score from its row alone [(n - 1) c]
    int* pick = nullptr;      // each sender's best candidate of the row after it [senders]
  } ds;

  // host mirror (valid whenever nothing is in flight)
  int position = 0, round = 0;
  int rope_delta = 0;                 // scalar MRoPE offset for decode / follow-up text turns
  int32_t prev_pos3d[3] = {0, 0, 0};  // last row's 3D position from previous multimodal prefill chunk (for MTP)
  int prev_pos3d_at = -1;             // sequence position that prev_pos3d corresponds to (-1: none)
  // The GDN state and conv window at the last multiple of prefill_align the
  // last prefill pass crossed (pass_snap: where the pass in progress puts it),
  // in round slot snap_slot; -1: none.  Valid until the next pass or round.
  int pass_snap = -1, snap_pos = -1, snap_slot = 0, snap_rope_delta = 0;
  int want_snap = -1;  // prefill_snapshot's position for the next pass (-1: the last multiple it crosses)
  unsigned long long seq_no = 0, read_no = 0;
  struct PendingOp {
    char kind = 0;  // 'P' prefill entry, 'R' round entry
    int rows = 0;   // rows the pass may write past `position`
  };
  std::deque<PendingOp> pending;
  std::atomic<int> write_hi{0};        // upper bound on position + in_flight_rows(); read across threads in copy_home
  int in_flight_rows() const {
    int total = 0;
    for (const PendingOp& p : pending) total += p.rows;
    return total;
  }
  void update_write_hi() { write_hi.store(position + in_flight_rows(), std::memory_order_release); }
  double round_wait_us = 0;            // moving average of the waits for round results
  cudaEvent_t idle_ev = nullptr;       // recorded to wait for the stream without spinning
  cudaEvent_t ready_ev = nullptr;      // everything queued on the stream: a batch round waits for it (launch_batch)
  struct Meta {  // per round slot: the committed rows of the round that started from it
    bool valid = false;
    int start = 0, rows = 0;
    int rope_delta = 0;
  } meta[FLYX_ROUND_SLOTS];

  bool graphs = true;
  // A round's graph per plan and what its head does: 1 the certified head
  // (greedy rounds with head_cert), 2 the penalties before the choice.
  struct RoundGraph {
    cudaGraphExec_t exec;
    int copy_prefix;              // positions transferred by fixed-size H2D nodes captured in this graph
    size_t bytes = 0;             // what it takes at most (graph_size), in graph_held
    unsigned long long used = 0;  // its last launch's seq_no
  };
  std::map<std::pair<const flyx_plan*, int>, RoundGraph> graph;
  // Its graphs' bytes and its shares of batch graphs, within its room.  Atomic: a batch graph dropped
  // by another batch's capture, on whatever thread drives that batch, gives its share back to every sequence it held.
  std::atomic<size_t> graph_held{0};
  unsigned long long graph_retry = 0;  // no capture before this seq_no: the last one found no room
  bool graph_short = false;            // a round ran without its graph for want of room (said once, verbose)
  flyx_graph_stats graph_stats{};      // its rounds from a graph and without, its captures, its graphs' parameters

  // Option profile=1: rounds run without graphs and every node is timed.
  struct Prof {
    double ms = 0;
    long calls = 0;
  };
  bool profile = false;
  std::map<std::string, Prof> prof;
  std::vector<cudaEvent_t> prof_ev;
  void print_profile();

  void release_mapped();  // runtime/tiers.cpp
  void release_prefill_pool();  // runtime/tiers.cpp
  ~flyx_seq() {
    if (st) cudaStreamSynchronize(st);
    if (stage_side.st) cudaStreamSynchronize(stage_side.st);
    if (side.st) cudaStreamSynchronize(side.st);
    release_prefill_pool();
    if (profile) print_profile();
    for (auto e : prof_ev) cudaEventDestroy(e);
    for (auto& g : graph) cudaGraphExecDestroy(g.second.exec);
    for (auto& e : ev)
      if (e) cudaEventDestroy(e);
    if (idle_ev) cudaEventDestroy(idle_ev);
    if (ready_ev) cudaEventDestroy(ready_ev);
    for (void* p : allocs) cudaFree(p);
    for (void* p : host_allocs) cudaFreeHost(p);
    release_mapped();
    for (auto e : stage_ready) cudaEventDestroy(e);
    for (auto e : stage_free) cudaEventDestroy(e);
    flyx::cuda::destroy_side(stage_side);
    if (dv.body) cudaStreamDestroy(dv.body);
    flyx::cuda::destroy_side(side);
    if (st) cudaStreamDestroy(st);
  }
  // dry: count the bytes the allocations would take, allocate nothing (seq_bytes).
  bool dry = false;
  size_t dry_bytes = 0, dry_host_bytes = 0;
  size_t dry_mapped_bytes = 0;  // of dry_host_bytes: the K/V's host parts, mapped for the device only (tiered_alloc)
  // Per-session slab arenas: sized in a dry pass so seq_new() does 1 cudaMalloc + 1 cudaHostAlloc instead of
  // 600+ per-layer driver calls (eliminates ~800 ms WDDM/WSL stalls when opening a concurrent session).
  size_t arena_dev_bytes = 0, arena_host_bytes = 0;
  unsigned char *dev_arena = nullptr, *host_arena = nullptr;
  size_t dev_arena_off = 0, dev_arena_cap = 0;
  size_t host_arena_off = 0, host_arena_cap = 0;
  // Lazy per-slot pinned host slabs for recurrent checkpoints: accounted in dry_host_bytes
  // upfront for budgeting, allocated on first checkpoint(c) so sessions that never fill all slots avoid
  // pinning hundreds of MB of host RAM at session creation.
  size_t ckp_slot_bytes = 0;
  unsigned char* ckp_arena[FLYX_CHECKPOINTS]{};
  // K/V positions [0, device_positions) in device memory, the rest in host memory: the caches whose
  // device part is shorter than the context are address ranges mapped from both (tiered_alloc), each with a view
  // whose rest is a device staging the caches of its role share (an attention reads one layer's at a time).
  int device_positions = INT32_MAX;
  // How a round's attention reads the K/V (flyx_seq_layout::method): FLYX_KV_STRATA keeps a low-bit copy of every
  // position of the caches in tiers, `kv_bits` a value, and reads it with strata attention.
  int kv_method = FLYX_KV_READ_ALL, kv_bits = 0;
  // FLYX_KV_AUTO: a round's rows at positions below strata_from read every position, the rest the
  // copy; FLYX_KV_STRATA is strata_from 0.  strata_phase: which of the two a round launched now reads (the rows of the
  // rounds in flight are not known on the host yet, so a few rounds about the switch run both).
  int strata_from = 0;
  int strata_phase(int rows) const;
  // Forgets the pairs strata attention kept on the device: positions from the one set on may be
  // written again.
  void forget_hot();
  flyx::cuda::StrataScratch strata;
  size_t strata_lse_bytes = 0;  // device bytes of strata.lse (reused for head logits / sampling sort after layers)
  struct Mapped {
    unsigned long long va = 0, view = 0;
    size_t dev = 0, bytes = 0, stage = 0;    // stage: the view's bytes past dev, its staging's
    unsigned long long handles[2] = {0, 0};  // device part, host part
    int slot = -1;                           // the staging the view maps past its device part
  };
  // A staging's size. 0: as large as a cache's host part. Else, with strata attention (FLYX_KV_STRATA
  // or AUTO), stage_window positions, and at least what a round reading the caches reaches past the device part
  // (stage_reach: auto's switch plus the rows of the rounds in flight); a prefill whose keys go past it reads them a
  // window at a time, carrying each row's softmax state in prefill_bufs.attn_state.
  int stage_window = 0, stage_reach = 0;
  // The MTP layer's last positions a draft step attends to, gathered in order, and the parameters
  // that make its attention read exactly them.
  flyx::cuda::KvRef mtp_window{};
  flyx::cuda::DevParams* P_window = nullptr;
  flyx::cuda::StrataScratch mtp_strata;  // strata attention over the MTP layer's copy: one row
  std::vector<Mapped> mapped;
  struct Staging {
    int role = 0;    // 0 K, 1 V
    int parity = 0;  // two of a role and size: the next cache's positions are copied to one while an attention reads
                     // the other (KernelOptions::stage_ahead; one without)
    size_t bytes = 0;
    size_t offset = 0;  // in the pool
    size_t row = 0;     // bytes a position: the staging's start holds position device_rows on
    int device_rows = 0;
    float* scales = nullptr;  // FP8: the same positions' scales of its role, device memory
  };
  std::vector<Staging> staging;
  // The device pool: the stagings, and the copy's device home (KvBuf::kq_dev) when the caches in
  // tiers have a copy, one memory as large as the larger of the two (staging_bytes, copy_bytes), mapped twice: a pass
  // that reads the caches through the stagings runs with the copy at home in host memory, a round that reads only the
  // copy with it in the pool.  The copy lies at the pool's start, the stagings at its end, so they share no more than
  // they must.  pool_finish, after the caches: the pool and its mappings.  copy_home moves only what changed, on the
  // sequence's stream: to the pool (dev), the 16s of positions written since the copy last moved (copy_dirty_from on)
  // and what the stagings wrote over (their rows up to staged_to); to host memory, the positions written.
  std::vector<unsigned long long> pool_pieces;  // the pool's own memory, a piece of the granularity each
  std::shared_ptr<flyx::cuda::VmmSlab> shared_prefill;  // model's prefill VMM slab mapped at [owned_pool_bytes, pool_bytes)
  unsigned long long pool_va = 0;
  size_t pool_bytes = 0, owned_pool_bytes = 0, prefill_shared = 0, staging_bytes = 0, copy_bytes = 0;
  // The copies in host memory: copy_host of the copies (flyx_seq_layout), spread over the layers,
  // read through copy_slots stagings of copy_slot_bytes in the pool (both parts), after the copies on the device.
  int copy_host = 0, copies = 0, copy_slots = 0;
  int copy_scan_end = 0;  // complete key tiles needed by the first half of a staged Strata scan's splits
  size_t copy_slot_bytes = 0, copy_stage_off = 0;
  bool copy_pooled = false, copy_dev = false, prefill_clobbered = true;
  bool uses_shared_prefill() const { return prefill_shared > 0 && shared_prefill && m && shared_prefill == m->prefill_slab; }
  int copy_dirty_from = 0, staged_to = 0;
  void pool_finish();
  void copy_home(bool dev);
  void copy_written(int from) { copy_dirty_from = std::min(copy_dirty_from, from); }
  void staged(int to) { staged_to = std::max(staged_to, std::min(to, cap)); }
  // `positions` x `row` bytes, rows [0, device_positions) in device memory and the rest in page-locked host memory,
  // one address range (runtime/tiers.cpp); a plain device allocation when every row fits there (view null).  Else
  // view is the range's view, mapping staging[slot] past its device rows.
  struct Tiered {
    void *at = nullptr, *view = nullptr;
    int device_rows = INT32_MAX, slot = -1, stage_rows = INT32_MAX;  // stage_rows: positions its staging holds
    float* scales = nullptr;  // its staging's scales (scale_floats a position: FP8's KV heads; 0: none)
  };
  Tiered tiered_alloc(size_t positions, size_t row, int role, int parity, int scale_floats);
  // The copies to the staging beside the pass (runtime/tiers.cpp). stage_begin, given what the pass's
  // attentions read through stagings in their order (a cache's views, or its copy in host memory),
  // forks the stage stream and starts the first one's copy of each staging there: the positions written before the
  // pass, while the layers before its attention run.  stage_before (copy_before for a copy), before an attention,
  // waits for its copy and copies the pass's own rows (all of them without stage_begin: a batch's pass); stage_after
  // marks the staging free and starts the copy of the next one that uses it; stage_end joins the stream back.
  // stage_text, for a pass after the round's attentions (DV's canvas, whose own rows go to the staging past the
  // text): the positions in host memory before P->position, at once.
  struct StageItem {
    int kv = -1;        // the cache (index in kv)
    bool copy = false;  // its low-bit copy, else its views
    int from = 0;  // first position the staging kernel copies; for a low-bit copy, DMA handles [0, from)
    int to = 0, parity = -1;  // prefill window [from, to) at parity; for DMA, first segment [0, to) when nonempty
  };
  flyx::cuda::SideStream stage_side{};
  std::vector<cudaEvent_t> stage_ready, stage_free;  // per staging: the K/V's (k's index), then the copies'
  std::vector<StageItem> stage_order;
  size_t stage_next = 0;
  bool stage_on = false;
  size_t stage_event(const StageItem& x) const {
    const flyx::cuda::KvBuf& b = kv[size_t(x.kv)];
    return x.copy ? staging.size() + size_t(b.copy_slot) : size_t(x.parity >= 0 ? b.wslot[x.parity] : b.slot);
  }
  // A prefill's windows of a cache's keys: none when its stagings hold what it reads, else the ones
  // past the device part's whole 64s, of the stagings' positions each (the first starts at window_from).
  int windows(const flyx::cuda::KvBuf& kb, int rows) const;
  int window_from(const flyx::cuda::KvBuf& kb) const {
    return kb.device_positions / flyx::cuda::kFlashWindowAlign * flyx::cuda::kFlashWindowAlign;
  }
  int window_len(const flyx::cuda::KvBuf& kb) const {
    return kb.stage_rows / flyx::cuda::kFlashWindowAlign * flyx::cuda::kFlashWindowAlign;
  }
  // Before a window's attention: its copy (made ahead, or now) done and the pass's rows in it copied; the KvRef the
  // attention reads its keys through.
  flyx::cuda::KvRef window_before(const flyx::cuda::KvBuf& kb, int w0, int w1, int rows, cudaStream_t st);
  void window_stage(const flyx::cuda::KvBuf& b, int w0, int w1, int parity, int rows, flyx::cuda::KvStagePart part,
                    int blocks, cudaStream_t st) const;
  // A small pass (a prefill's last rows, a short message) whose keys go past the stagings reads host memory where it
  // is: its round kernel reads each key once, at the bus's speed.
  bool reads_direct(const flyx::cuda::KvBuf& kb, int rows, bool round) const {
    return kb.vk && !round && rows <= FLYX_MAX_ROWS && (long long)position + rows > kb.staged_cap();
  }
  void stage_begin(std::vector<StageItem> order, cudaStream_t st);
  void stage_ahead(const StageItem& x);
  void stage_wait(const flyx::cuda::KvBuf& kb, bool copy, cudaStream_t st, bool first_segment = false);
  void stage_before(const flyx::cuda::KvBuf& kb, int pos_add, int span, cudaStream_t st);
  // When the first segment suffices to start the scan, return the event for the rest of the prefix.
  cudaEvent_t copy_before(const flyx::cuda::KvBuf& kb, int span, cudaStream_t st, bool scan_halves = false);
  void stage_after(const flyx::cuda::KvBuf& kb, cudaStream_t st);
  void stage_end(cudaStream_t st);
  void stage_text(const flyx::cuda::KvBuf& kb, cudaStream_t st);
  void create_stage_stream();
  std::vector<void*> host_allocs;
  // Small pinned host staging (not counted in dry_host_bytes; freed with host_allocs).
  template <class T>
  T* pinned_alloc(size_t n) {
    const size_t bytes = n * sizeof(T);
    const size_t aligned = (bytes + 255) / 256 * 256;
    if (dry) {
      arena_host_bytes += aligned;
      return reinterpret_cast<T*>(uintptr_t(256));
    }
    if (bytes == 0) return nullptr;
    if (host_arena && host_arena_off + aligned <= host_arena_cap) {
      unsigned char* p = host_arena + host_arena_off;
      host_arena_off += aligned;
      return reinterpret_cast<T*>(p);  // zeroed once with the arena
    }
    void* p = nullptr;
    CK(cudaMallocHost(&p, bytes));
    host_allocs.push_back(p);
    return static_cast<T*>(p);
  }
  // Page-locked host memory, zeroed: state the session copies to and from the device now and then and no kernel
  // reads in a round (the checkpoints). Not in dry_bytes (the device's budget).
  template <class T>
  T* host_alloc(size_t n) {
    const size_t bytes = n * sizeof(T);
    const size_t aligned = (bytes + 255) / 256 * 256;
    if (dry) {
      dry_host_bytes += aligned;
      arena_host_bytes += aligned;
      return reinterpret_cast<T*>(uintptr_t(256));
    }
    if (bytes == 0) return nullptr;
    if (host_arena && host_arena_off + aligned <= host_arena_cap) {
      unsigned char* p = host_arena + host_arena_off;
      host_arena_off += aligned;
      return reinterpret_cast<T*>(p);  // zeroed once with the arena
    }
    void* p = nullptr;
    CK(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
    std::memset(p, 0, bytes);
    host_allocs.push_back(p);
    return static_cast<T*>(p);
  }
  template <class T>
  T* alloc(size_t n) {
    const size_t bytes = n * sizeof(T);
    const size_t aligned = (bytes + 255) / 256 * 256;
    if (dry) {
      dry_bytes += aligned;
      arena_dev_bytes += aligned;
      return reinterpret_cast<T*>(uintptr_t(256));
    }
    if (bytes == 0) return nullptr;
    if (dev_arena && dev_arena_off + aligned <= dev_arena_cap) {
      unsigned char* p = dev_arena + dev_arena_off;
      dev_arena_off += aligned;
      return reinterpret_cast<T*>(p);  // zeroed once with the arena on st
    }
    void* p = nullptr;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMemsetAsync(p, 0, bytes, st));  // through the session's stream (runner.cpp: set_params)
    allocs.push_back(p);
    return static_cast<T*>(p);
  }
};

namespace flyx::cuda {

template <class B>
const B& block(const flyx_bmodel* m, int i) {
  return std::get<B>(m->blocks[size_t(i)]);
}
template <class B>
B& block(flyx_bmodel* m, int i) {
  return std::get<B>(m->blocks[size_t(i)]);
}

// The rows a pass works on: their buffers and leading dimensions, and the launch (the stream, the
// scratch the GEMM inputs are staged in).  A sequence's pass works on its own (rows_of); a round of several
// sequences runs its shared parts on the batch's rows and each sequence's own parts on that sequence's slice, all
// on the batch's stream.
struct RowView {
  RowBufs b;
  float* logits = nullptr;  // the head's rows, vocab floats each
  int ld_proj = 0, ld_act = 0, ld_feats = 0, ld_ckv = 0;
  Launch L{};
};
// The sequence's row buffers (its own when bufs is null, or the prefill buffers it borrows), on its stream.
RowView rows_of(flyx_seq* s, RowBufs* bufs = nullptr);

struct Pass {
  flyx_plan* p;
  flyx_seq* s;
  int rows;              // target rows in this pass
  bool round;            // round (else prefill)
  bool sample_last;      // prefill: head on the last row
  bool cert = false;     // the head chooses with the certified head (greedy, no mask, option head_cert)
  bool pen = false;      // the request has a penalty: the head's logits are penalized before the choice
  bool relaxed = false;  // the request asks for relaxed acceptance (a round with drafts)
  bool draft_conf = true;  // s->draft_prob / s->draft_margin were written by this round's drafter
  int strata = 0;        // a round's attention over the caches with a copy (flyx_seq::strata_phase): kStrataNone,
                         // kStrataAll, kStrataBoth
  int accept_width = -1; // override p->width for FLYX_NODE_ACCEPT inside adaptive verify branches (-1: p->width)
  // Multimodal prefill_ex overrides (device pointers; null/0 on pure-text passes and all rounds).
  const float* embeds = nullptr;
  const int32_t* embed_rows = nullptr;
  int n_embeds = 0;
  const int32_t* pos_3d = nullptr;
  const int32_t* mtp_pos_3d = nullptr;
  RowView v{};           // the rows it works on (rows_of(s) for a sequence's own pass)
  int width() const { return accept_width >= 0 ? accept_width : p->width; }
  bool reads_copy(const KvBuf& kb) const { return round && kb.kq && strata != kStrataNone; }
  bool reads_cache(const KvBuf& kb) const { return !reads_copy(kb) || strata == kStrataBoth; }
  // The copy alone: a copy in host memory read through its staging.
  bool reads_copy_only(const KvBuf& kb) const { return reads_copy(kb) && strata == kStrataAll; }
};

// loader/bind.cpp: blocks and the state table from the description's tables; uploads the weights.
void bind(flyx_bmodel* m, const flyx_model_desc& d, LoadTimes* times = nullptr);
// The DV block i's selector token tables (DvBlock::sel_ta, sel_tb: [vocab][sel_rank] float32 each, counted with the
// weights), made once, when lowering a plan that drafts with them: past bind, where out of GPU memory is no refusal
// of the model but a plan the engine lowers again once the backend has given back what it keeps for speed (trim).
void make_dv_tables(flyx_bmodel* m, int i);

// loader/weights.cpp
void make_draft_weights(flyx_bmodel* m);
void make_head_fp8(flyx_bmodel* m);
void make_head_w12(flyx_bmodel* m);
uint64_t trim(flyx_bmodel* m, uint64_t bytes);  // the ABI's trim: speed-only copies of the head

// regions/linear.cpp (gemm and gemm_presplit above)
void draft_logits(const HeadBlock& h, const float* x, int ldx, float* y, int rows, const Launch& L);

// regions/plan.cpp: which blocks the kernels take, the block a node runs, the
// implementation lower() picks for it, and running a plan's nodes.
bool supported(const flyx_bmodel* m, const Block& b, std::string* why);
const Block* block_of(const flyx_bmodel* m, const flyx_node& n, std::string* err);
const RegionImpl* resolve(const flyx_bmodel* m, const flyx_node& n, const Block* b, int max_rows);
extern const RegionImpl kGeneric;  // every node's generic implementation
std::string node_name(const flyx_node& n, const Block* b);
void forward(const Pass& ps, int copy_prefix = 0);
void forward_target(const Pass& ps, int copy_prefix = 0);
void draft_step_range(const Pass& ps, int start, int end);
bool sequential_draft(const flyx_plan* p);
void run_node(const Pass& ps, const flyx_node& n, const Block* b);  // the generic implementation of a node
bool generic_impl(const RegionImpl* r);                                // r is the generic one (not a fused override)
int lower(flyx_bmodel* m, const flyx_program* prog, flyx_plan** out);
void plan_free(flyx_plan* p);

// A round of several sequences: each sequence's pass (its plan, rows and head) on its slice of the
// batch's rows, and the batch's rows all together; one stream (regions/batch.cpp).
struct BatchPass {
  std::vector<Pass> seq;   // each sequence's pass; its view a slice of v
  std::vector<int> first;  // each sequence's first row in the batch
  RowView v;               // all the rows
  int rows = 0;
  bool cert = false;       // every sequence chooses with the certified head
};
void run_batch(const BatchPass& bp);

// runtime/state.cpp
RowNeeds row_needs(const flyx_bmodel* m);
int max_drafts(const Block& drafter);  // most drafts a round of this drafter can verify (0: not a drafter)
int ensure_prefill_rows(flyx_bmodel* m, int rows);
int seq_new(flyx_bmodel* m, int32_t max_ctx, flyx_seq** out);
void seq_free(flyx_seq* s);
int seq_bytes(flyx_bmodel* m, int32_t max_ctx, uint64_t* out);
int seq_new_in(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq** out);
int seq_size(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq_size* out);
int memory(flyx_bmodel* m, flyx_memory_info* out);
void alloc_batch(flyx_bmodel* m);  // the batch rounds' buffers, at open

// runtime/graphs.cpp: a captured graph's nodes by kind and what it takes at most once instantiated; a
// round plan lowered, its first graph's size (counted once: the first capture's), freed; the room for a session's
// graphs (0 with graphs off; `count`: the plans no session captured yet sized first, on a session of their own, which
// throws where that cannot be made); room for a graph of `bytes` beside the session's own and its shares of batch
// graphs, the ones it ran least lately given up (false: none, the round runs without a graph).
GraphSize graph_size(cudaGraph_t g);
void plan_lowered(flyx_bmodel* m, const flyx_plan* p);
void plan_counted(flyx_bmodel* m, const flyx_plan* p, size_t bytes);
void forget_graphs(flyx_bmodel* m, const flyx_plan* p);
size_t graph_keep(const flyx_seq& s, bool count = false);  // what the session's graphs may hold
size_t graph_room(const flyx_seq& s, bool count = false);  // and the pool's part beside it: what seq_bytes counts
bool make_graph_room(flyx_seq* s, size_t bytes);
// A batch graph dropped (once the batch stream is done with it: *synced), its share given back to its sequences
// (`quiet`: a CUDA error let go, not thrown, for seq_free); room for a share of a new one in each of `seqs` (the
// batch's lock held; false: none, the batch runs without a graph).
using BatchGraphs = std::map<BatchKey, BatchRounds::CachedGraph>;
BatchGraphs::iterator drop_batch_graph(BatchRounds& B, BatchGraphs::iterator it, bool* synced, bool quiet = false);
bool make_batch_room(BatchRounds& B, flyx_seq* const* seqs, int n, size_t share);

// runtime/sleep.cpp: sleeps about `us` microseconds (on Windows a high-resolution timer: at least ~0.5 ms)
void sleep_us(int64_t us);

// runtime/runner.cpp
int launch_batch(flyx_plan* const* plans, flyx_seq* const* seqs, const flyx_gen_params* const* gens,
                 int32_t n);
void sync_stream(flyx_seq* s);
void set_params(flyx_seq* s, int position, int anchor, int round);  // and forget_canvas
void reset_state(flyx_seq* s, int anchor);
// DV's canvas follows no round (DvCanvas::next_start = -1): its next round drafts DV's first tree and its canvas pass
// starts from MASK (a prefill's CONTEXT, a rewind, a restore, a snapshot load); queued on st.
void forget_canvas(flyx_seq* s, cudaStream_t st);
int prefill(flyx_plan* p, flyx_seq* s, const int32_t* tokens, int32_t n, const flyx_gen_params* sample);
int prefill_ex(flyx_plan* p, flyx_seq* s, const flyx_prefill_input* in, const flyx_gen_params* sample);
int encode_vision(flyx_plan* p, const flyx_vision_input* in, float* out_embeds, int32_t out_cap_rows,
                  int32_t* out_rows);
int launch(flyx_plan* p, flyx_seq* s, const flyx_gen_params* g);
// A greedy round of plan p on s, captured as launch() would and not kept: its graph's size (graph_room).
GraphSize round_graph_size(flyx_plan* p, flyx_seq* s);
int graph_stats(flyx_seq* s, flyx_graph_stats* out);  // the ABI's
int read_result(flyx_seq* s, flyx_round_result* out);
int read_logits(flyx_seq* s, int32_t row, float* out);
int ready(flyx_seq* s);
int set_reply(flyx_seq* s, const int32_t* tokens, int32_t n);
int rewind(flyx_seq* s, int32_t pos, int32_t anchor);
int checkpoint(flyx_seq* s, int32_t c);
int prefill_snapshot(flyx_seq* s, int32_t pos);
int restore(flyx_seq* s, int32_t c, int32_t anchor);
// A checkpoint in host memory and back (runtime/runner.cpp).
int snapshot_bytes(flyx_seq* s, int32_t c, int32_t begin, uint64_t* out);
int snapshot_save(flyx_seq* s, int32_t c, int32_t begin, const flyx_host_pieces* dst, uint64_t bytes);
int snapshot_load(flyx_seq* s, const flyx_host_pieces* src, uint64_t bytes, int32_t anchor);
int set_oracle(flyx_seq* s, const int32_t* tokens, int32_t n);

}  // namespace flyx::cuda
