// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A GEMM on a weight of any format (bf16, NVFP4, packed GGML blocks), and the draft logits.
#include <mutex>

#include "backends/cuda/backend.h"

namespace flyx::cuda {

// A linear layer's weight [N, K]: bf16, NVFP4 or packed blocks (see kernels/kernels.cuh).

// Packed blocks on the rows staged for them: each run of one block type in the stack computes its columns (two runs
// in one launch where the kernels have that pair).
void gemm_packed(const Lin& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  gemm_blocks_staged_runs(W.blk.data(), int(W.blk.size()), K, y, ldy, rows, accumulate, L);
}

// The GEMMs a run multiplies, each shape once (option log_shapes): what `cuda_bench tune gemm NxK...` takes.
void log_shape(const KernelOptions* opt, const char* fmt, int N, int K, int rows) {
  if (!opt || !opt->log_shapes) return;
  static std::mutex mu;
  static std::map<std::string, int> seen;  // log lines only
  const std::string key = std::string(fmt) + " " + std::to_string(N) + " " + std::to_string(K);
  std::lock_guard<std::mutex> lock(mu);
  if (seen[key]++ == 0) std::fprintf(stderr, "gemm shape %s (rows %d)\n", key.c_str(), rows);
}

// Skinny GEMM whose input a fused producer already wrote into the split scratch (8-bit inputs: in 8 bits).
void gemm_presplit(const Lin& W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  log_shape(L.opt, W.packed() ? "blocks" : W.q ? "nvfp4" : "bf16", N, K, rows);
  if (W.packed()) {
    if (!W.q8_inputs()) blocks_stage_presplit(K, rows, L, false);  // the producer's bf16 rows as these kernels hold them
    gemm_packed(W, K, y, ldy, rows, accumulate, L);
  } else if (W.q) gemm_fp4_presplit(W.q, W.s, W.rs, N, K, y, ldy, rows, accumulate, L);
  else flyx::cuda::gemm_presplit(W.w, N, K, y, ldy, rows, accumulate, L);
}

void gemm_glu_presplit(const Lin& W, int I, int K, int rows, const Launch& L) {
  log_shape(L.opt, W.q ? "nvfp4" : "bf16", 2 * I, K, rows);
  gemm_glu_staged(W.w, W.q, W.s, W.rs, I, K, rows, L);
}

void gemm_on_glu(const Lin& W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  log_shape(L.opt, W.q ? "nvfp4" : "bf16", N, K, rows);
  flyx::cuda::gemm_on_glu(W.w, W.q, W.s, W.rs, N, K, y, ldy, rows, accumulate, L);
}

void gemm(const float* x, int ldx, const Lin& W, int N, int K, float* y, int ldy, int rows, bool accumulate,
          const Launch& L) {
  log_shape(L.opt, W.packed() ? "blocks" : W.q ? "nvfp4" : "bf16", N, K, rows);
  if (L.round_rows && rows > 16) {  // a wide round: the 1..16-row bits for every row
    if (W.q && !W.packed()) return gemm_fp4_rows(x, ldx, W.q, W.s, W.rs, N, K, y, ldy, rows, accumulate, L);
    // bf16: k_gemm_skinny_mt, the weights read once for every 16-row tile (gemm's round rows)
    if (!W.packed()) return flyx::cuda::gemm(x, ldx, W.w, N, K, y, ldy, rows, accumulate, L);
    for (int r0 = 0; r0 < rows; r0 += 16)
      gemm(x + size_t(r0) * ldx, ldx, W, N, K, y + size_t(r0) * ldy, ldy, std::min(16, rows - r0), accumulate, L);
    return;
  }
  if (W.packed()) {
    blocks_stage(x, ldx, K, rows, L, W.blk[0].q8);
    gemm_packed(W, K, y, ldy, rows, accumulate, L);
  } else if (W.q) gemm_fp4(x, ldx, W.q, W.s, W.rs, N, K, y, ldy, rows, accumulate, L);
  else flyx::cuda::gemm(x, ldx, W.w, N, K, y, ldy, rows, accumulate, L);
}

// Draft logits for `rows` rows of x.
void draft_logits(const HeadBlock& h, const float* x, int ldx, float* y, int rows, const Launch& L) {
  if (h.draft.q)
    gemm(x, ldx, h.draft, h.vocab, h.hidden, y, h.vocab, rows, false, L);
  else if (h.wb.data)
    gemm_blocks(x, ldx, h.wb, h.hidden, y, h.vocab, rows, false, L);
  else
    gemm(x, ldx, h.w, h.vocab, h.hidden, y, h.vocab, rows, false, L);
}

}  // namespace flyx::cuda
