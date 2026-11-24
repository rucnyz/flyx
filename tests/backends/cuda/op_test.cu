// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Every CUDA kernel against the float64 reference ops (ops/ops.h).
//
//   cuda_op_test            run all checks, exit 1 on any failure
//
// Inputs are random; weights and caches go through bf16 on both sides, so
// the remaining difference is float32 arithmetic (and bf16 activations in the
// tensor-core GEMM).
//
// The tests live a family a file (op_test_*.cu); main() runs them in this order.
#include "tests/backends/cuda/op_test.cuh"

using namespace flyx::optest;

int main() {
  // A nonfinite element between two matching ones must not disappear in the max-error reduction.
  const Vec finite{1., 2., 3.};
  for (double bad : {std::nan(""), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
    const Vec invalid{1., bad, 3.};
    if (std::isfinite(relative_error(finite, invalid)) || std::isfinite(relative_error(invalid, finite)) ||
        std::isfinite(relative_error(invalid, invalid))) {
      std::printf("FAIL nonfinite values passed the reference comparison\n");
      return 1;
    }
  }
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("no CUDA device\n");
    return 2;
  }
  {  // scratch for every GEMM input the tests stage (prefill rows up to 1024, K up to 16384)
    const size_t bytes = scratch_bytes(1024, 16384);
    static Scratch sc = scratch_in(dev<unsigned char>(bytes), bytes, 1024, 16384);
    TL.scratch = &sc;
  }
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
#if defined(FLYX_TUNE_CANDIDATES) && !FLYX_TUNE_CANDIDATES
  std::printf("a release build (FLYX_TUNE_CANDIDATES off): the GEMM configurations no tuning table gives on this GPU "
              "are not compiled, and the tests of every configuration skip them\n");
#endif
  test_gemm();
  test_gemm_round();
  test_gemm_draft_tables();
  test_gemm_tables_compiled();
  test_blocks();
  test_blocks_round_stacks();
  test_blocks_round_glu();
  test_blocks_round_side();
  test_rmsnorm();
  std::printf("attention\n");
  test_attention(256, 64, 16, 4, 16, 300, true, 1e7);   // target layer shape, verify rows
  test_attention(256, 64, 16, 4, 1, 1000, true, 1e7);   // plain decode
  test_attention(256, 64, 16, 4, 37, 50, true, 1e7);    // prefill chunk
  test_attention(256, 256, 16, 4, 16, 200, false, 1e4); // drafter block
  test_attention(128, 128, 32, 8, 16, 300, false, 1e7, 100);  // drafter block, a sliding window (z-lab's drafters)
  test_attention(32, 8, 4, 2, 16, 9, false, 1e4, 8);    // a window shorter than the block (the tiny drafter's)
  test_attention(32, 8, 4, 2, 5, 9, true, 1e7);         // tiny model
  test_attention(256, 64, 16, 4, 256, 3000, true, 1e7);  // prefill chunk deep in the context
  test_attention(128, 128, 8, 2, 100, 0, true, 1e4);     // first chunk, rows not a multiple of 64
  // Six and eight query heads a KV head (Qwen3.5's 27B, Qwen3's 32B): the grouped kernel with one row a block at head
  // dim 256, two at 128.
  test_attention(256, 64, 24, 4, 16, 300, true, 1e7);   // the 27B's verify rows
  test_attention(256, 64, 24, 4, 1, 5000, true, 1e7);   // the 27B's decode deep in the context
  test_attention(256, 64, 24, 4, 100, 2000, true, 1e7);  // the 27B's prefill chunk
  test_attention(128, 64, 48, 8, 9, 700, true, 1e6);     // G = 6 at head dim 128, rows not a multiple of two
  test_attention(128, 64, 64, 8, 16, 300, true, 1e6);    // G = 8
  test_attention(256, 64, 32, 4, 3, 300, true, 1e6);     // G = 8 at head dim 256
  // No q/k norms (Llama) in MiniCPM5-1B's layer: 16 query heads on 2 KV heads of 128, the whole head
  // rotated.  100 rows give a block three heads, which the barrier standing in for the norm's sum keeps apart.
  test_attention(128, 128, 16, 2, 16, 300, true, 5e6, 0, false);   // verify rows
  test_attention(128, 128, 16, 2, 1, 1000, true, 5e6, 0, false);   // decode
  test_attention(128, 128, 16, 2, 100, 50, true, 5e6, 0, false);   // prefill chunk
  std::printf("attention, fp8 kv cache\n");
  test_attention_fp8(256, 64, 16, 4, 16, 300, 1e7);
  test_attention_fp8(256, 64, 16, 4, 1, 3000, 1e7);
  test_attention_fp8(256, 64, 16, 4, 100, 2000, 1e7);
  test_attention_fp8(32, 8, 4, 2, 5, 9, 1e7);
  test_attention_fp8(256, 64, 24, 4, 16, 300, 1e7);
  test_attention_fp8(256, 64, 24, 4, 32, 400, 1e7);  // full historical tiles before, across and after the scale split
  test_attention_fp8(256, 64, 24, 4, 1, 3000, 1e7);
  test_attention_fp8(256, 64, 24, 4, 70, 1000, 1e7);  // G = 6, rows in several tiles of 16, the last one short
  test_attention_fp8(128, 64, 64, 8, 9, 300, 1e6);
  test_attention_fp8(128, 64, 48, 8, 9, 700, 1e6);   // G = 6 at head dim 128: the grouped kernel, 3 heads a block
  test_attention_fp8(128, 64, 16, 8, 37, 300, 1e6);  // G = 2 at head dim 128 (Qwen3-0.6B, 1.7B): a block a head
  test_attention_fp8(256, 64, 8, 4, 37, 300, 1e7);   // G = 2 at head dim 256: a block a head
  test_attention_fp8(128, 128, 16, 2, 100, 50, 5e6, false);  // MiniCPM5-1B's layer, no q/k norms: a prefill chunk
  test_attention_fp8(128, 128, 16, 2, 1, 1000, 5e6, false);  // ... decode
  // A tree round of more than 16 rows: MiniCPM5-1B's layer, then about rows_attention_from (some rows on tensor
  // cores), then four query heads a KV head.
  test_attention_tree(128, 16, 2, 300);
  test_attention_tree(128, 16, 2, 8180);
  test_attention_tree(64, 16, 4, 1000);
  std::printf("strata attention\n");
  test_strata(256, 24, 4, 7, 3000, 3, true);   // the 27B's shape, a round of MTP's rows, FP8 cache
  test_strata(256, 24, 4, 1, 257, 3, false);   // single-row 3-bit scan with binary16 K/V and a partial tile
  test_strata(256, 24, 4, 4, 1022, 2, true);   // 1023..1026 keys: full 32-key tiles and an unpaired final key
  test_strata(256, 24, 4, 1, 1000, 2, false);  // plain decode, 2 bits, binary16 cache
  test_strata(256, 24, 4, 3, 900, 2, false);   // a round's rows on a binary16 cache: its pairs from pageable memory too
  test_strata(256, 16, 4, 9, 700, 4, true);    // G = 4, 4 bits, rows past 8
  test_strata(128, 16, 4, 5, 500, 3, false);   // head dim 128
  test_strata(64, 8, 2, 3, 300, 3, true);      // head dim 64 (the layout's shared words)
  test_strata(32, 4, 2, 5, 100, 3, false);     // the tiny model's
  // A round whose later rows fill one split more than its first (129 keys after 128, 257 after 256): their splits are
  // shorter (96 keys, not 128), so the round's last row is not the one whose first split ends last.
  test_strata(256, 24, 4, 4, 254, 2, false);   // 255..258 keys, 2 bits: the scan a warp runs alone
  test_strata(256, 24, 4, 7, 252, 3, true);    // 253..259 keys, 3 bits: the scan several warps run together
  test_strata(256, 16, 4, 15, 114, 4, false);  // 115..129 keys: every row but the last ends past the last one's 96
  test_strata(32, 4, 2, 5, 125, 2, true);      // the tiny model's, 126..130 keys
  test_pageable_kv_ops();
  // Every head dim the kernels take (the backend loads no other): decode, a prefill chunk on the tensor cores, and
  // the same with an FP8 cache.
  for (int hd = 32; hd <= 256; hd += 32)
    if (attention_head_dim(hd)) {
      std::printf("attention, head dim %d\n", hd);
      test_attention(hd, hd / 2, 8, 2, 1, 300, true, 1e4);
      test_attention(hd, hd / 2, 8, 2, 37, 50, true, 1e4);
      test_attention_fp8(hd, hd / 2, 8, 2, 37, 300, 1e4);
    }
  test_quantize_nvfp4();
  test_fused_producers();
  test_lora();
  test_gdn(GdnDims{2, 4, 16, 16, 4}, 7, 1e-5);     // tiny model
  test_gdn(GdnDims{2, 4, 64, 48, 4}, 37, 1e-5);    // several staged chunks, a partial block of state rows
  test_gdn(GdnDims{1, 2, 128, 128, 4}, 300, 1e-5); // Qwen3.5 head dims, more rows than a conv block
  test_gdn(GdnDims{2, 2, 24, 16, 3}, 9, 1e-5);     // generic kernels (dk not a multiple of 16, ck 3)
  test_gdn(GdnDims{2, 4, 64, 48, 4}, 4, 1e-5);     // one-kernel round path: a partial block of state rows
  test_gdn(GdnDims{4, 8, 128, 128, 4}, 4, 1e-5);   // one-kernel round path, Qwen3.5 head dims
  test_gdn(GdnDims{4, 8, 128, 128, 4}, 8, 1e-5);   // one-kernel round path at its most rows
  test_gdn(GdnDims{2, 4, 64, 48, 4}, 6, 1e-5);     // one-kernel round path, 6 rows, a partial block
  test_gdn(GdnDims{2, 4, 32, 32, 2}, 3, 1e-5);     // one-kernel round path, ck 2
  test_gdn_chunk(GdnDims{1, 2, 128, 128, 4}, 300, 0, 1e-2);   // bf16 products (8 bits) into float32, as FLA's
  test_gdn_chunk(GdnDims{1, 2, 128, 128, 4}, 200, 37, 1e-2);  // a pass starting inside a chunk
  test_gdn_chunk(GdnDims{2, 4, 128, 128, 4}, 1024, 0, 1e-2);  // a prefill chunk of the real size
  test_gemm_w4a4(200, 384, 512);   // rows and columns not multiples of the 128 x 128 tile
  test_gemm_w4a4(256, 256, 2560);
  test_gemm_w4a4(130, 256, 384);   // K no multiple of 256: the 128-value stages
  test_gemm_glu(40, 136, 96);     // a partial tile; NVFP4 scales the slow way (K % 256 != 0)
  test_gemm_glu(200, 384, 512);   // W4A4 too
  test_gemm_glu(300, 1024, 2560);
  test_gemm_fp8(1000, 2560);   // a partial column tile
  test_head_cert(5003, 256);   // a partial column group
  test_head_cert(20000, 2560);
  test_head_cert_input_rounding();
  test_gemm_w12(1000, 2560);
  test_gemm_w12(64, 256);
  test_draft_pick(20000, 512, true);
  test_draft_pick(200, 256, false);
  test_draft_pick(3000, 5120, false, false);  // K 5120 (the 27B's): the one-row tables past 48 KB of shared memory
  test_draft_pick_blocks();
  test_quantize_q8_0();
  test_grouped_conv();
  test_select();
  test_stop_decide();
  test_dv();
  test_dspine();
  test_sample();
  test_penalize();
  test_wide_rows_choice();
  test_fp4();
  test_layernorm();
  test_add_bias();
  test_vact();
  test_gemm_f32();
  test_vision_pos_embed();
  test_vision_rope_2d();
  test_vision_attention();
  test_vision_avg_pool();
  test_embed_scatter();
  test_mrope();
  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
