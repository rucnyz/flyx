// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Weights onto the device, and the NVFP4 and Q8_0 copies drafting reads.
#include "backends/cuda/loader/loader.h"

#include <condition_variable>
#include <exception>

namespace flyx::cuda {

// Persistent worker pool scoped to a single model load so WeightLoader::parallel_rows avoids spawning and joining
// thousands of OS threads across tensors while leaving zero background threads alive after bind() returns.
class RowPool {
 public:
  static size_t default_threads() {
    static const size_t n = std::max<size_t>(1, std::min<size_t>(16, std::thread::hardware_concurrency()));
    return n;
  }

  explicit RowPool(size_t max_threads) : max_threads_(std::max<size_t>(1, max_threads)) {
    workers_.reserve(max_threads_ - 1);
    try {
      for (size_t idx = 1; idx < max_threads_; ++idx) workers_.emplace_back([this, idx] { worker_loop(idx); });
    } catch (...) {
      shutdown();
      throw;
    }
  }

  ~RowPool() { shutdown(); }
  RowPool(const RowPool&) = delete;
  RowPool& operator=(const RowPool&) = delete;

  void run(size_t n, const std::function<void(size_t, size_t)>& fn) {
    const size_t parts = std::min(max_threads_, n);
    if (parts <= 1) {
      if (n) fn(0, n);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      n_ = n;
      parts_ = parts;
      fn_ = &fn;
      error_ = nullptr;
      active_ = parts - 1;
      ++gen_;
    }
    wake_cv_.notify_all();
    std::exception_ptr caller_err;
    try {
      fn(0, n / parts);
    } catch (...) {
      caller_err = std::current_exception();
    }
    std::exception_ptr err;
    {
      std::unique_lock<std::mutex> lock(mu_);
      done_cv_.wait(lock, [&] { return active_ == 0; });
      err = caller_err ? caller_err : error_;
      error_ = nullptr;
      fn_ = nullptr;
    }
    if (err) std::rethrow_exception(err);
  }

 private:
  void worker_loop(size_t idx) {
    uint64_t seen = 0;
    for (;;) {
      size_t a = 0, b = 0;
      const std::function<void(size_t, size_t)>* fn = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu_);
        wake_cv_.wait(lock, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        if (idx >= parts_) continue;
        a = n_ * idx / parts_;
        b = n_ * (idx + 1) / parts_;
        fn = fn_;
      }
      std::exception_ptr err;
      try {
        (*fn)(a, b);
      } catch (...) {
        err = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (err && !error_) error_ = err;
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }

  void shutdown() noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_cv_.notify_all();
    for (auto& th : workers_)
      if (th.joinable()) th.join();
  }

  size_t max_threads_ = 1;
  std::mutex mu_;
  std::condition_variable wake_cv_;
  std::condition_variable done_cv_;
  uint64_t gen_ = 0;
  size_t active_ = 0;
  size_t n_ = 0;
  size_t parts_ = 0;
  const std::function<void(size_t, size_t)>* fn_ = nullptr;
  std::exception_ptr error_;
  bool stop_ = false;
  std::vector<std::thread> workers_;
};

uint16_t float_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);  // round to nearest even
  return uint16_t(u >> 16);
}

// ---- weights -------------------------------------------------------------------------------
// Drafts never reach the output (the target verifies each one), so the
// weights only drafting reads are NVFP4, made at load: a quarter of the bytes;
// only the acceptance can change.
// * Draft logits (each MTP step, the DFlash2 block) read the whole head, ~2 ms
//   in bf16 for Qwen3.5's 248K x 2560: an NVFP4 copy (the target keeps its own).
// * The MTP layer (projections and fc, bf16 in the checkpoints) runs only for
//   drafts and their context: quantized in place, the bf16 freed; a block
//   drafter too (quantize_block_drafter_weights): in NVFP4 beside an NVFP4
//   target, in Q8_0 (8-bit inputs, the block kernels) beside GGML block
//   weights or a BF16 target, DSpine's in NVFP4 there.
// Option draft_weights=exact keeps them exact.
void rows_to_bf16(const float* x, int ldx, int rows, int K, bf16* out, cudaStream_t st);

void quantize_in_place(flyx_bmodel* m, Lin& l, int N, int K) {
  if (l.q || !l.w || K % 64 != 0) return;  // already NVFP4, or not a shape the NVFP4 GEMMs take
  l.q = m->alloc<uint8_t>(size_t(N) * K / 2);
  l.s = m->alloc<uint8_t>(size_t(N) * K / 16);
  l.rs = m->alloc<float>(N);
  quantize_nvfp4(l.w, N, K, l.q, l.s, l.rs, 0);
  CK(cudaDeviceSynchronize());
  m->release(l.w, size_t(N) * K * sizeof(bf16));
  l.w = nullptr;
}

void quantize_packed_in_place(flyx_bmodel* m, Lin& l, int N, int K) {
  if (!l.packed() || l.q || K % 64 != 0) return;
  int total_rows = 0;
  size_t old_bytes = 0;
  for (const BlockMat& bm : l.blk) {
    if (!bm.data || bm.rows <= 0) return;
    total_rows += bm.rows;
    const BlockLayout lo = blocks_layout(bm.dtype);
    const size_t per_row = size_t(K) / size_t(lo.values);
    old_bytes += size_t(bm.rows) * per_row * (size_t(lo.stride) + size_t(lo.scales) * sizeof(uint16_t));
  }
  if (total_rows != N) return;
  // Never inflate ultra-low-bit (2-3 bpw) GGUF drafter weights into 4.5-bpw NVFP4.
  const size_t new_bytes = size_t(N) * K / 2 + size_t(N) * K / 16 + size_t(N) * sizeof(float);
  if (old_bytes * 10 < new_bytes * 9) return;
  uint8_t *q = nullptr, *s = nullptr;
  float* rs = nullptr;
  try {
    q = m->alloc<uint8_t>(size_t(N) * K / 2);
    s = m->alloc<uint8_t>(size_t(N) * K / 16);
    rs = m->alloc<float>(N);
  } catch (const OutOfDeviceMemory&) {
    if (q) m->release(q, size_t(N) * K / 2);
    if (s) m->release(s, size_t(N) * K / 16);
    if (rs) m->release(rs, size_t(N) * sizeof(float));
    return;
  }
  constexpr int kRun = 1024;
  std::vector<int> ids(kRun);
  struct DeviceGuard {
    void* p = nullptr;
    ~DeviceGuard() {
      if (p) cudaFree(p);
    }
  } dids_g, f32_g, bf16_g;
  CK(cudaMalloc(&dids_g.p, sizeof(int) * kRun));
  CK(cudaMalloc(&f32_g.p, sizeof(float) * size_t(kRun) * K));
  CK(cudaMalloc(&bf16_g.p, sizeof(bf16) * size_t(kRun) * K));
  int* dids = static_cast<int*>(dids_g.p);
  float* f32 = static_cast<float*>(f32_g.p);
  bf16* b16 = static_cast<bf16*>(bf16_g.p);
  int dst_row = 0;
  for (const BlockMat& bm : l.blk) {
    for (int r0 = 0; r0 < bm.rows; r0 += kRun) {
      const int n = std::min(kRun, bm.rows - r0);
      for (int i = 0; i < n; ++i) ids[size_t(i)] = r0 + i;
      CK(cudaMemcpy(dids, ids.data(), sizeof(int) * size_t(n), cudaMemcpyHostToDevice));
      embed_blocks(dids, bm, K, f32, n, 0);
      rows_to_bf16(f32, K, n, K, b16, 0);
      quantize_nvfp4(b16, n, K, q + size_t(dst_row) * K / 2, s + size_t(dst_row) * K / 16, rs + dst_row, 0);
      dst_row += n;
    }
  }
  CK(cudaDeviceSynchronize());
  for (const BlockMat& bm : l.blk) {
    const BlockLayout lo = blocks_layout(bm.dtype);
    const size_t per_row = size_t(K) / size_t(lo.values);
    m->release(const_cast<uint8_t*>(bm.data), size_t(bm.rows) * per_row * size_t(lo.stride));
    m->release(const_cast<uint16_t*>(bm.scales), size_t(bm.rows) * per_row * size_t(lo.scales) * sizeof(uint16_t));
  }
  l.blk.clear();
  l.q = q;
  l.s = s;
  l.rs = rs;
}

// A bf16 matrix as Q8_0 blocks whose inputs are 8 bits (FLYX_ACT_Q8), the bf16 freed: a block drafter's projections
// beside a target of GGML block weights or a BF16 one (half their bytes a round).
void q8_in_place(flyx_bmodel* m, Lin& l, int N, int K) {
  if (l.packed() || l.q || !l.w || K % 32 != 0) return;
  const size_t blocks = size_t(N) * (K / 32);
  uint8_t* payload = m->alloc<uint8_t>(blocks * 32);
  uint16_t* scales = m->alloc<uint16_t>(blocks);
  quantize_q8_0(l.w, N, K, payload, scales, 0);
  CK(cudaDeviceSynchronize());
  m->release(l.w, size_t(N) * K * sizeof(bf16));
  l.w = nullptr;
  l.blk = {BlockMat{payload, scales, FLYX_DT_GGML_Q8_0, N, true}};
}

namespace {

// Gives back the certified head's copy (what of it was allocated: null fields were not).  Only the certified head's:
// the copy option head_fp8 reads changes the logits and is the user's choice.
void release_cert(flyx_bmodel* m, HeadBlock* h) {
  if (h->w8) m->release(h->w8, size_t(h->vocab) * h->hidden);
  if (h->rs8) m->release(h->rs8, size_t(h->vocab) * sizeof(float));
  if (h->cert) m->release(h->cert, size_t(h->vocab) * 2 * sizeof(float));
  h->w8 = nullptr, h->rs8 = nullptr, h->cert = nullptr;
}

// Gives back what of a W12 copy was allocated (null fields were not); E: the escape list's length as allocated.
void release_w12(flyx_bmodel* m, W12* w, size_t N, size_t K, size_t E) {
  if (w->hi) m->release(w->hi, N * K);
  if (w->lo) m->release(w->lo, N * K / 2);
  if (w->e0) m->release(w->e0, N);
  if (w->esc_start) m->release(w->esc_start, (N + 1) * sizeof(int));
  if (w->esc_k) m->release(w->esc_k, E * sizeof(uint16_t));
  if (w->esc_v) m->release(w->esc_v, E * sizeof(uint16_t));
  *w = W12{};
}

// Gives back a 1-bit SignHead copy (bits and scales; null fields were not allocated).
void release_sign_head(flyx_bmodel* m, SignHead* sh, int N, int K) {
  const size_t tiled = size_t(N + 255) / 256 * 256;
  if (sh->bits) m->release(sh->bits, tiled * size_t(K) / 32 * sizeof(uint32_t));
  if (sh->scales) m->release(sh->scales, tiled * size_t(K) / 64 * sizeof(bf16));
  *sh = SignHead{};
}

}  // namespace

// Option head_fp8: the target's logits from an e4m3 copy of the head (half
// the bytes of bf16; the bf16 stays, it may be the embedding table).  The
// certified head's copy (head_cert) is for speed only: where the GPU has no room
// for it (device_malloc) the whole head chooses, the same tokens.
void make_head_fp8(flyx_bmodel* m) {
  auto& ko = m->opt.kernels;
  if (!ko.head_fp8 && !ko.head_cert) return;
  for (Block& b : m->blocks)  // the head blocks (m->head_block is only the one drafts read)
    if (auto* h = std::get_if<HeadBlock>(&b); h && h->w && h->hidden % 32 == 0 && !h->w8) {
      try {
        h->w8 = m->alloc<uint8_t>(size_t(h->vocab) * h->hidden);
        h->rs8 = m->alloc<float>(h->vocab);
        if (ko.head_cert && !ko.head_fp8) h->cert = m->alloc<float>(size_t(h->vocab) * 2);
      } catch (const OutOfDeviceMemory& e) {
        if (ko.head_fp8) throw;
        release_cert(m, h);
        ko.head_cert = false;
        if (m->verbose)
          std::fprintf(stderr, "flyx-cuda: no room for the certified head (%s); the whole head\n", e.what());
        continue;
      }
      quantize_fp8_rows(h->w, h->vocab, h->hidden, h->w8, h->rs8, 0);
      if (h->cert) {
        head_cert_coef(h->w, h->w8, h->rs8, h->vocab, h->hidden, h->cert, 0);
        if (m->verbose)
          std::fprintf(stderr, "flyx-cuda: certified head: an e4m3 copy of %.0f MB screens the %d x %d head\n",
                       double(h->vocab) * h->hidden / 1e6, h->vocab, h->hidden);
      }
    }
  CK(cudaDeviceSynchronize());
}

// The target's logits for 1..16 rows from a W12 copy of the head (SpecPilot's
// lossless 12-bit format, gemm_decode.cu): three quarters of the bytes, the
// same bits (every weight is decoded and compared here; a mismatch keeps the
// bf16 path).  The bf16 stays: it is the embedding table (tied) and the head
// of longer passes.  Option head_w12=0 leaves it out; so does a GPU without room
// for it (device_malloc).
void make_head_w12(flyx_bmodel* m) {
  if (!m->opt.kernels.head_w12) return;
  for (Block& b : m->blocks)
    if (auto* h = std::get_if<HeadBlock>(&b); h && h->w && h->hidden % 64 == 0 && h->hidden < 65536 && !h->w12.hi) {
      const int N = h->vocab, K = h->hidden;
      W12 w;
      int esc = 0;
      try {  // a copy for speed only: no room for it leaves the bf16 path
        w.hi = m->alloc<uint8_t>(size_t(N) * K);
        w.lo = m->alloc<uint8_t>(size_t(N) * K / 2);
        w.e0 = m->alloc<uint8_t>(N);
        w.esc_start = m->alloc<int>(size_t(N) + 1);
        std::vector<int> starts(size_t(N) + 1);
        esc = w12_encode(h->w, N, K, &w, starts.data(), 0);
        if (esc < 0) throw std::runtime_error("W12 encoding of the head failed");
        w.esc_k = m->alloc<uint16_t>(std::max(esc, 1));
        w.esc_v = m->alloc<uint16_t>(std::max(esc, 1));
      } catch (const OutOfDeviceMemory& e) {
        release_w12(m, &w, size_t(N), size_t(K), size_t(std::max(esc, 1)));
        if (m->verbose)  // its parts come one at a time: the one refused may be a small one after the planes
          std::fprintf(stderr, "flyx-cuda: no room for the W12 head's %.0f MiB (%s); the bf16 head\n",
                       (w12_plane_bytes(N, K) + size_t(N) * 5) / 1048576.0, e.what());
        continue;
      }
      const int bad = w12_finish(h->w, N, K, &w, 0);
      if (bad < 0) throw std::runtime_error("W12 check of the head failed");
      if (bad != 0) {
        std::fprintf(stderr, "flyx-cuda: the W12 head decodes %d groups differently; using bf16\n", bad);
        release_w12(m, &w, size_t(N), size_t(K), size_t(std::max(esc, 1)));  // does not happen on a correct build
        continue;
      }
      h->w12 = w;
      h->w12_escapes = std::max(esc, 1);
      if (m->verbose)
        std::fprintf(stderr, "flyx-cuda: head in W12: %.2f GB (bf16 %.2f GB), %d escapes\n",
                     (w12_plane_bytes(N, K) + size_t(N) * 5 + size_t(esc) * 4) / 1e9, size_t(N) * K * 2 / 1e9, esc);
    }
  CK(cudaDeviceSynchronize());
}

// The engine's call when not even one session fits (the ABI's trim), step by
// step until `bytes` are free: first the W12 copies of the head (the bf16 head
// they were made from computes the same bits), then the certified head's e4m3
// copy (the whole head chooses: the same tokens), then an NVFP4 copy drafts read
// (option draft_head=full; drafts from the bf16 head: other drafts, the same
// output, a draft being kept only when it is the target's choice), and finally
// the 1-bit SignHead draft copy (h.coarse; drafts fall back to the full head).
// Nothing that changes a result goes: head_fp8's copy, the user's choice, stays.
uint64_t trim(flyx_bmodel* m, uint64_t bytes) {
  try {
    CK(cudaDeviceSynchronize());
    const size_t before = m->bytes;
    auto heads = [&](const std::function<void(HeadBlock&)>& fn) {
      for (Block& b : m->blocks)
        if (auto* h = std::get_if<HeadBlock>(&b)) fn(*h);
    };
    heads([&](HeadBlock& h) {
      if (!h.w12.hi) return;
      release_w12(m, &h.w12, size_t(h.vocab), size_t(h.hidden), size_t(h.w12_escapes));
      h.w12_escapes = 0;
    });
    const size_t w12 = before - m->bytes;
    if (w12 && m->verbose)
      std::fprintf(stderr, "flyx-cuda: the W12 head given back (%.2f GB): the bf16 head, the same bits\n", w12 / 1e9);
    if (w12 >= bytes) return w12;
    auto& ko = m->opt.kernels;
    if (ko.head_cert && !ko.head_fp8) {
      heads([&](HeadBlock& h) { release_cert(m, &h); });
      ko.head_cert = false;  // and no session takes head_certify's scratch (rounds of several borrow a session's)
      if (m->verbose)
        std::fprintf(stderr, "flyx-cuda: the certified head given back (%.2f GB): the whole head chooses, the same "
                     "tokens\n", (before - m->bytes - w12) / 1e9);
    }
    if (before - m->bytes >= bytes) return before - m->bytes;
    heads([&](HeadBlock& h) {
      if (!h.draft.q) return;
      const size_t VH = size_t(h.vocab) * size_t(h.hidden);
      m->release(h.draft.q, VH / 2);
      m->release(h.draft.s, VH / 16);
      m->release(h.draft.rs, size_t(h.vocab) * sizeof(float));
      h.draft = Lin{};
      if (m->verbose) std::fprintf(stderr, "flyx-cuda: the NVFP4 draft head given back: drafts read the bf16 head\n");
    });
    if (before - m->bytes >= bytes) return before - m->bytes;
    const size_t pre_coarse = m->bytes;
    heads([&](HeadBlock& h) {
      if (h.coarse.bits || h.coarse.scales) release_sign_head(m, &h.coarse, h.vocab, h.hidden);
    });
    if (m->verbose && pre_coarse != m->bytes)
      std::fprintf(stderr, "flyx-cuda: 1-bit draft head given back (%.2f GB): drafts read the full head\n",
                   double(pre_coarse - m->bytes) / 1e9);
    return before - m->bytes;
  } catch (const std::exception&) {
    return 0;
  }
}

namespace {

enum class TargetQuant { kFloat, kNvfp4, kPacked };

// Target precision from its own MLP layers (excluding MTP's own MLP, which is quantized separately).
TargetQuant target_quant_kind(const flyx_bmodel* m) {
  bool fp4 = false, packed = false;
  const int mtp_mlp = m->mtp_block >= 0 ? block<MtpBlock>(m, m->mtp_block).mlp : -1;
  for (size_t i = 0; i < m->blocks.size(); ++i)
    if (const auto* f = std::get_if<MlpBlock>(&m->blocks[i]); f && int(i) != mtp_mlp) {
      fp4 = fp4 || f->w_gu.q;
      packed = packed || f->w_gu.packed();
    }
  return fp4 ? TargetQuant::kNvfp4 : packed ? TargetQuant::kPacked : TargetQuant::kFloat;
}

// Unified 1-bit SignHead builder for either a bf16 matrix (w) or a packed GGML BlockMat (wb).
// Returns false without throwing if the GPU lacks room (caller falls back cleanly to the full matrix).
bool make_sign_head_for(flyx_bmodel* m, const bf16* w, const BlockMat* wb, int N, int K, SignHead* out,
                        const char* what) {
  if ((!w && !(wb && wb->data)) || N < kDraftPickFloats || K % 128 != 0) return false;
  const size_t tiled = size_t(N + 255) / 256 * 256;  // whole tiles of 256 rows
  try {
    out->bits = m->alloc<uint32_t>(tiled * size_t(K) / 32);
    out->scales = m->alloc<bf16>(tiled * size_t(K) / 64);
  } catch (const OutOfDeviceMemory& e) {
    release_sign_head(m, out, N, K);
    if (m->verbose)
      std::fprintf(stderr, "flyx-cuda: no room for 1-bit %s (%.0f MiB: %s); reading full matrix\n", what,
                   sign_head_bytes(N, K) / 1048576.0, e.what());
    return false;
  }
  if (w) {
    make_sign_head(w, N, K, out, 0);
  } else {
    constexpr int kRun = 1024;
    std::vector<int> ids(kRun);
    struct DeviceGuard {
      void* p = nullptr;
      ~DeviceGuard() {
        if (p) cudaFree(p);
      }
    } dids_g, rows_g;
    if (cudaMalloc(&dids_g.p, sizeof(int) * kRun) != cudaSuccess ||
        cudaMalloc(&rows_g.p, sizeof(float) * size_t(kRun) * size_t(K)) != cudaSuccess) {
      cudaGetLastError();
      release_sign_head(m, out, N, K);
      return false;
    }
    int* dids = static_cast<int*>(dids_g.p);
    float* rows = static_cast<float*>(rows_g.p);
    for (int n0 = 0; n0 < N; n0 += kRun) {  // the default stream throughout: each run after the last
      const int n = std::min(kRun, N - n0);
      for (int i = 0; i < n; ++i) ids[size_t(i)] = n0 + i;
      CK(cudaMemcpy(dids, ids.data(), sizeof(int) * size_t(n), cudaMemcpyHostToDevice));
      embed_blocks(dids, *wb, K, rows, n, 0);
      make_sign_head_rows(rows, n0, n, K, out, 0);
    }
  }
  CK(cudaDeviceSynchronize());
  if (m->verbose)
    std::fprintf(stderr, "flyx-cuda: 1-bit %s (%.1f MB) for two-stage draft screening\n", what,
                 sign_head_bytes(N, K) / 1e6);
  return true;
}

// Quantize MTP's single sequential layer to NVFP4 (both BF16 and Q8_0/packed checkpoints).
void quantize_mtp_weights(flyx_bmodel* m) {
  if (m->mtp_block < 0) return;
  auto& b = block<MtpBlock>(m, m->mtp_block);
  auto& a = block<AttnBlock>(m, b.attn);
  auto& f = block<MlpBlock>(m, b.mlp);
  const size_t before = m->bytes;
  auto q_mtp = [&](Lin& l, int N, int K) {
    if (l.packed()) quantize_packed_in_place(m, l, N, K);
    else quantize_in_place(m, l, N, K);
  };
  q_mtp(b.fc, a.hidden, 2 * a.hidden);
  q_mtp(a.w_in, a.n_in(), a.hidden);
  q_mtp(a.w_out, a.hidden, a.nh * a.hd);
  q_mtp(f.w_gu, 2 * f.inter, f.hidden);
  q_mtp(f.w_down, f.hidden, f.inter);
  if (m->verbose && m->bytes != before)
    std::fprintf(stderr, "flyx-cuda: MTP layer -> NVFP4 (VRAM delta %+.2f GB)\n",
                 double(int64_t(m->bytes) - int64_t(before)) / 1e9);
}

// Shared block-drafter weight policy (DFlash2, DFlash and DSpine), leaving already-quantized block drafters intact
// (draft_weights=exact keeps them as the file has them): BF16 projections to NVFP4 beside an NVFP4 target (drafted
// against a BF16 target NVFP4 lost 9% acceptance, -2.6% overall; NVFP4 target +9.6%), to Q8_0 beside GGML block weights
// (the 27B's drafter reads 3.3 GB of bf16 a round), and beside a BF16 target to Q8_0, DSpine's to NVFP4.  Qwen3-8B's
// DSpine read 1.9 GB of bf16 a round, its layers as large as the target's: Q8_0 kept its tokens a round, NVFP4 cost at
// most 1% of them (its walk re-scores the head's top 16 through its bf16 bridge) and made a round 2-3% faster than
// Q8_0's on the eight prompt sets of docs/benchmarks.md section 6.
void quantize_block_drafter_weights(flyx_bmodel* m, TargetQuant tq) {
  if (m->dflash_block < 0) return;
  auto& d = block<DFlashBlock>(m, m->dflash_block);
  const bool fp4 = tq == TargetQuant::kNvfp4 || (tq == TargetQuant::kFloat && d.dspine);
  auto copy = [&](Lin& l, int N, int K) {
    if (fp4) quantize_in_place(m, l, N, K);
    else q8_in_place(m, l, N, K);
  };
  // DSpine: no convs or selector (their Lins empty: no copy); its bridge stays bf16 (blocks.h DSpineBridge).
  const int H = d.hidden, TG = d.conv_cols(), nqkv = (d.nh + 2 * d.nkv) * d.hd;
  for (int j = 0; j < d.layers(); ++j) {
    auto& l = block<DFlashLayer>(m, d.first_layer + j);
    copy(l.attn_proj, 2 * TG, H);
    copy(l.w_qkv, nqkv, H);
    copy(l.w_o, H, d.nh * d.hd);
    copy(l.mlp_proj, 2 * TG, H);
    copy(l.w_gu, 2 * d.inter, H);
    copy(l.w_down, H, d.inter);
  }
  copy(d.fc, H, d.n_taps * d.target_hidden);
  copy(d.ctx, d.ctx_width(), H);
  copy(d.sel_hid, d.sel_rank, H);
}

}  // namespace

void make_draft_weights(flyx_bmodel* m) {
  if (m->opt.exact_drafts || (m->mtp_block < 0 && m->dflash_block < 0)) return;
  // Quantize drafter layers first so freed VRAM is available for optional 1-bit SignHead copies.
  quantize_mtp_weights(m);
  quantize_block_drafter_weights(m, target_quant_kind(m));

  auto& h = block<HeadBlock>(m, m->head_block);
  const bool want_sign =
      !m->opt.full_draft_head && h.hidden % 128 == 0 && h.vocab >= kDraftPickFloats && (h.wb.data || h.hidden <= 4096);
  if (want_sign && (h.wb.data || h.w)) {
    make_sign_head_for(m, h.wb.data ? nullptr : h.w, h.wb.data ? &h.wb : nullptr, h.vocab, h.hidden, &h.coarse,
                       "draft head");
  } else if (!h.wb.data && h.w && h.hidden % 32 == 0) {
    h.draft.q = m->alloc<uint8_t>(size_t(h.vocab) * h.hidden / 2);
    h.draft.s = m->alloc<uint8_t>(size_t(h.vocab) * h.hidden / 16);
    h.draft.rs = m->alloc<float>(h.vocab);
    quantize_nvfp4(h.w, h.vocab, h.hidden, h.draft.q, h.draft.s, h.draft.rs, 0);
  }
}

namespace {

void ggml_nvfp4_planes_with(
    const flyx_tensor_view* v, const std::function<size_t(size_t)>& order, uint8_t* q, uint8_t* s,
    const std::function<void(size_t, const std::function<void(size_t, size_t)>&)>& dispatch) {
  const size_t N = size_t(v->shape[0]), per_row = size_t(v->shape[1]) / 64;  // blocks a row
  const auto* data = static_cast<const uint8_t*>(v->data);
  dispatch(N, [&](size_t r_begin, size_t r_end) {
    for (size_t r = r_begin; r < r_end; ++r) {
      const uint8_t* in = data + (order ? order(r) : r) * per_row * 36;
      for (size_t k = 0; k < per_row; ++k, in += 36) {
        const size_t b = r * per_row + k;  // the block's place in the planes
        const uint8_t* d = in;
        const uint8_t* qs = in + 4;
        for (int g = 0; g < 4; ++g) {
          if (d[g] == 0xFF)
            throw std::runtime_error(std::string(v->name) + ": an NVFP4 scale of 0xFF (no e4m3 holds it)");
          s[4 * b + size_t(g)] = d[g] == 0x7F ? 0 : uint8_t(d[g] & 0x7F);
          auto code = [&](int l) { return l < 8 ? qs[8 * g + l] & 15 : qs[8 * g + l - 8] >> 4; };
          for (int i = 0; i < 8; ++i)
            q[32 * b + 8 * size_t(g) + size_t(i)] = uint8_t(code(2 * i) | code(2 * i + 1) << 4);
        }
      }
    }
  });
}

}  // namespace

// GGML's NVFP4 as the NVFP4 planes the kernels read: per row the e2m1 codes two a byte in order
// (a GGML block keeps a group of 16 as the low nibbles of 8 bytes, then their high nibbles) and an e4m3 scale per 16,
// read as GGML reads it: the sign bit ignored, 0x7F (e4m3's NaN) zero. 0xFF, which GGML reads as 480, is no e4m3:
// refused (GGML's quantizer writes at most 0x7E). The values are GGML's exactly, with a row scale of 1.
void ggml_nvfp4_planes(const flyx_tensor_view* v, const std::function<size_t(size_t)>& order, uint8_t* q,
                       uint8_t* s) {
  ggml_nvfp4_planes_with(v, order, q, s, parallel_rows);
}

// ---- WeightLoader --------------------------------------------------------------------------
WeightLoader::WeightLoader(flyx_bmodel* model, const flyx_model_desc& desc)
    : m_(model), weights_(desc.weights), n_weights_(desc.n_weights) {}

WeightLoader::~WeightLoader() = default;

void WeightLoader::parallel_rows(size_t n, const std::function<void(size_t, size_t)>& fn) {
  const size_t threads = RowPool::default_threads();
  if (n <= 1 || threads <= 1) {
    if (n) fn(0, n);
    return;
  }
  if (!pool_) pool_ = std::make_unique<RowPool>(threads);
  pool_->run(n, fn);
}

void WeightLoader::finish() {
  pool_.reset();  // join workers while the last async staging copies finish on staging_->st
  if (staging_) {
    CK(cudaStreamSynchronize(staging_->st));
    staging_.reset();
  }
}

const flyx_tensor_view* WeightLoader::t(int w) const {
  if (w < 0 || w >= n_weights_ || !weights_[w].t) throw std::runtime_error("weight " + std::to_string(w) + " missing");
  return weights_[w].t;
}

size_t WeightLoader::numel(const flyx_tensor_view* v) {
  size_t n = 1;
  for (int i = 0; i < v->ndim; ++i) n *= size_t(v->shape[i]);
  return n;
}

bool WeightLoader::numbers(const flyx_tensor_view* v) {
  return v->dtype == FLYX_DT_BF16 || v->dtype == FLYX_DT_F16 || v->dtype == FLYX_DT_F32 ||
         v->dtype == FLYX_DT_F64 || flyx_block_values(v->dtype) != 0;
}

void WeightLoader::upload_staged(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (bytes < (size_t(1) << 20)) {
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    return;
  }
  if (!staging_) staging_ = std::make_unique<LoadStaging>();
  LoadStaging& S = *staging_;
  const auto* in = static_cast<const uint8_t*>(src);
  auto* out = static_cast<uint8_t*>(dst);
  constexpr size_t kStripe = size_t(1) << 20;  // 1 MiB stripes keep page faults and memory channels busy
  for (size_t off = 0; off < bytes; off += LoadStaging::kBytes) {
    const size_t n = std::min(LoadStaging::kBytes, bytes - off);
    const int b = S.next;
    S.next ^= 1;
    CK(cudaEventSynchronize(S.done[b]));
    const size_t stripes = (n + kStripe - 1) / kStripe;
    parallel_rows(stripes, [&](size_t a, size_t e) {
      const size_t p0 = a * kStripe;
      const size_t p1 = std::min(n, e * kStripe);
      std::memcpy(S.buf[b] + p0, in + off + p0, p1 - p0);
    });
    CK(cudaMemcpyAsync(out + off, S.buf[b], n, cudaMemcpyHostToDevice, S.st));
    CK(cudaEventRecord(S.done[b], S.st));
  }
}

void WeightLoader::put_bf16(bf16* dst, int w, const RowOrder& order, bool host) {
  const auto* v = t(w);
  const size_t n = numel(v);
  if (n == 0) return;
  const size_t bytes = n * sizeof(uint16_t);
  if (v->dtype == FLYX_DT_BF16 && !order) {
    if (host) {
      const auto* in = static_cast<const uint8_t*>(v->data);
      auto* out = reinterpret_cast<uint8_t*>(dst);
      constexpr size_t kStripe = size_t(1) << 20;
      const size_t stripes = (bytes + kStripe - 1) / kStripe;
      parallel_rows(stripes, [&](size_t a, size_t e) {
        const size_t p0 = a * kStripe;
        const size_t p1 = std::min(bytes, e * kStripe);
        std::memcpy(out + p0, in + p0, p1 - p0);
      });
    } else {
      upload_staged(dst, v->data, bytes);
    }
    return;
  }
  if (!numbers(v)) throw std::runtime_error(std::string(v->name) + ": a bf16 matrix from this dtype");
  const size_t rows = size_t(v->shape[0]), K = n / rows;
  auto convert_rows = [&](size_t r0, size_t count, uint16_t* out) {
    if (v->dtype == FLYX_DT_BF16) {
      const auto* in = static_cast<const uint16_t*>(v->data);
      for (size_t i = 0; i < count; ++i) {
        const size_t src_r = order ? order(r0 + i) : (r0 + i);
        std::memcpy(out + i * K, in + src_r * K, K * sizeof(uint16_t));
      }
      return;
    }
    for (size_t i = 0; i < count; ++i) {
      const size_t from = (order ? order(r0 + i) : (r0 + i)) * K;
      for (size_t k = 0; k < K; ++k)
        out[i * K + k] = float_to_bf16(static_cast<float>(flyx::ops::load(v, from + k)));
    }
  };
  if (host) {
    auto* out = reinterpret_cast<uint16_t*>(dst);
    parallel_rows(rows, [&](size_t a, size_t e) { convert_rows(a, e - a, out + a * K); });
    return;
  }
  const size_t row_bytes = K * sizeof(uint16_t);
  if (bytes < (size_t(1) << 20) || row_bytes > LoadStaging::kBytes) {
    std::vector<uint16_t> h(n);
    parallel_rows(rows, [&](size_t a, size_t e) { convert_rows(a, e - a, h.data() + a * K); });
    CK(cudaMemcpy(dst, h.data(), bytes, cudaMemcpyHostToDevice));
    return;
  }
  if (!staging_) staging_ = std::make_unique<LoadStaging>();
  LoadStaging& S = *staging_;
  const size_t chunk = std::max<size_t>(1, LoadStaging::kBytes / row_bytes);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    const size_t nr = std::min(chunk, rows - r0);
    const int b = S.next;
    S.next ^= 1;
    CK(cudaEventSynchronize(S.done[b]));
    auto* hp = reinterpret_cast<uint16_t*>(S.buf[b]);
    parallel_rows(nr, [&](size_t a, size_t e) { convert_rows(r0 + a, e - a, hp + a * K); });
    CK(cudaMemcpyAsync(dst + r0 * K, hp, nr * row_bytes, cudaMemcpyHostToDevice, S.st));
    CK(cudaEventRecord(S.done[b], S.st));
  }
}

bf16* WeightLoader::mat(int w) {
  auto it = mats_.find(w);
  if (it != mats_.end()) return it->second;
  bf16* p = m_->alloc<bf16>(numel(t(w)));
  put_bf16(p, w);
  return mats_[w] = p;
}

bf16* WeightLoader::mat_host(int w) {
  bf16* p = m_->host_alloc<bf16>(numel(t(w)));
  put_bf16(p, w, {}, /*host=*/true);
  return p;
}

bf16* WeightLoader::stack(const std::vector<int>& ws, int* rows_out, const std::vector<RowOrder>& orders) {
  size_t total = 0;
  for (int w : ws) total += numel(t(w));
  bf16* p = m_->alloc<bf16>(total);
  size_t off = 0;
  int rows = 0;
  for (size_t i = 0; i < ws.size(); ++i) {
    const int w = ws[i];
    put_bf16(p + off, w, i < orders.size() ? orders[i] : RowOrder{});
    off += numel(t(w));
    rows += int(t(w)->shape[0]);
  }
  if (rows_out) *rows_out = rows;
  return p;
}

bool WeightLoader::packable(int w) const {
  const auto* v = t(w);
  return m_->opt.packed_blocks && v->ndim == 2 && blocks_supported(v->dtype, int(v->shape[1]));
}

Lin WeightLoader::lin_stack(const std::vector<int>& ws, int* rows_out, const std::vector<RowOrder>& orders) {
  Lin L;
  bool all_packable = true;
  for (int w : ws) all_packable = all_packable && packable(w) && t(w)->shape[1] == t(ws[0])->shape[1];
  if (all_packable) {
    int rows = 0;
    for (int w : ws)
      if (weights_[w].activations != weights_[ws[0]].activations)
        throw std::runtime_error(std::string(t(w)->name) + ": a stack of tensors that round their inputs apart");
    for (size_t i = 0; i < ws.size();) {
      size_t end = i + 1;
      while (end < ws.size() && t(ws[end])->dtype == t(ws[i])->dtype) ++end;
      std::vector<int> run(ws.begin() + i, ws.begin() + end);
      std::vector<RowOrder> run_orders;
      for (size_t j = i; j < end; ++j) run_orders.push_back(j < orders.size() ? orders[j] : RowOrder{});
      L.blk.push_back(run.size() == 1 ? pack(run[0], run_orders[0]) : pack_run(run, run_orders));
      rows += L.blk.back().rows;
      i = end;
    }
    if (rows_out) *rows_out = rows;
    return L;
  }
  // NVFP4, ModelOpt's or GGML's (repacked: ggml_nvfp4_planes).
  auto is_fp4 = [&](int w) { return t(w)->dtype == FLYX_DT_NVFP4 || t(w)->dtype == FLYX_DT_GGML_NVFP4; };
  const bool fp4 = is_fp4(ws[0]);
  for (int w : ws)
    if (is_fp4(w) != fp4) throw std::runtime_error(std::string(t(w)->name) + ": mixed weight formats in one stack");
  if (!fp4) {
    L.w = stack(ws, rows_out, orders);
    return L;
  }
  size_t rows = 0;
  const size_t K = size_t(t(ws[0])->shape[1]);
  for (int w : ws) {
    if (size_t(t(w)->shape[1]) != K || K % 32)
      throw std::runtime_error(std::string(t(w)->name) + ": NVFP4 K must match and be a multiple of 32");
    rows += size_t(t(w)->shape[0]);
  }
  L.q = m_->alloc<uint8_t>(rows * K / 2);
  L.s = m_->alloc<uint8_t>(rows * K / 16);
  L.rs = m_->alloc<float>(rows);
  size_t r0 = 0;
  for (size_t i = 0; i < ws.size(); ++i) {
    const int w = ws[i];
    const size_t nr = size_t(t(w)->shape[0]);
    const RowOrder none, &order = i < orders.size() ? orders[i] : none;
    if (t(w)->dtype == FLYX_DT_GGML_NVFP4) {  // repacked on the host: in the row order the kernels want
      if (K % 64) throw std::runtime_error(std::string(t(w)->name) + ": GGML NVFP4 K must be a multiple of 64");
      std::vector<uint8_t> q(nr * K / 2), sc(nr * K / 16);
      ggml_nvfp4_planes_with(t(w), order, q.data(), sc.data(),
                             [&](size_t n, const auto& fn) { parallel_rows(n, fn); });
      upload_staged(L.q + r0 * K / 2, q.data(), q.size());
      upload_staged(L.s + r0 * K / 16, sc.data(), sc.size());
      std::vector<float> one(nr, 1.f);
      CK(cudaMemcpy(L.rs + r0, one.data(), nr * 4, cudaMemcpyHostToDevice));
      r0 += nr;
      continue;
    }
    if (order) throw std::runtime_error(std::string(t(w)->name) + ": NVFP4 rows are uploaded as stored");
    const auto* sc = weights_[w].scale;
    const auto* s2 = weights_[w].scale2;
    if (!sc || !s2 || sc->dtype != FLYX_DT_F8E4M3 || s2->dtype != FLYX_DT_F32)
      throw std::runtime_error(std::string(t(w)->name) + ": bad NVFP4 scales");
    upload_staged(L.q + r0 * K / 2, t(w)->data, nr * K / 2);
    upload_staged(L.s + r0 * K / 16, sc->data, nr * K / 16);
    std::vector<float> g(nr, static_cast<const float*>(s2->data)[0]);
    CK(cudaMemcpy(L.rs + r0, g.data(), nr * 4, cudaMemcpyHostToDevice));
    r0 += nr;
  }
  if (rows_out) *rows_out = int(rows);
  return L;
}

std::vector<float> WeightLoader::host_f32(int w) const {
  const auto* v = t(w);
  if (!numbers(v)) throw std::runtime_error(std::string(v->name) + ": unsupported dtype");
  std::vector<float> out(numel(v));
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(flyx::ops::load(v, i));
  return out;
}

float* WeightLoader::vecs(const std::vector<int>& ws) {
  std::vector<float> all;
  for (int w : ws) {
    auto h = host_f32(w);
    all.insert(all.end(), h.begin(), h.end());
  }
  return put(all);
}

float* WeightLoader::put(const std::vector<float>& values) {
  float* p = m_->alloc<float>(values.size());
  CK(cudaMemcpy(p, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
  return p;
}

VisionVec WeightLoader::vision_put(const std::vector<float>& values) {
  if (values.empty()) return {};
  const size_t bytes = values.size() * sizeof(float);
  if (m_->opt.vision == FLYX_VISION_DEV_DISK) {
    const uint64_t off = m_->vision_disk.append(values.data(), bytes);
    m_->vision_disk_bytes += bytes;
    return VisionVec{nullptr, values.size(), false, true, off};
  }
  if (m_->opt.vision == FLYX_VISION_DEV_HOST) {
    float* p = m_->host_pinned<float>(values.size());
    std::memcpy(p, values.data(), bytes);
    m_->vision_host_bytes += bytes;
    return VisionVec{p, values.size(), true, false, 0};
  }
  float* p = put(values);
  m_->vision_dev_bytes += bytes;
  return VisionVec{p, values.size(), false, false, 0};
}

VisionLin WeightLoader::vision_lin_stack(const std::vector<int>& ws, int* rows_out) {
  if (ws.empty()) return {};
  const bool disk = m_->opt.vision == FLYX_VISION_DEV_DISK;
  const bool host = m_->opt.vision == FLYX_VISION_DEV_HOST;
  const int vq = m_->opt.vision_quant;
  const size_t K = size_t(t(ws[0])->shape[1]);
  size_t total_rows = 0;
  for (int w : ws) {
    if (t(w)->ndim != 2 || size_t(t(w)->shape[1]) != K)
      throw std::runtime_error(std::string(t(w)->name) + ": vision linear shape mismatch");
    total_rows += size_t(t(w)->shape[0]);
  }
  if (rows_out) *rows_out = int(total_rows);

  VisionLin vl;
  vl.N = int(total_rows);
  vl.K = int(K);
  vl.K_pad = int(K);
  vl.on_host = host;
  vl.on_disk = disk;

  auto align256 = [](size_t b) { return (b + 255) / 256 * 256; };
  auto write_disk_blob = [&](const std::vector<std::pair<const void*, size_t>>& parts, VisionLin::DiskKind kind) {
    size_t total = 0;
    for (const auto& [ptr, nb] : parts) total += align256(nb);
    std::vector<uint8_t> blob(total, 0);
    size_t off = 0;
    for (const auto& [ptr, nb] : parts) {
      if (nb) std::memcpy(blob.data() + off, ptr, nb);
      off += align256(nb);
    }
    vl.disk_kind = kind;
    vl.disk_off = m_->vision_disk.append(blob.data(), blob.size());
    vl.disk_bytes = blob.size();
    m_->vision_disk_bytes += blob.size();
  };
  auto record_bytes = [&](size_t b) {
    if (host) m_->vision_host_bytes += b;
    else if (!disk) m_->vision_dev_bytes += b;
  };

  // 1. Pre-quantized GGML block weights (e.g. mmproj-Q8_0.gguf / Q4_K / Q6_K): keep packed as-is.
  bool all_packable = true;
  for (int w : ws) all_packable = all_packable && packable(w);
  if (all_packable) {
    for (int w : ws)
      if (weights_[w].activations != weights_[ws[0]].activations)
        throw std::runtime_error(std::string(t(w)->name) + ": mixed activation rounding in vision stack");
    std::vector<std::vector<uint8_t>> disk_datas;
    std::vector<std::vector<uint16_t>> disk_scales;
    std::vector<std::pair<const void*, size_t>> disk_parts;
    for (size_t i = 0; i < ws.size();) {
      size_t end = i + 1;
      while (end < ws.size() && t(ws[end])->dtype == t(ws[i])->dtype) ++end;
      std::vector<int> run(ws.begin() + i, ws.begin() + end);
      const BlockLayout lo = blocks_layout(t(run[0])->dtype);
      const size_t per_row = K / size_t(lo.values);
      const size_t row_bytes = per_row * size_t(lo.stride), row_scales = per_row * size_t(lo.scales);
      size_t run_rows = 0;
      for (int w : run) run_rows += size_t(t(w)->shape[0]);
      if (disk) {
        blocks_init();
        disk_datas.emplace_back(run_rows * row_bytes);
        disk_scales.emplace_back(run_rows * row_scales);
        size_t r0 = 0;
        for (int w : run) {
          pack_rows(w, {}, disk_datas.back().data() + r0 * row_bytes, disk_scales.back().data() + r0 * row_scales,
                    /*host=*/true);
          r0 += size_t(t(w)->shape[0]);
        }
        vl.w.blk.push_back(BlockMat{nullptr, nullptr, t(run[0])->dtype, int(run_rows),
                                    weights_[run[0]].activations == FLYX_ACT_Q8});
        disk_parts.emplace_back(disk_datas.back().data(), disk_datas.back().size());
        disk_parts.emplace_back(disk_scales.back().data(), disk_scales.back().size() * sizeof(uint16_t));
      } else if (host) {
        blocks_init();
        uint8_t* data = m_->host_pinned<uint8_t>(run_rows * row_bytes);
        uint16_t* scales = m_->host_pinned<uint16_t>(run_rows * row_scales);
        size_t r0 = 0;
        for (int w : run) {
          pack_rows(w, {}, data + r0 * row_bytes, scales + r0 * row_scales, /*host=*/true);
          r0 += size_t(t(w)->shape[0]);
        }
        vl.w.blk.push_back(BlockMat{data, scales, t(run[0])->dtype, int(run_rows),
                                    weights_[run[0]].activations == FLYX_ACT_Q8});
      } else {
        vl.w.blk.push_back(run.size() == 1 ? pack(run[0]) : pack_run(run, {}));
      }
      record_bytes(run_rows * (row_bytes + row_scales * sizeof(uint16_t)));
      i = end;
    }
    if (disk) write_disk_blob(disk_parts, VisionLin::kDiskPacked);
    return vl;
  }

  // 2. Pre-quantized NVFP4 weights: keep NVFP4 planes as-is.
  auto is_fp4 = [&](int w) { return t(w)->dtype == FLYX_DT_NVFP4 || t(w)->dtype == FLYX_DT_GGML_NVFP4; };
  const bool fp4 = is_fp4(ws[0]);
  for (int w : ws)
    if (is_fp4(w) != fp4) throw std::runtime_error(std::string(t(w)->name) + ": mixed weight formats in vision stack");
  if (fp4) {
    for (int w : ws)
      if (weights_[w].activations != weights_[ws[0]].activations)
        throw std::runtime_error(std::string(t(w)->name) + ": mixed activation rounding in vision stack");
    vl.a4 = weights_[ws[0]].activations == FLYX_ACT_NVFP4;
    if (!host && !disk) {
      vl.w = lin_stack(ws);
      record_bytes(total_rows * K / 2 + total_rows * K / 16 + total_rows * sizeof(float));
      return vl;
    }
    if (K % 32) throw std::runtime_error(std::string(t(ws[0])->name) + ": NVFP4 K must be a multiple of 32");
    std::vector<uint8_t> q_tmp, s_tmp;
    std::vector<float> rs_tmp;
    uint8_t *q_dst = nullptr, *s_dst = nullptr;
    float* rs_dst = nullptr;
    if (disk) {
      q_tmp.resize(total_rows * K / 2);
      s_tmp.resize(total_rows * K / 16);
      rs_tmp.resize(total_rows);
      q_dst = q_tmp.data();
      s_dst = s_tmp.data();
      rs_dst = rs_tmp.data();
    } else {
      q_dst = vl.w.q = m_->host_pinned<uint8_t>(total_rows * K / 2);
      s_dst = vl.w.s = m_->host_pinned<uint8_t>(total_rows * K / 16);
      rs_dst = vl.w.rs = m_->host_pinned<float>(total_rows);
    }
    size_t r0 = 0;
    for (int w : ws) {
      const size_t nr = size_t(t(w)->shape[0]);
      if (t(w)->dtype == FLYX_DT_GGML_NVFP4) {
        if (K % 64) throw std::runtime_error(std::string(t(w)->name) + ": GGML NVFP4 K must be a multiple of 64");
        ggml_nvfp4_planes_with(t(w), {}, q_dst + r0 * K / 2, s_dst + r0 * K / 16,
                               [&](size_t n, const auto& fn) { parallel_rows(n, fn); });
        std::fill_n(rs_dst + r0, nr, 1.f);
      } else {
        const auto* sc = weights_[w].scale;
        const auto* s2 = weights_[w].scale2;
        if (!sc || !s2 || sc->dtype != FLYX_DT_F8E4M3 || s2->dtype != FLYX_DT_F32)
          throw std::runtime_error(std::string(t(w)->name) + ": bad NVFP4 scales");
        std::memcpy(q_dst + r0 * K / 2, t(w)->data, nr * K / 2);
        std::memcpy(s_dst + r0 * K / 16, sc->data, nr * K / 16);
        std::fill_n(rs_dst + r0, nr, static_cast<const float*>(s2->data)[0]);
      }
      r0 += nr;
    }
    if (disk) {
      write_disk_blob({{q_tmp.data(), q_tmp.size()}, {s_tmp.data(), s_tmp.size()},
                       {rs_tmp.data(), rs_tmp.size() * sizeof(float)}},
                      VisionLin::kDiskNvfp4);
    } else {
      record_bytes(total_rows * K / 2 + total_rows * K / 16 + total_rows * sizeof(float));
    }
    return vl;
  }

  // Helper to read row `r` of a floating-point tensor into float32 (fast path for BF16).
  auto read_row_f32 = [](const flyx_tensor_view* v, size_t r, size_t cols, float* dst) {
    if (v->dtype == FLYX_DT_BF16) {
      const auto* in = static_cast<const uint16_t*>(v->data) + r * cols;
      for (size_t k = 0; k < cols; ++k) {
        const uint32_t u = uint32_t(in[k]) << 16;
        std::memcpy(dst + k, &u, 4);
      }
    } else if (v->dtype == FLYX_DT_F32) {
      const auto* in = static_cast<const float*>(v->data) + r * cols;
      std::memcpy(dst, in, cols * sizeof(float));
    } else {
      for (size_t k = 0; k < cols; ++k) dst[k] = static_cast<float>(flyx::ops::load(v, r * cols + k));
    }
  };

  // 3. On-the-fly Q8_0 quantization (8.5 bpw, ~46% smaller, zero-padded to K % 32 == 0).
  if (vq == FLYX_VISION_QUANT_Q8) {
    blocks_init();
    const size_t K_pad = (K + 31) / 32 * 32;
    const size_t blocks_per_row = K_pad / 32;
    vl.K_pad = int(K_pad);
    const size_t pay_bytes = total_rows * K_pad;
    const size_t sc_elems = total_rows * blocks_per_row;
    uint8_t* pay_host = nullptr;
    uint16_t* sc_host = nullptr;
    std::vector<uint8_t> pay_tmp;
    std::vector<uint16_t> sc_tmp;
    if (host) {
      pay_host = m_->host_pinned<uint8_t>(pay_bytes);
      sc_host = m_->host_pinned<uint16_t>(sc_elems);
    } else {
      pay_tmp.resize(pay_bytes);
      sc_tmp.resize(sc_elems);
      pay_host = pay_tmp.data();
      sc_host = sc_tmp.data();
    }
    size_t r0 = 0;
    for (int w : ws) {
      const auto* v = t(w);
      if (!numbers(v)) throw std::runtime_error(std::string(v->name) + ": unsupported vision weight dtype");
      const size_t nr = size_t(v->shape[0]);
      parallel_rows(nr, [&](size_t a, size_t e) {
        std::vector<float> row(K);
        for (size_t r = a; r < e; ++r) {
          read_row_f32(v, r, K, row.data());
          flyx::ops::quantize_row_q8_0(row.data(), int(K), int(K_pad), pay_host + (r0 + r) * K_pad,
                                       sc_host + (r0 + r) * blocks_per_row);
        }
      });
      r0 += nr;
    }
    if (disk) {
      vl.w.blk = {BlockMat{nullptr, nullptr, FLYX_DT_GGML_Q8_0, int(total_rows), true}};
      write_disk_blob({{pay_tmp.data(), pay_bytes}, {sc_tmp.data(), sc_elems * sizeof(uint16_t)}},
                      VisionLin::kDiskPacked);
    } else if (host) {
      vl.w.blk = {BlockMat{pay_host, sc_host, FLYX_DT_GGML_Q8_0, int(total_rows), true}};
      record_bytes(pay_bytes + sc_elems * sizeof(uint16_t));
    } else {
      uint8_t* d_pay = m_->alloc<uint8_t>(pay_bytes);
      uint16_t* d_sc = m_->alloc<uint16_t>(sc_elems);
      upload_staged(d_pay, pay_host, pay_bytes);
      upload_staged(d_sc, sc_host, sc_elems * sizeof(uint16_t));
      vl.w.blk = {BlockMat{d_pay, d_sc, FLYX_DT_GGML_Q8_0, int(total_rows), true}};
      record_bytes(pay_bytes + sc_elems * sizeof(uint16_t));
    }
    return vl;
  }

  // 4. On-the-fly NVFP4 quantization (4.5 bpw, ~70% smaller, zero-padded to K % 64 == 0).
  if (vq == FLYX_VISION_QUANT_NVFP4) {
    const size_t K_pad = (K + 63) / 64 * 64;
    vl.K_pad = int(K_pad);
    const size_t q_bytes = total_rows * K_pad / 2;
    const size_t s_bytes = total_rows * K_pad / 16;
    uint8_t *q_host = nullptr, *s_host = nullptr;
    float* rs_host = nullptr;
    std::vector<uint8_t> q_tmp, s_tmp;
    std::vector<float> rs_tmp;
    if (host) {
      q_host = m_->host_pinned<uint8_t>(q_bytes);
      s_host = m_->host_pinned<uint8_t>(s_bytes);
      rs_host = m_->host_pinned<float>(total_rows);
    } else {
      q_tmp.resize(q_bytes);
      s_tmp.resize(s_bytes);
      rs_tmp.resize(total_rows);
      q_host = q_tmp.data();
      s_host = s_tmp.data();
      rs_host = rs_tmp.data();
    }
    size_t r0 = 0;
    for (int w : ws) {
      const auto* v = t(w);
      if (!numbers(v)) throw std::runtime_error(std::string(v->name) + ": unsupported vision weight dtype");
      const size_t nr = size_t(v->shape[0]);
      parallel_rows(nr, [&](size_t a, size_t e) {
        std::vector<float> row(K);
        for (size_t r = a; r < e; ++r) {
          read_row_f32(v, r, K, row.data());
          flyx::ops::quantize_row_nvfp4(row.data(), int(K), int(K_pad), q_host + (r0 + r) * (K_pad / 2),
                                        s_host + (r0 + r) * (K_pad / 16), rs_host + (r0 + r));
        }
      });
      r0 += nr;
    }
    if (disk) {
      write_disk_blob({{q_tmp.data(), q_bytes}, {s_tmp.data(), s_bytes}, {rs_tmp.data(), total_rows * sizeof(float)}},
                      VisionLin::kDiskNvfp4);
    } else if (host) {
      vl.w.q = q_host;
      vl.w.s = s_host;
      vl.w.rs = rs_host;
      record_bytes(q_bytes + s_bytes + total_rows * sizeof(float));
    } else {
      vl.w.q = m_->alloc<uint8_t>(q_bytes);
      vl.w.s = m_->alloc<uint8_t>(s_bytes);
      vl.w.rs = m_->alloc<float>(total_rows);
      upload_staged(vl.w.q, q_host, q_bytes);
      upload_staged(vl.w.s, s_host, s_bytes);
      CK(cudaMemcpy(vl.w.rs, rs_host, total_rows * sizeof(float), cudaMemcpyHostToDevice));
      record_bytes(q_bytes + s_bytes + total_rows * sizeof(float));
    }
    return vl;
  }

  // 5. Exact BF16 (disk, host pinned, or GPU resident).
  if (disk) {
    std::vector<bf16> tmp(total_rows * K);
    size_t off = 0;
    for (int w : ws) {
      put_bf16(tmp.data() + off, w, {}, /*host=*/true);
      off += numel(t(w));
    }
    write_disk_blob({{tmp.data(), tmp.size() * sizeof(bf16)}}, VisionLin::kDiskBf16);
  } else if (host) {
    bf16* p = m_->host_pinned<bf16>(total_rows * K);
    size_t off = 0;
    for (int w : ws) {
      put_bf16(p + off, w, {}, /*host=*/true);
      off += numel(t(w));
    }
    vl.w.w = p;
    record_bytes(total_rows * K * sizeof(bf16));
  } else {
    vl.w.w = stack(ws);
    record_bytes(total_rows * K * sizeof(bf16));
  }
  return vl;
}

// ---- GGML block weights, packed ------------------------------------------------------------
// The file's blocks become payloads and scales (blocks_pack) through host
// buffers of a few megabytes, in the row order the kernels want; the device
// then holds the bytes the file has, not the 16 bits a weight of bf16.
BlockMat WeightLoader::pack(int w, const RowOrder& order) {
  if (!order)
    if (auto it = packs_.find(w); it != packs_.end()) return it->second;
  const BlockMat b = pack_run({w}, {order});
  if (!order) packs_[w] = b;
  return b;
}

// Tensors of one block type and width, one after the other in one allocation: one matrix of all their rows
// (host: in host memory, host_alloc).
BlockMat WeightLoader::pack_run(const std::vector<int>& ws, const std::vector<RowOrder>& orders, bool host) {
  blocks_init();  // the decoders' tables, before any round is captured
  const auto* first = t(ws[0]);
  const BlockLayout l = blocks_layout(first->dtype);
  const size_t per_row = size_t(first->shape[1]) / size_t(l.values);
  const size_t row_bytes = per_row * size_t(l.stride), row_scales = per_row * size_t(l.scales);
  size_t total = 0;
  for (int w : ws) {
    if (t(w)->dtype != first->dtype || t(w)->shape[1] != first->shape[1])
      throw std::runtime_error(std::string(t(w)->name) + ": a run of packed tensors differs in block type or width");
    total += size_t(t(w)->shape[0]);
  }
  for (int w : ws)
    if (weights_[w].activations != weights_[ws[0]].activations)
      throw std::runtime_error(std::string(t(w)->name) + ": a run of packed tensors differs in how it rounds inputs");
  auto* data = host ? m_->host_alloc<uint8_t>(total * row_bytes) : m_->alloc<uint8_t>(total * row_bytes);
  auto* scales = host ? m_->host_alloc<uint16_t>(total * row_scales) : m_->alloc<uint16_t>(total * row_scales);
  size_t r_first = 0;
  for (size_t i = 0; i < ws.size(); ++i) {
    pack_rows(ws[i], i < orders.size() ? orders[i] : RowOrder{}, data + r_first * row_bytes,
              scales + r_first * row_scales, host);
    r_first += size_t(t(ws[i])->shape[0]);
  }
  return BlockMat{data, scales, first->dtype, int(total), weights_[ws[0]].activations == FLYX_ACT_Q8};
}

void parallel_rows(size_t n, const std::function<void(size_t, size_t)>& fn) {
  const size_t parts = std::min(RowPool::default_threads(), n);
  if (parts <= 1) {
    if (n) fn(0, n);
    return;
  }
  RowPool pool(parts);
  pool.run(n, fn);
}

LoadStaging::LoadStaging() {
  for (int i = 0; i < 2; ++i) {
    CK(cudaHostAlloc(&buf[i], kBytes, cudaHostAllocDefault));
    CK(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
  }
  CK(cudaStreamCreate(&st));  // a blocking stream: work on the legacy stream after waits for its copies
}

LoadStaging::~LoadStaging() {
  if (st) cudaStreamSynchronize(st);
  for (int i = 0; i < 2; ++i) {
    if (done[i]) cudaEventDestroy(done[i]);
    if (buf[i]) cudaFreeHost(buf[i]);
  }
  if (st) cudaStreamDestroy(st);
}

// One tensor's rows packed at data and scales: laid out by parallel_rows, into host memory directly (host) or into a
// staging buffer whose copy to the device runs while the next buffer is laid out.
void WeightLoader::pack_rows(int w, const RowOrder& order, uint8_t* data, uint16_t* scales, bool host) {
  const auto* v = t(w);
  const BlockLayout l = blocks_layout(v->dtype);
  const size_t rows = size_t(v->shape[0]), per_row = size_t(v->shape[1]) / size_t(l.values);
  const size_t row_bytes = per_row * size_t(l.stride), row_scales = per_row * size_t(l.scales);
  const auto* file = static_cast<const uint8_t*>(v->data);
  const size_t file_row = per_row * size_t(l.bytes);
  const int dtype = v->dtype;
  // Rows r0 .. r0 + n - 1 of the upload at hp, hs.
  auto lay_out = [&](size_t r0, size_t n, uint8_t* hp, uint16_t* hs) {
    if (!order) {
      blocks_pack(dtype, file + r0 * file_row, n * per_row, hp, hs);
      return;
    }
    for (size_t r = 0; r < n; ++r)
      blocks_pack(dtype, file + order(r0 + r) * file_row, per_row, hp + r * row_bytes, hs + r * row_scales);
  };
  times_.file_bytes += rows * file_row;
  if (host) {  // page-locked host memory the kernels read in place: no copy
    const double t0 = load_clock();
    parallel_rows(rows, [&](size_t a, size_t b) { lay_out(a, b - a, data + a * row_bytes, scales + a * row_scales); });
    times_.pack_s += load_clock() - t0;
    return;
  }
  if (!staging_) staging_ = std::make_unique<LoadStaging>();
  LoadStaging& S = *staging_;
  const size_t scale_bytes = row_scales * sizeof(uint16_t);
  const size_t chunk = std::max<size_t>(1, (LoadStaging::kBytes - 16) / (row_bytes + scale_bytes));
  if (row_bytes + scale_bytes + 16 > LoadStaging::kBytes)
    throw std::runtime_error(std::string(v->name) + ": a row larger than the load's staging buffer");
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    const size_t n = std::min(chunk, rows - r0);
    const int b = S.next;
    S.next ^= 1;
    const double t0 = load_clock();
    CK(cudaEventSynchronize(S.done[b]));  // the buffer's last copy is done
    const double t1 = load_clock();
    uint8_t* hp = S.buf[b];
    auto* hs = reinterpret_cast<uint16_t*>(hp + (n * row_bytes + 15) / 16 * 16);
    parallel_rows(n, [&](size_t a, size_t e) { lay_out(r0 + a, e - a, hp + a * row_bytes, hs + a * row_scales); });
    CK(cudaMemcpyAsync(data + r0 * row_bytes, hp, n * row_bytes, cudaMemcpyHostToDevice, S.st));
    CK(cudaMemcpyAsync(scales + r0 * row_scales, hs, n * scale_bytes, cudaMemcpyHostToDevice, S.st));
    CK(cudaEventRecord(S.done[b], S.st));
    times_.copy_s += t1 - t0;
    times_.pack_s += load_clock() - t1;
  }
}

}  // namespace flyx::cuda
