# Numerics contract

Every CUDA kernel promises three things: how close it is to the float64
reference (`ops/<op>/ref.h`, `ops/sampling/sampling.h`), which other paths
give its result bit for bit, and the coding rules that keep those bits
equal. The table below lists every public launcher of
`backends/cuda/kernels/kernels.cuh`. `tools/check_numerics.py` (a ctest) fails
when a launcher is missing here or a listed test does not exist in
`tests/backends/cuda/op_test*.cu` (a file a family of kernels). A new kernel
gets its row in the same change.

## Bits that must agree

| class | what | why |
|---|---|---|
| R: rows | a row's result does not depend on which or how many rows run with it; a wide lookup round (17..64 rows, the `wide` backend option) gives every row a 16-row round's bits (NVFP4 GEMMs by gemm_fp4_rows, the rest 16 rows at a time); a tree round of 17..64 rows the same, each row at its depth's position over its ancestors' keys (attention_split, never attention_flash), its choice, penalties and relaxed rules 16 rows at a time (choose_rows, penalize_rows, row_stats, child_logits) | plain decoding and verification agree, so speculation never changes the output (rule 4) |
| C: configurations | every compiled configuration of a kernel gives the same bits (bf16 skinny tiles; NVFP4 per K partition; prefill GEMM tiles) | a tuning table may pick any of them per GPU, shape and row count |
| F: fused | a fused producer and the GEMM reading its staged input give the generic producer's and GEMM's bits; one-kernel GDN round = its four kernels | region overrides change speed, never results (rule 2) |
| P: passes | a round's GDN recurrence gives the same state written in place, folded later (commit, rollback) or row by row (decode) | a fold reproduces the verification that produced the outputs |
| K: chunks | a prefill row's result does not depend on where its pass ends, nor on where it starts when the pass starts at a multiple of the backend's prefill alignment (32 positions with the chunked GDN recurrence, else 1), among passes of more than 16 rows (a pass of 16 or fewer computes as a round, with a round's kernels, and for NVFP4 weights a round's inputs); core starts passes and keeps checkpoints there | a reused prompt prefix and a fresh prefill give the same continuation when their passes past the prefix are longer than 16 rows |

Not promised: the prefill GEMMs (bf16 inputs on tensor cores) and attention
(binary16 inputs on tensor cores) and the 1..16-row ones (split activations; on GGUF weights the same 8-bit
inputs with another order of float32 additions) differ from each other, and so do
the chunked prefill recurrence (gdn_chunk, bf16 products on tensor cores) and the
row-by-row one of rounds; prefill and decode therefore agree only within
tolerance.

A wide round's bf16 GEMMs (a linear's, through regions/linear.cpp, and an adapter's) of up to 64 rows take
`k_gemm_skinny_mt` in three parts: it reads each span of weights once for every 16-row tile and gives each tile the
MMAs, chunks and sums the 1..16-row kernel gives it (`gemm` 16 rows at a time where it does not fit), so every row has
the bits it has alone (R, C).  A pass only drafting reads (DV's canvas, `Launch::draft`) is bound to neither: its bf16
GEMMs of 17..64 rows take bf16 inputs on `k_gemm_skinny_mt` in one part (`gemm`, and `gemm_draft` for several weights
on the same rows in one launch), deterministic but not the bits of a row alone.  Their K partition (the chunks, which
decide the sums) is K's alone: 8 chunks, 16 from K = 4096 (`gemm_draft_chunks`), in every table and at every row
count; a table (`bf16_draft`) and `cuda_bench tune rows` choose only the tile and the unroll.  So the weights one
launch takes each get the sums of a launch of their own (F: option fusion=0 runs them one by one), and no tuning run
moves a canvas.

## Rules that keep the bits

1. **Fixed reduction order.** K partitions, split counts and block sizes
   depend on the shape (or a row's own position), never on the row count or
   the configuration; partial sums are added in a fixed order. No float
   atomics on a result.
2. **Explicit rounding** (`__fmul_rn`, `__fmaf_rn`) in arithmetic two paths
   share: left to contraction, the compiler may fuse differently in each
   kernel (squares feeding a sum, the recurrence, the conv).
3. **Correctly rounded intrinsics for constants** (`__frsqrt_rn`): a
   compile-time argument folds to the exact value, a run-time one to the
   approximate instruction.
4. **Same block size, same sum.** Two launchers of one kernel that must agree
   use the same block size (`rmsnorm` and `rmsnorm_split`: 512; a round's few
   rows spread over blocks of 512 that each sum the whole row as one block
   does, then write a slice: `k_rmsnorm_q8`, `k_rmsnorm_spread`).
5. **Dependent launch.** What the kernel before writes is never read through
   a `const __restrict__` pointer or `__ldg` (the compiler may hoist the load
   above `griddepcontrol.wait`).

## Against llama.cpp

No rounding by default is coarser than llama.cpp's default CUDA path on the
same model file (AGENTS.md rule 5), except where a measurement on a real model
showed no visible drop (the FP8 K/V cache on Qwen; a Llama, where FP8 showed
one, keeps binary16 by default). Where the two stand, from llama.cpp's sources
(aa75ff3, 2026-11-10):

| What is rounded | flyx | llama.cpp | |
|---|---|---|---|
| K/V cache | FP8 by default where under 32 GB of GPU memory are left after the weights (binary16 above): e4m3 with a float32 scale per position and head, 4 significant bits; on the 27B, KL 0.0009 to 0.0016 against binary16 up to 12,000 tokens of context, where the weights' 8-bit inputs make 0.0011; on MiniCPM5-1B (no q/k norms) KL 0.0064, the same top-1 97.2%, perplexity +1.4% (1,200 positions, each text after the `<s>` its tokenizer puts first), so a Llama's default is binary16 whatever the room (`--kv fp8` on request). `--kv 16`: binary16 (11 significant bits) | binary16 (`cache_type_k`, `cache_type_v`) | coarser by default (measured); equal with `--kv 16` |
| Prefill attention: q and the probabilities into the tensor cores | binary16 (with FP8 caches on grouped kernels, p times its key's value scale goes in, times a power of two, as in decode's `attention_rows_mma`) | binary16 (`fattn-mma-f16`) | equal |
| Prefill attention: the sums of p v | float32 | binary16 (`T_C_VKQ`) | finer |
| Decode and verification attention: q and the probabilities | rows of 8192 keys or more (binary16 and FP8 caches): binary16 parts into the tensor cores, one (11 significant bits) with more than 4 query heads a KV head at head dim 256 or more, where llama.cpp rounds them to one binary16 at every row count, three (33) elsewhere; with FP8 caches p times its key's value scale goes in, times a power of two; below 8192 keys and `rows_attention=cores`: float32 | binary16 in `fattn-mma-f16` (rounds of more than one row; one row with more than 4 query heads a KV head at head dim 256, or past 8192 keys), float32 in `fattn-vec` (other single rows) | equal on those shapes, finer elsewhere |
| Chunked GDN recurrence (prefill): the products' inputs | bf16, one product each, as vLLM's and SGLang's (FLA's chunk kernels; high and low bf16 parts before); T solved in float32; only on sm_86, 89, 120 and 121, float32 recurrence elsewhere (`gdn_chunk_device`) | high and low bf16 parts, three products (`gated-delta-net-mma.cu`, enabled on those four; float32 on the others: A100, H100, B200) | coarser (the 27B NVFP4, 4,000 tokens of context: KL 0.0071, the same top-1 97.0%, against FP8 K/V's 0.0083, 97.5%) |
| GDN in-projection rows (prefill passes with NVFP4 inputs): q, k, v, z and the gates' inputs | bf16, each the float32 product rounded once (gemm_staged_bf16), as vLLM and SGLang hold them; float32 in rounds | float32 | coarser |
| The residual stream (prefill passes of more than 16 rows with NVFP4 inputs, every residual add a W4A4 GEMM, a bf16 embedding) | bf16, each add's sum in float32 then rounded once, as vLLM, SGLang and basalt keep it (vLLM's fused add and norm: the add in float32, stored in the model's dtype); float32 in rounds | float32 | coarser |
| GDN state | float32 | float32 | equal |
| GGUF weights: inputs of passes of more than 16 rows | 8 bits, a scale per 32 | 8 bits, a scale per 32 (`q8_1`); Q2_K a scale per 64 | equal; Q2_K finer |
| GGUF weights: inputs of rounds | 8 bits, a scale per 32 | the same (`q8_1`) | equal |
| NVFP4 weights declaring NVFP4 inputs: passes of more than 16 rows | NVFP4 on a GPU with FP4 tensor cores; bf16 on others | FP4 on Blackwell by default (`can_use_q4`); 8 bits elsewhere (`q8_1`) | equal; finer without FP4 tensor cores |
| bf16 weights (and the W12 head): inputs of 1..16-row GEMMs | three bf16 parts (24 bits): exact products, float32 sums | float32 (`mmvf`, a few rows; bf16 through cuBLAS beyond) | equal |
| NVFP4 weights: inputs of 1..16-row GEMMs | bf16, each value its own exponent | 8 bits, a scale per 32 (`q8_1`) | finer |
| GGML NVFP4 weights (GGUF): inputs of passes of more than 16 rows | NVFP4 (a file whose `general.tensor_extra.prec_a4` names any tensor false, or with a GGML NVFP4 matrix whose K is no multiple of 128, or a GPU without FP4 tensor cores: bf16 for all) | FP4 on Blackwell by default (`can_use_q4`; a tensor the file's `general.tensor_extra.prec_a4` names false, and every tensor on other GPUs: 8 bits) | equal (bf16: finer) |
| 8-bit inputs: the rounding | from the float32 value, x / (max / 127) by a correctly rounded division, to nearest; the scale and each group's sum in float32 (the sum exact: of the integers) | from the float32 value, x times 127 / max (fast-math), half away from zero; for Q4_0, Q4_1, Q5_1, Q4_K, Q5_K and IQ1_S the scale and a partial sum of the unrounded values as binary16 (`ds4`) | equal or finer |
| NVFP4 inputs (W4A4): the rounding | from the float32 value, one rounding: fused producers leave a pass's rows in float32; the gate and up GEMM leaves its SwiGLU in bf16 with each row's largest magnitude, as vLLM and SGLang keep theirs, rounded from there; a scale per row besides the e4m3 one per 16 | from the float32 value (`quantize_mmq_nvfp4`) | equal or finer; the down projection's inputs coarser (bf16 first) |
| exp, division, square roots, values below 2^-126 | correctly rounded where the kernels do not ask for the fast intrinsic; subnormals kept | `-use_fast_math`: approximate, subnormals flushed to zero | equal or finer |
| Norms, SwiGLU, the GDN convolution, gated norm and row-by-row recurrence, sampling | float32 | float32 | equal |
| Vision encoder weights (`mmproj`) | `Q8_0` linear weights + 8-bit inputs by default on floating-point `mmproj` files (`--vision-quant q8`, ~46% smaller; pre-quantized GGML `mmproj` files stay packed; disk offload `vision=disk` by default stores weights in an auto-deleted flushed temp file for 0 resident VRAM and 0 locked host RAM at KV-budget time, 3-stage streamed to GPU for bit-identical outputs to `vision=host` and `vision=gpu`): on Qwen3.8-27B (`UD-IQ3_XXS` + `mmproj-BF16`, 800 multimodal positions across 4 images via `kld`), KL mean 0.00125 (p90 0.00351, p99 0.01418), top-1 98.75%, perplexity 1.2296 vs 1.2280 (+0.13%) against BF16 (`--vision-quant exact`), matching FP8 K/V and GGUF 8-bit activations; opt-in `--vision-quant exact` keeps file precision (BF16/F16/F32); opt-in `--vision-quant nvfp4` (NVFP4 + bf16 inputs, ~70% smaller; KL 0.00516, p99 0.06681, top-1 97.75%) | file precision (`mmproj-BF16` / `mmproj-Q8_0`, `--no-mmproj-offload` runs on CPU) | coarser by default on BF16 `mmproj` (measured); equal with `--vision-quant exact` or pre-quantized `mmproj` |

## Launchers

GGML block weights (GGUF files) are what
`ops/dtype/ggml.h` says they are, in float64; `ggml_blocks` checks every
value of random blocks of every type against the `gguf` package's decoder.
The CUDA backend keeps them packed and its kernels (`gemm_blocks`,
`embed_blocks`) decode them in registers to exactly those values. A weight is
`d * s * q + minimum`: `q` a small integer, `s` a few bits per group of 16 or
32 weights, `d` the block's binary16 scale. The tensor cores multiply
binary16 activations by the integers `s * q`, which binary16 holds exactly
(checked when the decoders' tables are built: `gemm_blocks_tables.cu`), a
block at a time from zero, and `d` multiplies the sums in float32; a minimum,
and the second part of a scale whose product with `q` would need 12 bits
(Q6_K, IQ4_XS), is a second such sum with its own coefficient. In passes of
more than 16 rows a minimum's sum is taken over runs instead of weights: a
minimum is one value for 16 or 32 weights, so its part of an output is the
sum over those runs of (the row's sum over the run) times (the minimum).
The staging kernel adds up each row's runs of 16 in order, in float32, and
the GEMM holds such a sum as two binary16 values, its high and low parts:
22 bits of a partial sum, of which float32 keeps 24.

The activations are rounded to bf16, as for NVFP4 weights, and then held as
binary16 over a power of two per row that puts the row's largest value just
under 2^16. That step is exact for every value within 2^-32 of the row's
largest and rounds smaller ones to the binary16 subnormals; the reference
(`linear_a16h`, and `cpu_ref` for a block weight with these inputs) does the
same.

That is the exact contract (`--activations exact`). By default a block
weight's inputs are what its file says (`flyx_weight.activations`):
for GGUF files 8-bit integers with a scale per 32
values in every pass (`round_q8`: from the float32 values, `d = m / 127` for
the group's largest magnitude `m` and `p` = the nearest integer of `value / d`,
both in float32 as the kernels compute them: one rounding, as llama.cpp's;
a fused producer writes these
rows itself from its float32 values, a warp a group of 32, with
`blocks_stage`'s bits, and the attention's output for them stays float32
and is staged). A sub-group's products of codes
and `p` are added up exactly in integers (the tensor cores in passes of more
than 16 rows, `dp4a` in rounds), from the bits of 1.5 · 2^23 so that one
subtraction reads the sum as float32; its factors and `d` multiply that in
float32, `acc = fma(sum, d f, acc)` and for a minimum
`acc = fma(d sum p, m, acc)`, sub-group after sub-group. A prefill pass
takes the sub-groups in the order of K. A round splits each column's groups
of 32 among a warp's 32 lanes (lane l: groups l, l + 32, ...), each lane in
the order of K, and adds the lanes' sums by a fixed tree (xor 1, 2, 4, 8,
16): the same in a round of any number of rows. The reference (`linear_a8`,
and `cpu_ref` in every pass) rounds the same way. Because a group is scaled
by its largest value, one float32 step upstream (the float32 norm against the
float64 one) can move a whole group's integers; where the reference's two
best tokens nearly tie, the backend may pick the other one (`gguf_cuda`
compares the tiny models' replies teacher forced for that reason). With
option `blocks=bf16` the weights are instead decoded at load and rounded to
bf16 like any float32 matrix, which is outside this contract: on the tiny
test models it changed the greedy tokens of two of the nineteen types, the
packed kernels of none.

GGML's own NVFP4 (GGUF type 40: blocks of 64 e2m1 values with an unsigned
e4m3 scale per 16, no scale per tensor) is what
`ops/dtype/ggml.h` says, checked against the `gguf` package's decoder like
the other block types (every scale byte, 0x7F read as zero and the sign bit
ignored as ggml reads them). The CUDA backend repacks it at load into the
NVFP4 planes below with a row scale of 1, in the model's row order
(`ggml_nvfp4_planes`, the loader's: the same e2m1 codes and scales exactly; a
scale byte of 0xFF, no e4m3, refused; checked end to end by `gguf_e2e_test`'s
quant_NVFP4.gguf and tiny_nvfp4.gguf against cpu_ref), and both backends then
treat it as an NVFP4 weight: bf16 inputs in rounds, NVFP4 ones in passes of
more rows, as llama.cpp reads the same file.

NVFP4 weights take bf16 inputs in rounds. In passes of more rows, those of a
file that says its inputs are NVFP4 (`"quant": {"activations": "nvfp4"}`,
what the converter writes for ModelOpt's checkpoints, which are calibrated
that way) take NVFP4 inputs on a GPU with FP4 tensor cores, bf16 ones on others
(`FLYX_ACT_NVFP4`;
`round_nvfp4`: from the float32 values, one rounding; a
scale per row g = 6 · 448 / m for the row's
largest magnitude m, so no block scale saturates; per 16 values an e4m3
scale, the nearest to (their largest magnitude / 6) · g, and each value the
nearest e2m1 of value · g / scale; all in float32 as the kernels compute
them). `gemm_w4a4` multiplies them on the FP4 tensor cores with float32
sums; the reference (`linear_a4`, and `cpu_ref` in passes of the same rows)
rounds the same way. The kernels need K a multiple of 128 and sm_120a: a
backend that cannot compute the contract refuses the model at open rather
than computing another one. `--activations exact` keeps bf16 inputs in
every pass; `--activations nvfp4` gives NVFP4 inputs in passes of more rows
whatever the file says, with the same refusals.

Tolerances are relative errors against the float64 reference as the tests
check them; "golden" means covered end to end by `golden_test` (tiny models
against the reference) rather than by a kernel test.

| launcher | reference | tolerance | bits | tests |
|---|---|---|---|---|
| gemm_rows | linear | 2e-5 | R (16 rows at a time: any count) | test_gemm, test_gemm_round |
| gemm_skinny, gemm_skinny_cfg | linear | 2e-5 (activations as three bf16 parts, 24 bits: exact products, float32 sums) | R, C | test_gemm, test_gemm_round |
| gemm_mma | linear_a16 (bf16 activations) | 2e-5 | K | test_gemm |
| gemm_tc | linear_a16 (bf16 activations) | 2e-5 | C, K | test_gemm |
| gemm | dispatch to the above by rows and shape; a round's rows past 16 (Launch::round_rows: a linear's bf16 weights, regions/linear.cpp, and an adapter's) gemm_skinny_mt_cfg in three parts (its configuration from the GPU's table, bf16_rows, at 64 rows for more; more than 64 rows, DSpine's walk, in one launch, 64 a grid row), else the 1..16-row choice a tile of 16 rows at a time; a drafting pass's (Launch::draft) of 17..64 rows gemm_skinny_mt_cfg in one part (its tile and unroll from bf16_draft, its chunks K's) | as the path taken | R, C, K (a round's rows: R at any count, 17..64 and 65..240 checked); a drafting pass's: deterministic, its K's partition in every table | test_gemm, test_gemm_round, test_lora, test_gemm_draft_tables, test_fused_producers |
| gemm_skinny_mt_cfg | linear (three parts), linear_a16 (one part: bf16 activations), 17..64 rows (three parts: more, 64 a grid row), each span of weights read once for up to four tiles of 16 rows | 2e-5 | three parts (16 chunks, a warp each): R (every row the bits gemm_skinny gives it alone), C; one part: the same bits run to run, its chunks (8 or 16) deciding them, a drafting pass in K's (gemm_draft_chunks) | test_gemm_round |
| gemm_draft, gemm_draft_cfg, gemm_draft_chunks | linear_a16 of up to four weights on the same bf16 rows in one launch, each from its own columns of them, plain, accumulating or rounded to bf16 (DV's canvas pass past 16 slots); the first weight's tile and unroll (bf16_draft), the chunks K's (gemm_draft_chunks: 8, 16 from K = 4096, every table, every row count) | 2e-5 | F: each weight the bits of gemm() of a drafting pass alone (under every table, MiniCPM5-1B's canvas groups of weights of other shapes); deterministic | test_gemm_round, test_gemm_draft_tables, golden (past 16 slots: fusion=0's trees) |
| stage_bf16 | float32 rows to bf16, nearest (gemm_draft's input) | exact | R | test_gemm_round |
| gemm_presplit | as gemm on the producer's staged input | as gemm | F | test_fused_producers |
| lora_down | linear_lora's first product (ops/linear/ref.h): an adapter's a (bf16, the a's of a block's linears that read one input stacked) on the rows the linear read, through gemm, or (x null) through gemm_presplit on a fused producer's staged input; times the scale, one float32 product after the sums (none at scale 1, bifill-dv's); its second product y += u b^T is gemm on u, accumulating after the linear's own (a V-LoRA, ABI 43: every pass of the target; DV's canvas adapters too, but for a canvas of 17 to 64 slots on bf16 weights at scale 1, whose a's gemm_draft runs beside the base weight, its product in bf16) | 2e-5 as gemm (1..16 rows: inputs as three bf16 parts; a prefill pass's: bf16 inputs, u's too, which the reference rounds first) | R (a round's rows past 16 on k_gemm_skinny_mt in three parts, 16 rows at a time where K is no multiple of 32; both products: 17..64 checked, lora_in's layout one launch a GEMM), F (the staged input: the generic producer's bits, bf16 and NVFP4 linears beside it), C | test_lora |
| gemm_fp4_skinny_cfg | linear_a16 of dequant_nvfp4 | 2e-5 | R; C per K partition | test_fp4 |
| gemm_fp4_mma, gemm_fp4_tc | linear_a16 of dequant_nvfp4 | 2e-5 | C (tc), K | test_fp4 |
| gemm_fp4 | dispatch to the above | as the path taken | R, C, K | test_fp4, test_fused_producers |
| gemm_fp4_presplit | as gemm_fp4 on the staged input | as gemm_fp4 | F | test_fused_producers |
| gemm_blocks, blocks_stage, blocks_stage_presplit, gemm_blocks_staged, gemm_blocks_staged_runs, gemm_blocks_cfg, gemm_blocks_q8_cfg, gemm_blocks_q8_runs, gemm_blocks_round_q8_kernel, gemm_blocks_round_q8_cfg, gemm_blocks_round_q8_stack_cfg | linear_a16h of the block weights (ops/dtype/ggml.h), every block type; 1..16 rows as pairs of rows in the launch's scratch, more as binary16 rows in its prefill buffer; with inputs in 8 bits (BlockMat::q8, FLYX_ACT_Q8) every pass against linear_a8, the rows as int8 with a scale per 32 in the scratch (a round's laid out for its kernel, gemm_blocks_round.cu, with each 16's sum times the scale) | 2e-5 (measured 5e-8 to 6e-7: the order of float32 additions, and for more than 16 rows with binary16 inputs a minimum's sums over runs of activations, kept to 22 bits) | R (1..16 rows: one order of additions whatever the row count; binary16 inputs: K in 8 runs of whole windows added as a balanced tree, the same in every configuration; 8-bit inputs: a column's groups split among 32 streams, a stream's in the order of K, the streams' sums by a fixed tree, in the dp4a kernel a stream a lane and in the four tensor-core kernels (the rows in the MMA's first operand, or the columns, or either with the streams split among blocks) 8 a warp, the split kernels' four sets of 8 added by the tree's last two steps (xor 8, 16) by whichever of a tile's four warps arrives last, in the fifth (k_round_blocks_warps, IQ4_XS and the K types Q4_K to Q6_K: the streams split among a block's 8 warps) 4 a warp, added by the tree's first two steps in the warp and its last three by the block's first warp, in the warps' order, whatever the columns a warp takes or the rows a kernel holds, and whether a stack's run shares its launch with the next one or two runs, of other types: gemm_blocks_staged_runs, gemm_blocks_round_q8_stack_cfg); C (gemm_blocks_cfg, gemm_blocks_q8_cfg, gemm_blocks_round_q8_kernel, gemm_blocks_round_q8_cfg: every compiled configuration, the six round kernels and every configuration of the dp4a one (columns a warp, groups ahead); gemm_blocks_q8_cfg also every spread of the runs below); K (more rows: the bits of a longer pass, whatever the tiles its rows fall in; with 8-bit inputs K is added up in runs of whole windows fixed per shape, gemm_blocks_q8_runs from variants blocks_q8_runs by the layer's columns and K (gemm_blocks_q8_cfg's runs: another partition for benchmarks and tests, by the same rules), each run from zero in the order of K and the runs folded onto y in order, the same whether a block takes its tile's runs in turn or each run has a block of its own (the tile's last folding them all from the scratch, or each folding its own onto y once the runs before it have: fold_runs)); F (a producer's bf16 rows staged by blocks_stage_presplit are blocks_stage's) | test_blocks, test_blocks_round_stacks, test_blocks_round_side, golden (gguf_cuda) |
| embed_blocks | the block weights of the table's rows, as float32 (the table on the GPU or, when no head reads it, in host memory, the same bytes) | exact (the planes' integers times their coefficients in one fused operation) | R | test_blocks (both places) |
| blocks_pack, blocks_supported, blocks_init | the file's blocks as payloads and scales (host); which types and widths the kernels take; the decoders' device tables (ggml's codebook rows as selectors of their magnitudes; magnitudes times scale, bit fields and scales as binary16) | exact (test_blocks decodes what it packed, through the tables; the tables' builder refuses an integer binary16 does not hold) | - | test_blocks |
| gemm_glu_staged | silu(a) b of the gate and up GEMM, in its epilogue (rows > 16) | as gemm_presplit / gemm_fp4_presplit and swiglu_split: the same bits | F, C, K | test_gemm_glu, golden |
| gemm_on_glu | as gemm_presplit on gemm_glu_staged's rows | as gemm_presplit | F | test_gemm_glu |
| quantize_fp8_rows | e4m3 of w / (the row's largest magnitude / 448), nearest | exact (codes and scales as the host's) | - | test_gemm_fp8 |
| gemm_fp8_skinny | linear of the e4m3 weights (opt-in: head_fp8); where K % 256 == 0 of the rows rounded once to bf16 (k_gemm_fp8_sliced, basalt's sliced product: 16 weight rows a block over the whole K, 4 warps on fixed 64-value slices of every 256, their sums added in warp order), else of the rows as three bf16 parts | 2e-5 | R (a row's sums depend on K alone), C | test_gemm_fp8 |
| gemm_skinny_cols | the listed columns of gemm_skinny | gemm_skinny's bits | R; gemm_skinny's | test_head_cert |
| head_cert_coef | per vocabulary entry, `coef = norm(w - q) + 1e-3 * (norm(w) + norm(q))` and `qn = norm(q)` for the scaled e4m3 copy q | both rounded up | - | test_head_cert, test_head_cert_input_rounding |
| head_certify | greedy choice over the vocabulary. Both screening bounds use the radius `norm(x) * coef + dn * qn`, with `dn = norm(x - bf16(x))` on the sliced path and zero on the three-part path | gemm_skinny's greedy token alone and in either batch order, survivors with gemm_skinny's bits | R | test_head_cert, test_head_cert_input_rounding, golden |
| w12_encode, w12_finish | the head's bf16 weights in W12 (12 bits, lossless), every one decoded and compared | exact (the bf16 bits back) | - | test_gemm_w12 |
| gemm_w12_skinny, gemm_w12_skinny_cfg | as gemm_skinny on the decoded weights (the head, 1..16 rows, by default) | the bits of gemm_skinny | R, C | test_gemm_w12 |
| make_sign_head | the drafts' 1-bit head: sign of each weight, mean abs of each 64 (bf16) | exact (as the host makes it) | - | test_draft_pick |
| gemm_sign_rough, gemm_sign_rough_add | linear of the 1-bit weights (drafts' candidate scores; `gemm_sign_rough_add` accumulates into existing rough scores with optional bias and L2-cached table reads for additive draft heads) | 1e-4 (bf16 activations; one row and exact_1row multi-row: float32); 2 to 8 rows (the head's columns as the MMA's rows) each row's bits as 9 to 16 rows give them; exact_1row multi-row gives each row the bits of 1 row | C | test_draft_pick |
| draft_select_cands, draft_score_bf16, draft_combine_cands, draft_pick_topk, draft_pick | shared two-stage candidate-head pipeline (drafts only): select top 256 rough candidates per row, score survivors against bf16 weights (overwrite or accumulate + optional bias + optional softcap) or combine packed `dots_blocks` candidate scores, and pick top k with truncated softmax prob/margin | 1e-5 (the logits, 3e-5 with accumulate/softcap); the exact top k when the best stand out | C | test_draft_pick |
| make_sign_head_rows | make_sign_head from float32 rows (a packed head's, decoded by embed_blocks), a run of rows at a time | exact (as the host makes it from the rows' values) | - | test_draft_pick_blocks |
| draft_pick_blocks | draft_pick on a packed head: every row's candidates scored in float32 from the packed rows (dots_blocks) | 1e-5 (the logits); the exact top k when every token is a candidate | C | test_draft_pick_blocks |
| dots_blocks | dots of rows of a block table (their values as embed_blocks gives them) with rows of x, in float32 | 1e-5; bit for bit the dots of the rows embed_blocks decodes, read 4 values a lane in order | R | test_draft_pick_blocks |
| quantize_rows_nvfp4_bf16 | quantize_rows_nvfp4 of the bf16 rows as float32, each row's largest magnitude given (the SwiGLU rows the W4A4 gate and up GEMM leaves) | exact | K | test_gemm_w4a4 |
| quantize_rows_nvfp4 | the host's NVFP4 rounding of a row (device.cuh: kNvfp4RowRange, nvfp4_block_scale, e2m1_nearest), whose values are round_nvfp4's (ops/dtype/ref.h; rows reaching e4m3's subnormal block scales and spread over 30 binades included) | exact | K | test_gemm_w4a4 |
| gemm_on_glu_bf16 | gemm_on_glu into bf16 rows (accumulate: the float32 sum with the row's value, rounded once; the down projection into a prefill pass's bf16 residual) | exact against gemm_w4a4 then the sum and the rounding | C | test_gemm_w4a4 |
| gemm_staged_bf16 | gemm_w4a4 on the scratch's float32 rows, each product rounded once to bf16 (y16; the GDN in-projection of a prefill pass) | exact against gemm_w4a4 then the rounding | C | test_gemm_w4a4 |
| gemm_w4a4 | linear_a4: linear of the rows as round_nvfp4 gives them and the dequantized NVFP4 weights (FLYX_ACT_NVFP4: passes of more than 16 rows of a file that declares NVFP4 inputs, or of any file with `--activations nvfp4`) | 1e-5; against bf16 activations ~1e-1 (their NVFP4 rounding); every tile configuration the same bits (64-value k steps in order) | K | test_gemm_w4a4; end to end gguf_e2e_test (tiny_w4a4.flyx against cpu_ref, teacher forced; tiny_w4a4_a16.flyx with `--activations nvfp4`: its tokens, greedy and sampled) |
| fp4_decode_test | e2m1_to_double | exact | - | test_fp4 |
| quantize_nvfp4 | dequant_nvfp4 | block error <= 1.07/6 of the block max | - | test_quantize_nvfp4 |
| quantize_q8_0 | Q8_0 of a bf16 matrix: d = binary16(max abs / 127) a block of 32, q = round(w / d) (a block drafter's projections beside GGML block weights or a BF16 target: drafts only) | exact (codes and scales as the host computes them) | - | test_quantize_q8_0 |
| rmsnorm | rmsnorm | 1e-5 | F | test_rmsnorm, test_fused_producers |
| rmsnorm_split | rmsnorm, staged for the next GEMM (q8: in 8 bits, blocks_stage's rows) | as rmsnorm | F | test_fused_producers |
| swiglu, swiglu_split | silu(a) b (q8: in 8 bits, blocks_stage's rows) | through the GEMM after it | F (q8, a round of up to 4 rows on block weights: also written by the dp4a gate and up GEMM, a block a group of 32 outputs, Launch::glu, gemm_blocks_round_glu_takes: the same bits) | test_fused_producers, test_blocks_round_glu, golden |
| attn_prep | prep_head / prep_head_3d: rmsnorm, rope (1D or 3D MRoPE with per-row pos_3d on prefill or scalar pos + P->rope_delta on decode); a block without q/k norms (Llama): rope of the head as projected (a tree round's row: rotated to its depth's position, written to its own row's slot; optional 16-aligned `strata_base` writes staged StrataFlux words and grids from offset 0 with zeroed padding words; a DV canvas pass's row w (CanvasKeys): rotated to P->start + 1 + w, written to the scratch slot s0 + w and no other; a canvas pass with a tree or a low-bit copy is refused: DV's rounds read every position exactly, so neither is there) | 1e-5 (q); K and V caches: binary16 rounding, saturating at +-65504 (round_kv16: an infinity there made every row that attends to it NaN), 2e-3 | R | test_attention, test_kv16_saturation, test_attention_fp8 (both with and without q/k norms), test_strata, test_canvas_attention, test_mrope, golden (tree rounds; tiny Llama) |
| rope_register | rotary factors theta^(-2 i / rot_dim) in float64, once at load, by the kernels' own pow (rope_inv), for attn_prep and draft_ctx_kv to read instead of a pow a launch | exact (the same function as a launch's pow) | - | test_attention |
| attention, attention_split | attend (a drafter's sliding window: over the keys within it; a tree round's row (up to FLYX_MAX_WIDE_ROWS rows): its round's keys read from its ancestors' slots in depth order, one row a block, the bits of decoding its path; a DV canvas pass's row (CanvasKeys): the text [0, P->start] and the canvas's scratch slots, all or up to its own when causal, never the round's tree slots, one row a block on the float32 kernels at every length (no tensor-core prefill or rows_mma for it), binary16 and FP8 caches; windowed short prefill (`SplitWindow`, 64-key aligned windows for pageable KV layers with <= 16 rows): carries each straddling split's exact per-warp online softmax state in `kSplitStateFloats`, read from the buffer the launch before wrote and written to another (a split that ends inside a window and the one after it are two blocks of one launch), reproducing single-pass `attention_split` bit-for-bit across `k_attention_gqa`, `k_attention_rows_mma`, and `k_attention`) | 2e-5 (1e-4 past 1000 keys) | R (a canvas row alone: the same bits); the chain's and tree's kernels compiled as before the canvas map (the same SASS) | test_attention, test_attention_fp8, test_canvas_attention, test_pageable_kv_ops (`SplitWindow` bit-identical across GQA, rows_mma, mixed, per-head, with splits ending inside windows and a whole split inside one), golden (tree rounds, pageable short-tail prefill bit-identical to pinned) |
| attention_rows_mma | attend (binary16 and FP8 caches, no window; FP8: the bytes made binary16 exactly in shared memory, a score times its key's scale, p times its key's value scale and 2^(14 - E), E the largest value scale's exponent of the row's keys so far: attention_split's partials on tensor cores; q and p as three binary16 parts, one (llama.cpp's) with more than 4 query heads a KV head at head dim 256 or more (`rows_attention_parts` 1 to 3); `rows_attention_q=fp8` over FP8 caches on `sm_89+`: q as e4m3 times a power of two (its largest value below 2^8), scored on `mma.m16n8k32` FP8 tensor cores on `sm_90+` and on binary16 tensor cores on `sm_89` (avoiding Ada's 22-bit FP8 accumulator); every product exact, float32 sums; a row in warps of its own; a tree round's rows share the tiles of keys before the round and take the tiles holding the round's keys one row at a time) | 2e-5 (1e-4 past 1000 keys); one part 3e-3 (prefill's); q as e4m3 against the reference with q rounded the same way | R | test_attention, test_attention_fp8 (both kernels, every part count, q as e4m3); golden (rounds on tensor cores: drafted == plain, binary16 and FP8, one part, q as e4m3) |
| attention_flash, attention_flash_gqa_cfg | attend (binary16 q and p; binary16 caches as they are; FP8 caches on the grouped kernel: K and V as their e4m3 bytes are, exact in binary16, a score times its key's scale, and p times its key's value scale and 2^(14 - E), E the largest value scale's exponent of the row's keys so far, carried in FlashWindow state slot 5; on the per-head kernel: V e4m3 times its scale rounded to binary16); a window of keys at a time, the softmax state carried exactly between windows on the same tiles: one pass's bits. Aligned historical windows within one scale tier can skip the causal mask and zero fill, preserving every MMA and softmax reduction. If every lane stays strictly below its finite running maximum, the grouped kernels retain that maximum and unit rescale, with the same probability-sum rounding. | 3e-3 | K | test_attention, test_attention_fp8 (windows of 64 keys bit for bit against a full pass, including FP8 24:4 HD256 passes with 2, 3 and 6 query heads per block, split scale storage and partial query tiles; 1,024-row FP8 state and gated outputs bit-identical across these three groupings) |
| gdn_conv, gdn_conv_fold | conv_row, conv_push (a tree round's row: its ancestors' inputs before it, the same products in the same order) | 1e-6 (window) | R, P | test_gdn, golden (tree rounds) |
| gdn_prep | l2norm, sigmoid, softplus | through gdn_recur | R, P | test_gdn |
| gdn_recur | gdn_row | 1e-5 | R, P | test_gdn |
| gdn_fold_layers | gdn_recur and gdn_conv_fold of every gated delta net of a sequence in one launch each (a round's commit) | the same bits (the same kernels, each layer's buffers from a table) | R | golden (MTP and DFlash2 rounds = plain decoding, rollback) |
| gemm_fp4_rows | linear_a16 of dequant_nvfp4 (bf16 activations), 17..64 rows on up to four 16-row tiles a weight decode | 2e-5; the bits of the 1..16-row kernel 16 rows at a time, every configuration | R | test_fp4 |
| shift_params | the round's parameters with its position moved on by a count (a wide round's 16-row strips of attention) | exact | - | golden (wide rounds) |
| gdn_recur_tree | gdn_row along each row's path of a tree round (gdn_recur's lanes kernel; a row whose parent is not the row before starts again from the round's state and replays its ancestors' steps) | the same bits as decoding the path (the same steps in the same order) | R | golden (tree rounds = plain decoding) |
| attention_strata, strata_words, strata_supported | strata attention (`--kv-method strata`, and past the switch of `--kv-method auto`; rows below the switch are exact attention's): a 2-4-bit copy of every position in the Walsh-Hadamard basis (its range and step per position and head in binary16, the codes chosen on that grid) read for all, each row's most attended pairs replaced by the cache's own values (StrataFlux's equation 8) | an approximation, not a rounding of the reference: KL against exact binary16 on the 27B measured (0.0012 to 0.0083 by bits, budget and context; 2 bits with 1 pair in 16 or 3 bits with 1 in 64 refined at FP8's 0.0012 after 30,000 tokens; after 100,000 tokens p90 0.0127 with at most 256 pairs a row, FP8's 0.0101); with every pair selected the cache's own attention (KL 0.00000 on the tiny model); the pairs re-read from host memory kept on the device and the selection in chunks give the same bits; the scan's softmax runs 32 keys a step and a pair's log-sum-exp is taken from its weights; the scan's MMAs take the codes exactly (K as 1024 + c 2^(bits sp) against the query, one binary16 part, scaled by 2^-(bits sp); V's codes against p times the key's step, one binary16 part, times a power of two a tile), the grids' steps and bases applied in float32: against the float32 scan the heads' outputs 2.9e-4 mean, the pairs' log-sum-exps 1.35e-4; the hosted HD256 three-bit compact layout stores the same codes in 24 words per head and position and restores the original 26-word snapshot representation | R (multi-row scans pack `(row, query-head)` columns across rows into full 8-column MMA warps with per-column split ends and V scales, using 1-warp zero-barrier scans for 2-bit and multi-warp `cp.async` CTAs for 3/4-bit; the complete two-bit HD256 splits with six query heads per KV head write adjacent pair scores together, preserving the even-then-odd probability sum, binary16 pair scores and FP32 split partials. The selection and the sums depend on the row's own query and position; a staged scan over two grids keeps every original key split and partial slot, with bit-identical partials and output, tested by test_strata; when the recent pairs fill the quota, their contiguous range gives the radix selector's ordered pairs and output bits, tested with both scans) | op (test_strata: the copy within half a step of the rotated K / V; the scan against float64 attention over the copy as stored, 3e-3, and each pair's log-sum-exp, with an unwritten score failing the comparison; the maximum 128 chunks give one chunk's pair IDs and output bits; every pair refined = the cache's attention, 6e-3; for colliding pairs selected by multiple rows, the lowest pair wins, FP8 K/V bytes and scales match the backing cache, and outputs match uncached reads bit for bit; a row alone = among the round's rows, bit for bit, also in rounds whose rows pass 128 or 256 keys, where a later row's keys fill one split more and its splits are shorter than the rows' before it (115..129, 126..130, 253..259 and 255..258 keys: a packed scan runs each split to the last of its rows' ends, which is not the last row's there)), golden (StrataFlux rounds: DFlash2 and MTP == plain, K/V in host memory too with pairs kept on the device, also over rounds across 128 and 256 keys, rounds of several sequences; the selection in chunks of 8 pairs = one chunk's tokens; every pair refined = the reference's tokens; auto: exact's tokens before the switch, drafted == plain across it); `op_test_attention.cu` compares compact and original K/V codes, grids, split partials, pair scores and corrected output bits, with the same float64 reference; pageable host-RAM gathers (`PageGatherCtx` via `cudaLaunchHostFunc` + bounce buffer + hot-pair cache) reproduce pinned host-KV output bits identically across FP8 and binary16 including forced hot-slot collisions |
| params_at | a draft step's parameters: the sequence's, its position moved | exact | - | golden (the MTP layer over its copy) |
| kv_window | the MTP layer's first and last positions a draft step reads, copied in order from its ring of slots to a buffer, with their scales | exact | - | golden (a window as long as the context: the drafts of none; a ring of 128 slots: the drafts of a slot a position, across its wraps; snapshots of the ring) |
| kv_stage | copy of a cache's positions in host memory to its view's staging, which the attention then reads; of a StrataFlux copy in host memory, 16 positions at a time, to its staging. In single-session decode with stage-ahead, fixed-size H2D transfers carry the committed whole tiles and their grids by default. kv_stage carries the tail, or the whole prefix with stage_dma=0. | exact | - | golden (K/V all in host memory = the reference's tokens; the GPU's part ending inside the prompt and inside the decoded tokens = all on the GPU; the copy in host memory: plain decoding's tokens under DFlash2 and MTP, two turns' logits = the copy on the GPU; DMA and kernel copies give the same tokens and logits on long, edited and restored prompts) |
| kv_stage_ring | copy of positions from a pageable KV layer's device tail ring (indexed via `KvRef::slot_of`, skipping 1 straddling slot `ring_pad = 1` when `dev % row != 0` so the tail ring stays strictly outside the view-shared device VMM handle) into its linear view/window staging buffer, with FP8 scales | exact | - | test_pageable_kv_ops (aligned and `nkv=10` unaligned VMM `row=2560`), golden (pageable KV prefill, exact transition, rollback, checkpoint restore, and snapshots = pinned host KV) |
| kv_drain_pageable | copy of a round's committed rows (`e.n_tokens`) from each pageable KV layer's device tail ring into its mapped pinned drain ring slot (`P->seq_no % kRing`) so `read_result` commits them to OS pageable RAM | exact | - | golden (pageable and hybrid WDDM+pageable KV rounds, speculation, rollback, restore, and snapshots = pinned host KV) |
| gdn_compact, compact_kv, compact_rows | copies of a tree round's accepted rows to rows 0 .. n (replay rows, K/V slots and their e4m3 scales, the drafters' input rows) | exact | - | golden (tree rounds, and the MTP rounds after them = after paths that did not move) |
| gdn_gated_norm, gdn_gated_norm_split | rmsnorm, silu (q8: in 8 bits, blocks_stage's rows) | 1e-5 | F | test_gdn, test_fused_producers |
| gdn_round, gdn_round_fits | the four kernels above (q8: its output in 8 bits, blocks_stage's rows; optional `row_offset` slice for batched rounds) | as they are | F, R | test_gdn |
| gdn_chunk | gdn_row (every product's inputs rounded to bf16, the sums in float32; T solved in float32; float32 state) | 1e-2 (4e-3 to 7e-3 measured) | K | test_gdn_chunk, golden |
| argmax_rows | argmax | the same token | R | test_select |
| choose_rows | sample_token (with the grammar mask on row 0; a tree round's row samples its depth's position, a wide round's 16 rows at a time from row tree_from; every Gumbel draw uses a uniform strictly between 0 and 1; both signs of zero tie at top-k and top-p cutoffs) | the same token | R | test_sample (including signed-zero ties in plain and verification rows, the highest 24-bit draw and subnormal temperatures with and without a mask), golden (tree rounds, sampling) |
| penalize_rows | penalty of the count of each token in the full reply and, in a round, in the drafts before the row (a tree's row: its ancestors', up to 63; a wide round's 16 rows at a time from row tree_from) (ops/sampling/sampling.h) | 1e-6 (float32: the penalty's product and one subtraction); tokens the reply does not have: untouched | R | test_penalize (including draft and committed counts above 65,535), golden |
| topk_rows | topk; a row without a finite score: ids 0..k-1 (argmax_rows: 0), never past the vocabulary | the same tokens | R | test_select |
| selector_walk | DFlash2 selector (python/reference); DFlash's (no selector: one candidate, sel_rank 0) each row's best | the same drafts | - | test_select, golden (DFlash = the reference's drafts) |
| selector_tree | ops/draft/ref.h best_first_tree on the DFlash2 selector's scores (float32 sums and softmax over a node's candidates) | the same drafts and tree (inputs without near ties; drafts only: the round verifies whatever they are) | - | test_select, golden (DFlash2 trees = the reference's tokens) |
| dv_selector_tables | DV's selector token tables TA = A E^T, TB = B E^T (ops::selector_table's pred and succ: the selector's float32 A, B, the target's bf16 embedding), once at load | 1e-5 (one float32 fma chain over hidden a value, from 0 in order: that chain's bits whatever the tiles) | - | test_dv_selector_tables |
| canvas_input | ops::canvas_input (DV's canvas pass input rows: the two products on gemm() as a round's, the 1..16-row kernels' arithmetic a tile of 16 rows, their inputs as three bf16 parts; silu(g) u and the sum in float32) | 2e-5; a slot with z = 0 its embedding exactly; a canvas of another round (next_start): mask_emb exactly | R (a slot's bits whatever the canvas's size) | test_canvas_input |
| canvas_stats | ops::canvas_stats (the top 16, ties to the lower id; the log-sum-exp from the largest logit; p = exp((l - max) - log sum); their mass; zn) and the selector's H h / sqrt(rank): the top 16 the best of 16 parts' best 16 (a block a part); float32 sums in a fixed order (a thread's terms in order, block_sum's / warp_sum's trees; hid a warp a slot's dot) | ids exact; lse 1e-6; p, mass, zn, hid 1e-5; the same bits run to run | - | test_canvas_stats |
| dv_slide | ops::slide (the canvas moved by the tokens a round committed, n = P->position - P->start), cov = ops::slot_coverage of the slid logp; nothing after a no-op round | exact: inputs (p >= fill in float32), candidates, live slots, the z and hid rows copied; logp and cov 1e-6 (float32 log and exp) | - | test_dv_slide |
| dv_live | whether a round graph runs its canvas pass (the body of a conditional node): not after a no-op round (P->noop), whose pass dv_slide would leave without effect | exact | - | golden (DV rounds read a round at a time and two in flight: the no-op round behind the last) |
| dv_selector_table | ops::selector_table (float32: TA[prev] hid TB[cand] summed over the rank in a fixed order, alpha logp, / temp, log-softmax over the 16) | 2e-6 | - | test_dv_selector_table |
| dv_tree_beam, dv_tree_pool | ops::beam_tree<float> (DYN) and ops::pool_tree<float> (ST) on the same float32 table: the float instantiations, their constants rounded to float and their sums in their order (no contraction: __fadd_rn, __fmul_rn), ties to the lower index; DV's first tree and the test's chain | the same drafts and parents bit for bit (1103 random tables a kind with dead slots, MASK drafts, ties, beams wider than 16 and edges above the depth's step, a parent not drafted leaving its child under the anchor; each rank count's deepest pool that fits a block's shared memory; drafts only: the round verifies whatever they are); TreeRows the tree the parents describe, as set_tree writes a given one | - | test_dv_trees |
| grouped_conv | grouped_conv | 1e-6 | R | test_grouped_conv |
| dspine_normalize, dspine_codes, dspine_message, dspine_apply | ops::dspine_write (DSpine's code bridge after a layer: float32 rms and code norms, block_sum's tree; sigmoid with expf; its read, its stacked gate, pred and prev, and its write products on gemm(), 1..16 rows, bf16 weights) | 2e-5 (the rows after the write, the gate projections); rms 1e-6; the rows before it exactly | R (a row's values its own: a block a row) | test_dspine_write |
| dspine_heads, dspine_senders, dspine_scores, dspine_walk | ops::dspine_walk (the candidates' head rows times the final norm's weight and their base scores, float32; their heads through the last write and the senders' stacked pred and prev on gemm()'s round kernels 64 rows at a time (every row's inputs as three bf16 parts); a warp a candidate's message . row, the first best on ties; the chain through the picks; the test's oracle tokens) | base 2e-5; every pick the float64 best or within 1e-5 of the scores' scale of it (a near tie either way: drafts only, the round verifies whatever they are); the chain the picks' | - | test_dspine_walk, golden (DSpine = the reference's tokens; its drafts the reference's from exact weights) |
| embed, build_rows, tap_copy, copy_ints | copies, sums (embed: the table on the GPU or in host memory) | exact | R | golden (the tiny Qwen3's table: in host memory in golden_cuda_qwen3, on the GPU in golden_cuda_qwen3_embed_gpu) |
| mtp_gather, hring_put, mtp_oracle, draft_ctx_kv | MTP and DFlash2 inputs (python/reference); draft_ctx_kv's caches binary16 saturating as attn_prep's | drafts only | - | golden, test_kv16_saturation |
| given_drafts | copy of the launch's drafts (gen.drafts) | drafts only | - | golden (lookup drafts) |
| accept, prefill_done | ops/sampling/sampling.h acceptance; the reply's tokens counted for the penalties | exact | - | golden |
| accept_tree | ops::accept_tree (a tree's path, or a chain; relaxed rules when the request asks, a wide round's on the logits child_logits gathered; no token past the context's end: a round of more than 16 rows stops there) | exact | - | golden (tree rounds; min-p 1 = plain decoding; trees of 31 drafts, penalties and min-p 1; rounds of 32 and 64 rows at the context's end) |
| child_logits | each draft's logit in its parent's row (a copy), for a wide round's relaxed rules | exact | - | golden (trees of 31 drafts, min-p 1 = plain decoding) |
| stop_decide_step, stop_decide_block | ops::stop_decide_step_ref, ops::stop_decide_block_ref (intra-round confidence stopping for sequential and parallel block drafters; drafts only) | 1e-5 (survival); exact decision away from ties | - | test_stop_decide |
| row_stats | max, Z = sum exp(z), entropy of a row's distribution for the relaxed acceptance rules (float32 sums in a fixed order; a zero-probability token contributes zero to entropy) | which drafts qualify (a lossy rule the request asks for) | - | test_sample (normalizer, entropy and min-p 1 at subnormal temperatures), golden (min-p 1) |
| layernorm | ops::layernorm (ops/norm/ref.h): float32 mean, variance, rsqrt and optional affine scale/bias per row | 1e-5 | R | test_layernorm |
| add_bias, vision_residual_add | ops::add_bias (ops/vision/ref.h) and optional LayerScale residual add in float32 | 1e-6 | R | test_add_bias |
| vact_apply, vact_gated | ops::vact (ops/activation/ref.h): GELU (tanh and erf), SiLU, QuickGELU, plain and gated | 1e-5 | R | test_vact |
| vision_zero_tail | exact zero fill of padded columns `[valid_cols, ld)` per row for quantized vision linears with `K` not divisible by 32/64 | exact | R | test_vact |
| gemm_f32 | ops::linear (ops/linear/ref.h) on float32 inputs and float32 weights for arbitrary K (vision patch stems, including patch_size=14) | 2e-5 | R | test_gemm_f32 |
| vision_pos_embed_add, vision_pos_embed_bilinear_add | fixed modulo and ops::vision_pos_embed_bilinear_add (ops/vision/ref.h): align_corners=true bilinear interpolation in float64 coordinates, float32 accumulation | 1e-5 | R | test_vision_pos_embed |
| vision_rope_2d | ops::rope_2d_axial (ops/rope/ref.h): 2D axial rotary embedding in double angles, float32 sin/cos | 1e-5 | R | test_vision_rope_2d |
| vision_attention | ops::attend_range (ops/attention/ref.h): bidirectional block-diagonal online-softmax attention in float32 for arbitrary head_dim <= 256 (including Qwen3.8 head_dim=72) | 1e-5 | R | test_vision_attention |
| vision_avg_pool_blocks | ops::vision_avg_pool_blocks (ops/vision/ref.h): float32 mean over contiguous spatial blocks | 1e-6 | R | test_vision_avg_pool |
| embed_scatter | row-wise overwrite of float32 or bf16 residual rows with precomputed multimodal embeddings | exact | R | test_embed_scatter |
| attention_head_dim, kv_fp8_supported, device_supported, compiled_arch, attention_partial_floats, gemm_tuning_for, gemm_tuning_named, gemm_tuning_name, gemm_tuning_compiled_for, gemm_tuning_pdl, gemm_skinny_cfg_compiled, gemm_fp4_cfg_compiled, gemm_mt_cfg_compiled, gemm_blocks_default, gemm_blocks_cfg_compiled, gemm_blocks_q8_default, gemm_blocks_q8_cfg_compiled, gemm_blocks_round_default, gemm_blocks_round_default_kernel, gemm_blocks_round_cfg_compiled, gemm_blocks_round_kernel_compiled, gemm_blocks_round_glu_compiled, gemm_blocks_round_stack_compiled, gemm_blocks_round_glu_takes, gemm_blocks_round_warps_takes, gemm_config, glu_inputs, side_scratch, scratch_bytes, scratch_carve, scratch_in, gdn_chunk_align, gdn_chunk_rows, gdn_chunk_takes, gdn_chunk_device, gemm_w4a4_supported, w12_plane_bytes, sign_head_bytes, dv_tree_takes | queries, no numerics | - | - | - |
