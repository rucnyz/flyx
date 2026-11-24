# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Build a tiny random Qwen3.5 + DFlash2 pair in HuggingFace layout, convert it,
and write golden outputs from the reference.  Used by the C++ tests.  Also a
tiny Qwen3 (a second architecture) with golden outputs
from transformers itself.

    python tests/e2e/make_tiny.py OUT_DIR

Outputs in OUT_DIR: hf_target/, hf_draft/, tiny.flyx, golden.json,
tokenizer_cases.json, chat_cases.json, hf_qwen3/, tiny_qwen3.flyx,
golden_qwen3.json, hf_llama/, tiny_llama.flyx, golden_llama.json (a Llama
without q/k norms, MiniCPM5's layers, with golden outputs from
transformers), tiny_llama_dv.flyx (the same with a tiny DV drafter: dv_tiny/
holds its bifill-dv checkpoints and what export_dv.py made of them),
tiny_llama_dv_pool.flyx (the same drafter with a pool tree and a causal canvas, dv_tiny/export_pool),
tiny_llama_dv32.flyx (32 slots, dv_tiny/export_32), tiny_llama_dv31.flyx and tiny_llama_dv31_pool.flyx (32 slots,
trees of 31 drafts: dv_tiny/export_31, export_31_pool), tiny_llama_dv31_wide.flyx and tiny_llama_dv31_wide_pool.flyx
(16 slots, trees of 31 drafts from a flatter selector: dv_tiny/export_31w, export_31w_pool), golden_llama_dv.json (the
float64 reference's DV rounds on those files, for the CUDA backend's),
tiny_llama_nvfp4.flyx and golden_llama_nvfp4.json (its linears in NVFP4: prompts alone, the file held to its own
plain decoding), tiny_llama_lora.flyx, golden_llama_lora.json, tiny_llama_dv_lora.flyx, tiny_llama_dv31_lora.flyx,
golden_llama_dv_lora.json, tiny_llama_nvfp4_lora.flyx and golden_llama_nvfp4_lora.json (the tiny Llama with a V-LoRA
of its own, beside DV's trees of 15 and of 31 drafts in the second and third: make_llama_lora),
tokenizer_cases_minicpm5.json (its tokenizer: MiniCPM5's
pre-tokenizer, and the <s> its post-processor puts before a text),
tiny_gdn128.flyx, golden_gdn128.json, tiny_w4a4.flyx
(NVFP4 inputs in long passes; checked against the float64 backend); the
drafter alone as z-lab publishes DFlash2 drafters (hf_draft_zlab/) converted
to a drafter file (tiny_draft.flyx), with blocks of 8 and 7 (tiny_block8.flyx and tiny_block7.flyx),
and with a sliding window of 8
(hf_draft_window/: tiny_window.flyx with the target, golden_window.json,
and alone, tiny_draft_window.flyx).  dv_cases.json: DV's trees, selector tables and
canvas steps from the Python reference, for the float64 references in ops/
(tests/ops/draft/dv_tree_test.cpp, tests/ops/canvas/canvas_test.cpp).  tiny_dspine.flyx and golden_dspine.json: the
tiny Qwen3.5 with a tiny DSpine drafter (hf_dspine/, as tencent publishes them) in place of DFlash2's;
tiny_dspine_window.flyx and golden_dspine_window.json the same with a sliding window of 8; tiny_draft_dspine.flyx the
DSpine drafter alone; tiny_qwen3_dspine.flyx the tiny Qwen3 with a DSpine drafter (its tokens golden_qwen3.json's);
tiny_dspine1.flyx, golden_dspine1.json and tiny_draft_dspine1.flyx (hf_dspine1/) with a DSpine drafter of one layer.

The assets are made again only when what they are made from changed (the
digest in OUT_DIR/.inputs): this script, the reference and the converter, the
chat cases and their templates, the versions of the packages that build them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
STAMP = ".inputs"
OUTPUTS = ("tiny.flyx", "tiny_nvfp4.flyx", "tiny_qwen3.flyx", "tiny_gdn128.flyx", "tiny_w4a4.flyx",
           "golden.json", "golden_nvfp4.json", "golden_qwen3.json", "golden_gdn128.json", "tokenizer_cases.json",
           "chat_cases.json", "tiny_draft.flyx", "tiny_block8.flyx", "tiny_block7.flyx",
           "tiny_window.flyx", "golden_window.json", "tiny_draft_window.flyx", "tiny_w4a4_a16.flyx",
           "tiny_llama.flyx", "golden_llama.json",
           "tokenizer_cases_minicpm5.json", "dv_cases.json", "tiny_llama_dv.flyx", "tiny_llama_dv_pool.flyx",
           "tiny_llama_dv32.flyx", "tiny_llama_dv31.flyx", "tiny_llama_dv31_pool.flyx", "tiny_llama_dv31_wide.flyx",
           "tiny_llama_dv31_wide_pool.flyx", "golden_llama_dv.json", "tiny_llama_nvfp4.flyx", "golden_llama_nvfp4.json",
           "tiny_llama_lora.flyx", "golden_llama_lora.json", "tiny_llama_dv_lora.flyx", "golden_llama_dv_lora.json",
           "tiny_llama_dv31_lora.flyx", "tiny_llama_nvfp4_lora.flyx", "golden_llama_nvfp4_lora.json",
           "tiny_dspine.flyx", "golden_dspine.json", "tiny_dspine_window.flyx", "golden_dspine_window.json",
           "tiny_draft_dspine.flyx", "tiny_qwen3_dspine.flyx", "tiny_dspine1.flyx", "golden_dspine1.json",
           "tiny_draft_dspine1.flyx")


def inputs_digest() -> str:
    from importlib import metadata
    files = [os.path.abspath(__file__), os.path.join(ROOT, "tests", "core", "render", "chat_cases.py"),
             os.path.join(ROOT, "tests", "python", "reference", "dv_test.py")]  # its random_dv makes the tiny DV
    for sub in (("python", "reference"), ("python", "convert"), ("tests", "core", "render", "templates")):
        for d, dirs, names in os.walk(os.path.join(ROOT, *sub)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            files += [os.path.join(d, n) for n in sorted(names)]
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, ROOT).replace(os.sep, "/").encode() + b"\0")
        with open(f, "rb") as fh:
            h.update(fh.read())
    for pkg in ("torch", "transformers", "tokenizers", "safetensors", "jinja2"):
        h.update(f"{pkg} {metadata.version(pkg)}\0".encode())
    return h.hexdigest()


def up_to_date(out: str) -> bool:
    try:
        with open(os.path.join(out, STAMP), encoding="utf-8") as f:
            made_from = f.read().strip()
    except OSError:
        return False
    return made_from == inputs_digest() and all(os.path.exists(os.path.join(out, o)) for o in OUTPUTS)


# --digest: what the assets are made from, for a cache of OUT_DIR to be kept under (.github/workflows/ci.yml).
if __name__ == "__main__" and sys.argv[1:] == ["--digest"]:
    print(inputs_digest())
    sys.exit(0)

# Before the imports below (torch alone takes seconds): most test runs find the assets up to date.
if __name__ == "__main__" and up_to_date(sys.argv[1] if len(sys.argv) > 1 else "tiny_assets"):
    print("assets up to date: nothing they are made from changed")
    sys.exit(0)

import torch  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "python", "reference"))
sys.path.insert(0, os.path.join(ROOT, "python", "convert"))

from flyx_convert import convert, convert_draft  # noqa: E402
from flyx_convert.fileformat import Reader  # noqa: E402
from flyx_convert.tokenizer_export import MINICPM5_NUMBERS, MINICPM5_SPLIT, QWEN35_SPLIT  # noqa: E402
from flyx_ref import dv  # noqa: E402
from flyx_ref import greedy_plain, greedy_spec, greedy_spec_mtp  # noqa: E402
from flyx_ref.load import load  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|mask|>"]
ADDED = ["<think>", "</think>"]

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's 2026; we'll ship v0.1 soon!",
    "def add(a, b):\n    return a + b\n\nprint(add(3, 41))\n",
    "推理引擎需要又快又准。我们先把正确的路径写出来，再一步步变快。",
    "Numbers: 123456 0.75 -42 1e-6 and emoji 🚀✨ plus café naïve résumé",
    "  leading spaces\t\ttabs\r\nwindows lines\n\n\nblank lines   trailing   ",
    "I'm sure you've seen they'd done it; THEY'LL DO IT AGAIN, we're told.",
    "Combining marks: é ä ñ and Hangul 한국어 텍스트, Ελληνικά, русский текст.",
    "for (int i = 0; i < n; ++i) { sum += x[i] * w[i]; }  // dot product",
] * 20

CASES = [
    "Hello world", "hello  world", "  two leading", "trailing  ", "line1\nline2\r\n\r\nline3",
    "tabs\tand\tspaces \t mix", "I'm, you're, he'll, we'd, they've, it's, I'M", "123 4567 89",
    "价格是 3 元，好的。", "é vs é (NFC should merge)", "emoji 🚀🚀 and ✨",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "x<|im_end|>y<think>z", "code: a[i]+=b[j]*2;", "    def f():\n        pass\n", "!!!???...",
    "mixed 中文English123混合", "", " ", "\n\n", "a" * 50, "Ελληνικά русский 한국어",
    # NUL is text: NFC goes on after it
    "a\u0000bcd", "12\u000034", "\u0000", "e\u0301\u0000e\u0301", "abc\u0000<|im_end|>xyz",
]
# MiniCPM5's pre-tokenizer cuts numbers into groups of three from the left, then splits each piece by itself: runs
# of digits, numbers of every script, numbers next to spaces, letters and punctuation.
NUMBERS = [
    "1234567890", "12 345 6789 0", "  12", "x 12", "a1b22c333d4444e55555", "3.14159265", "-42.5e-10",
    "Q3 2026: revenue 1,234,567.89 USD (+12.5%)", "①②③④⑤ ⅫⅩⅠ ½¾ x²³", "２０２６年１１月１７日", "٣٤٥٦ ۱۲۳۴",
    "A farm has 1250 hens; each lays 6 eggs a week. Eggs sell at $3.75 a dozen. How much in 52 weeks?",
    "x=12;y=345\n\n  678\t9", "007 0000 00000", "'12 '123'", "1\n22\r\n333  4444",
]


def make_tokenizer(out_dir: str, minicpm5: bool = False):
    """Qwen3.5's tokenizer pipeline (NFC, its split), or MiniCPM5's (no normalizer; numbers in groups of three, then
    Qwen2's split with runs of numbers; its post-processor's <s> before a text).  Each case: the text's ids alone,
    and with the special tokens the post-processor adds (prompt_ids: a raw prompt's tokens)."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    tok = Tokenizer(models.BPE())
    if minicpm5:
        splits = [MINICPM5_NUMBERS, MINICPM5_SPLIT]
    else:
        tok.normalizer = normalizers.NFC()
        splits = [QWEN35_SPLIT]
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(r), behavior="isolated", invert=False) for r in splits] +
        [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=SPECIALS + (["<s>"] if minicpm5 else []),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(CORPUS + (NUMBERS * 10 if minicpm5 else []), trainer)
    tok.add_tokens(ADDED)
    if minicpm5:
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A", pair="<s> $A <s>:1 $B:1", special_tokens=[("<s>", tok.token_to_id("<s>"))])
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    texts = CASES + (NUMBERS + ["<s>x <s>", "y<s>"] if minicpm5 else []) + CORPUS[:8]
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids, "prompt_ids": tok.encode(t).ids}
             for t in texts]
    return tok, cases


def init_(model, gen):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2 and "embed" in name:
                p.copy_(torch.randn(p.shape, generator=gen))
            elif p.dim() >= 2:
                fan_in = p.shape[-1] if p.dim() == 2 else p.shape[-1] * p.shape[1]
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(fan_in) * (2.0 if "conv1d" in name else 1.0))
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.rand(p.shape, generator=gen) * 3 + 1))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
            else:  # zero-centred RMSNorm weights
                p.copy_(0.2 * torch.randn(p.shape, generator=gen))


def make_qwen3(out_dir: str, tok_dir: str, vocab: int, gen):
    """A tiny Qwen3 (dense: attention with q/k norms, plain RMSNorm weights, untied head)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    os.makedirs(out_dir, exist_ok=True)
    cfg = Qwen3Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}, max_position_embeddings=4096,
        eos_token_id=2)
    save_dense(Qwen3ForCausalLM(cfg).to(torch.float32), out_dir, tok_dir, gen)


def make_llama(out_dir: str, vocab: int, gen):
    """A tiny Llama shaped like MiniCPM5-1B's layers: no q/k norms, 16 query heads on 2 KV heads of
    128 (the whole head rotated, theta 5e6), plain RMSNorm weights, untied head.  Its tokenizer and template are
    already in out_dir."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=16, num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 5000000.0}, max_position_embeddings=4096,
        attention_bias=False, mlp_bias=False, eos_token_id=2)
    save_dense(LlamaForCausalLM(cfg).to(torch.float32), out_dir, None, gen)


def save_dense(model, out_dir: str, tok_dir: str | None, gen):
    """A tiny dense model's random weights, saved in bf16 with the tokenizer and template of tok_dir (None: there)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2 and ("embed" in name or "lm_head" in name):
                p.copy_(torch.randn(p.shape, generator=gen) * (1.0 if "embed" in name else 0.5))
            elif p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(p.shape[1]))
            else:  # RMSNorm weights, used as they are (not zero-centred)
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
    model.to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    json.dump({"eos_token_id": [2]}, open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8"))
    for f in ("tokenizer.json", "chat_template.jinja") if tok_dir else ():
        shutil.copy(os.path.join(tok_dir, f), os.path.join(out_dir, f))


def write_golden_hf(hf_dir: str, out_json: str, prompts, n=48, min_margin=0.02, lora=None):
    """Greedy tokens from transformers itself (float64 weights and activations), prompts with clear margins.  lora:
    (an export_dv.py --vlora directory's tensors, its scale): the target's own adapters merged into its weights in
    float64, W + scale B A (the same model as the engine's unmerged one, to float64's rounding)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float64).eval()
    if lora:
        tensors, scale = lora
        with torch.no_grad():
            for l, layer in enumerate(model.model.layers):
                for name, blk in dv._LINEARS:
                    w = getattr(layer.self_attn if blk == "attn" else layer.mlp, name + "_proj").weight
                    w += scale * (tensors[f"L{l}.{name}.lora_b"].double() @ tensors[f"L{l}.{name}.lora_a"].double())
    golden = {"eos": [2], "plain_only": True, "cases": []}
    with torch.no_grad():
        for pr in prompts:
            ids, margins = list(pr), []
            for _ in range(n):
                logits = model(torch.tensor([ids])).logits[0, -1]
                top = torch.topk(logits, 2)
                margins.append(float(top.values[0] - top.values[1]))
                ids.append(int(top.indices[0]))
            if min(margins) >= min_margin:
                golden["cases"].append({"prompt": list(pr), "tokens": ids[len(pr):], "min_margin": min(margins)})
    assert golden["cases"], "no prompt with a clear enough margin"
    json.dump(golden, open(out_json, "w", encoding="utf-8"), indent=0)
    return len(golden["cases"])


def make_target(out_dir: str, vocab: int, gen, gdn_heads=(2, 4), gdn_dims=(16, 16), hidden=64, inter=128):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=gdn_heads[0], linear_num_value_heads=gdn_heads[1],
        linear_key_head_dim=gdn_dims[0], linear_value_head_dim=gdn_dims[1],
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        max_position_embeddings=4096, eos_token_id=2)
    model = Qwen3_5ForCausalLM(cfg).to(torch.float32)
    init_(model, gen)
    model = model.to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    json.dump({"eos_token_id": [2]}, open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8"))
    make_mtp(out_dir, cfg, gen)
    return cfg


_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
_LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "linear_attn.in_proj_qkv",
           "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def quantize_nvfp4(w: torch.Tensor):
    """ModelOpt-style NVFP4: per-tensor scale2 = amax / (6 * 448); per 16 values an e4m3 scale; e2m1 values."""
    w = w.to(torch.float64)
    N, K = w.shape
    s2 = w.abs().max().clamp(min=1e-12) / (6.0 * 448.0)
    blocks = w.view(N, K // 16, 16)
    sb = (blocks.abs().amax(-1) / 6.0 / s2).clamp(max=448.0).to(torch.float8_e4m3fn)
    sbv = sb.to(torch.float64).clamp(min=2.0 ** -9)
    x = blocks / (sbv[..., None] * s2)
    idx = (x.abs()[..., None] - _E2M1).abs().argmin(-1)              # nearest magnitude
    code = (idx | ((x < 0) & (idx > 0)).to(torch.long) * 8).view(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, sb, torch.tensor(float(s2), dtype=torch.float32)


def make_nvfp4_dir(src: str, dst: str):
    """A copy of a HuggingFace directory with the target's linears in ModelOpt's NVFP4 layout."""
    import shutil
    from safetensors.torch import load_file, save_file
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith(".safetensors"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                  "producer": {"name": "flyx make_tiny"}}
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=1)
    for f in os.listdir(src):
        if not f.endswith(".safetensors"):
            continue
        out = {}
        for k, v in load_file(os.path.join(src, f)).items():
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            if base and not k.startswith("mtp.") and any(base.endswith(l) for l in _LINEAR):
                q, sc, s2 = quantize_nvfp4(v)
                out[k] = q
                out[base + ".weight_scale"] = sc
                out[base + ".weight_scale_2"] = s2
                out[base + ".input_scale"] = torch.tensor(1.0)
            else:
                out[k] = v
        save_file({k: t.contiguous() for k, t in out.items()}, os.path.join(dst, f))


def make_mtp(out_dir: str, cfg, gen):
    """A random multi-token-prediction layer in its own file (transformers does not save one)."""
    from safetensors.torch import save_file
    H, I, nh, nkv, hd = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, \
        cfg.head_dim
    r = lambda *sh, scale=1.0: torch.randn(*sh, generator=gen) * scale  # noqa: E731
    W = {"mtp.fc.weight": r(H, 2 * H, scale=(2 * H) ** -0.5),
         "mtp.pre_fc_norm_embedding.weight": 0.2 * r(H), "mtp.pre_fc_norm_hidden.weight": 0.2 * r(H),
         "mtp.norm.weight": 0.2 * r(H),
         "mtp.layers.0.input_layernorm.weight": 0.2 * r(H), "mtp.layers.0.post_attention_layernorm.weight": 0.2 * r(H),
         "mtp.layers.0.self_attn.q_proj.weight": r(2 * nh * hd, H, scale=H ** -0.5),
         "mtp.layers.0.self_attn.k_proj.weight": r(nkv * hd, H, scale=H ** -0.5),
         "mtp.layers.0.self_attn.v_proj.weight": r(nkv * hd, H, scale=H ** -0.5),
         "mtp.layers.0.self_attn.o_proj.weight": r(H, nh * hd, scale=(nh * hd) ** -0.5),
         "mtp.layers.0.self_attn.q_norm.weight": 0.2 * r(hd), "mtp.layers.0.self_attn.k_norm.weight": 0.2 * r(hd),
         "mtp.layers.0.mlp.gate_proj.weight": r(I, H, scale=H ** -0.5),
         "mtp.layers.0.mlp.up_proj.weight": r(I, H, scale=H ** -0.5),
         "mtp.layers.0.mlp.down_proj.weight": r(H, I, scale=I ** -0.5)}
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in W.items()}, os.path.join(out_dir, "mtp.safetensors"))


def make_draft(out_dir: str, vocab: int, mask_id: int, gen, H=64, layers=2, heads=4, kv=2, hd=32,
               inter=128, taps=(2, 5, 7), block=16, ktaps=2, group=16, rank=16, topk=16):
    from safetensors.torch import save_file
    G = H // group
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=gen) * scale)  # noqa: E731
    W = {"fc.weight": r(H, len(taps) * H, scale=1 / math.sqrt(len(taps) * H)),
         "hidden_norm.weight": 1 + 0.2 * r(H), "norm.weight": 1 + 0.2 * r(H),
         "candidate_selector.predecessor_codebook": r(vocab, rank, scale=0.5),
         "candidate_selector.successor_codebook": r(vocab, rank, scale=0.5),
         "candidate_selector.hidden_projection.weight": r(rank, H, scale=1 / math.sqrt(H))}
    for j in range(layers):
        p = f"layers.{j}."
        W.update({
            p + "input_layernorm.weight": 1 + 0.2 * r(H), p + "post_attention_layernorm.weight": 1 + 0.2 * r(H),
            p + "self_attn.q_proj.weight": r(heads * hd, H, scale=H ** -0.5),
            p + "self_attn.k_proj.weight": r(kv * hd, H, scale=H ** -0.5),
            p + "self_attn.v_proj.weight": r(kv * hd, H, scale=H ** -0.5),
            p + "self_attn.o_proj.weight": r(H, heads * hd, scale=(heads * hd) ** -0.5),
            p + "self_attn.q_norm.weight": 1 + 0.2 * r(hd), p + "self_attn.k_norm.weight": 1 + 0.2 * r(hd),
            p + "mlp.gate_proj.weight": r(inter, H, scale=H ** -0.5),
            p + "mlp.up_proj.weight": r(inter, H, scale=H ** -0.5),
            p + "mlp.down_proj.weight": r(H, inter, scale=inter ** -0.5)})
        for c in ("attention_conv", "mlp_conv"):
            base = 0.3 * r(2, ktaps, H)
            base[:, 0] += 1.0
            W[p + c + ".base_kernel"] = base
            W[p + c + ".kernel_projection.weight"] = r(2 * ktaps * G, H, scale=0.3 * H ** -0.5)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in W.items()}, os.path.join(out_dir, "model.safetensors"))
    cfg = {"architectures": ["DFlash2DraftModel"], "speculators_model_type": "dflash2", "block_size": block,
           "mask_token_id": mask_id, "aux_hidden_state_layer_ids": list(taps), "draft_vocab_size": vocab,
           "conv_kernel_size": ktaps, "conv_group_size": group, "selector_rank": rank, "selector_top_k": topk,
           "sample_from_anchor": False, "sliding_window_non_causal": True, "target_hidden_size": None,
           "transformer_layer_config": {
               "model_type": "qwen3", "hidden_size": H, "num_hidden_layers": layers,
               "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
               "intermediate_size": inter, "rms_norm_eps": 1e-6, "vocab_size": vocab,
               "layer_types": ["full_attention"] * layers,
               "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}}}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=1)


def make_dspine(out_dir: str, vocab: int, mask_id: int, target_layers: int, gen, H=64, layers=3, heads=4, kv=2,
                hd=32, inter=128, taps=(1, 4, 7), block=16, code=32, rank=16, topk=16, window=0):
    """A random DSpine drafter in its authors' layout (dflash_config with the code bridge; the bridge's tensors under
    selected_prefix_bridge., the layers' reads and writes stacked): what python/convert/flyx_convert/dspine.py
    reads.  Its own generator, so the assets made before it stay what they were."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=gen) * scale)  # noqa: E731
    one = lambda: torch.ones((), dtype=torch.int64)  # noqa: E731  (a tensor of its own each)
    W = {"fc.weight": r(H, len(taps) * H, scale=1 / math.sqrt(len(taps) * H)),
         "hidden_norm.weight": 1 + 0.2 * r(H), "norm.weight": 1 + 0.2 * r(H),
         "draft_block_causal_version": one(), "last_write_refine_version": one()}
    b = "selected_prefix_bridge."
    W.update({b + "codebook": r(vocab, code), b + "read_weight": r(code, H, scale=H ** -0.5),
              b + "early_read_weights": r(layers - 1, code, H, scale=H ** -0.5),
              # messages of the size of the rows they join: they move drafts, as a trained bridge's do
              b + "feature_write_weights": r(layers, H, rank, scale=1.5 * rank ** -0.5),
              b + "gate.weight": r(rank, code, scale=code ** -0.5), b + "gate.bias": 0.5 * r(rank),
              b + "predecessor_gate.weight": r(rank, code, scale=code ** -0.5),
              b + "prev.weight": r(rank, code, scale=code ** -0.5), b + "out.weight": torch.zeros(code, rank),
              b + "independent_write_version": one(), b + "terminal_disabled_version": one()})
    for j in range(layers):
        p = f"layers.{j}."
        W.update({
            p + "input_layernorm.weight": 1 + 0.2 * r(H), p + "post_attention_layernorm.weight": 1 + 0.2 * r(H),
            p + "self_attn.q_proj.weight": r(heads * hd, H, scale=H ** -0.5),
            p + "self_attn.k_proj.weight": r(kv * hd, H, scale=H ** -0.5),
            p + "self_attn.v_proj.weight": r(kv * hd, H, scale=H ** -0.5),
            p + "self_attn.o_proj.weight": r(H, heads * hd, scale=(heads * hd) ** -0.5),
            p + "self_attn.q_norm.weight": 1 + 0.2 * r(hd), p + "self_attn.k_norm.weight": 1 + 0.2 * r(hd),
            p + "mlp.gate_proj.weight": r(inter, H, scale=H ** -0.5),
            p + "mlp.up_proj.weight": r(inter, H, scale=H ** -0.5),
            p + "mlp.down_proj.weight": r(H, inter, scale=inter ** -0.5)})
    save_file({k: (v if v.dtype == torch.int64 else v.to(torch.bfloat16)).contiguous() for k, v in W.items()},
              os.path.join(out_dir, "model.safetensors"))
    cfg = {"architectures": ["DSpineDraftModel"], "model_type": "qwen3", "attention_bias": False, "block_size": block,
           "hidden_act": "silu", "hidden_size": H, "intermediate_size": inter, "num_hidden_layers": layers,
           "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd, "rms_norm_eps": 1e-6,
           "vocab_size": vocab, "draft_vocab_size": vocab, "num_target_layers": target_layers,
           "max_position_embeddings": 4096, "tie_word_embeddings": False,
           "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}, "rope_theta": 1000000.0,
           "layer_types": ["sliding_attention" if window else "full_attention"] * layers,
           "use_sliding_window": bool(window), "sliding_window": window or None, "max_window_layers": layers,
           "dflash_config": {
               "draft_block_causal": True, "mask_token_id": mask_id, "target_layer_ids": [t - 1 for t in taps],
               "selected_prefix_code_bridge": {"code_size": code, "eps": 1e-6, "kappa": 10.0, "rank": rank},
               "selected_prefix_decode_mode": "serial", "selected_prefix_feature_gt_training": True,
               "selected_prefix_feature_layers": list(range(1, layers + 1)),
               "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
               "selected_prefix_last_write_refine": {"head_topk": topk, "kind": "exact_sender", "version": 1,
                                                     "weight": 1.0},
               "selected_prefix_terminal": False}}
    if window:  # as the 27B's drafter says it (the second pass over the last write; the same width)
        cfg["code_bridge_second_pass"] = {"first_exact": layers - 1, "head_topk": topk, "mode": "serial_table"}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=1)


def zlab_drafter(src: str, dst: str, target_layers: int, window: int = 0, block: int = 0):
    """The drafter in src (speculators' layout) as z-lab publishes DFlash2 drafters: the same tensors, its settings
    in dflash_config, target_layer_ids the layers whose outputs it reads (speculators' indices - 1), with a
    window every layer a non-causal sliding one, and with a block that block size (the 27B's drafter's is 8)."""
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "model.safetensors"), dst)
    sc = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    tl = sc["transformer_layer_config"]
    cfg = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "is_causal": False,
           "num_target_layers": target_layers, "use_sliding_window": bool(window),
           "sliding_window": window or None,
           "layer_types": ["sliding_attention" if window else "full_attention"] * tl["num_hidden_layers"],
           "dflash_config": {"block_size": block or sc["block_size"], "mask_token_id": sc["mask_token_id"],
                             "conv_kernel_size": sc["conv_kernel_size"], "conv_group_size": sc["conv_group_size"],
                             "selector_rank": sc["selector_rank"], "selector_top_k": sc["selector_top_k"],
                             "target_layer_ids": [t - 1 for t in sc["aux_hidden_state_layer_ids"]]}}
    cfg.update({k: v for k, v in tl.items() if k not in ("layer_types",)})
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=1)


def same_drafter(model_path: str, drafter_path: str):
    """A drafter file holds what the model file holds of its drafter: the same config and tensors."""
    with Reader(model_path) as a, Reader(drafter_path) as b:
        assert a.header["draft"] == b.header["draft"], (a.header["draft"], b.header["draft"])
        names = sorted(n for n in a.header["tensors"] if n.startswith("draft."))
        assert names == sorted(b.header["tensors"]), "the drafter file's tensors differ from the model's drafter"
        for n in names:
            assert bytes(a.raw(a.header["tensors"][n])) == bytes(b.raw(b.header["tensors"][n])), n


def main(out):
    os.makedirs(out, exist_ok=True)
    made_from = inputs_digest()  # before anything is made: an input edited meanwhile makes them again next time
    if os.path.exists(os.path.join(out, STAMP)):
        os.remove(os.path.join(out, STAMP))  # a run that fails leaves no stamp
    gen = torch.Generator().manual_seed(1234)
    tdir, ddir = os.path.join(out, "hf_target"), os.path.join(out, "hf_draft")
    os.makedirs(tdir, exist_ok=True); os.makedirs(ddir, exist_ok=True)
    tok, cases = make_tokenizer(tdir)
    with open(os.path.join(tdir, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write("{# tiny #}<|im_start|>{{ messages }}<think>")
    vocab = 768
    assert tok.get_vocab_size() <= vocab
    make_target(tdir, vocab, gen)
    make_draft(ddir, vocab, tok.token_to_id("<|mask|>"), gen)
    path = os.path.join(out, "tiny.flyx")
    convert(tdir, path, ddir, log=lambda *_: None)
    with open(os.path.join(out, "tokenizer_cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=0)
    n_chat = write_chat_cases(tdir, os.path.join(out, "chat_cases.json"))

    prompts = [tok.encode(t).ids for t in ("The quick brown fox", "def add(a, b):\n", "推理引擎需要")]
    prompts.append([5, 17, 33, 9])
    write_golden(path, os.path.join(out, "golden.json"), prompts, vocab)
    # The drafter as z-lab publishes DFlash2 drafters, converted alone: what tiny.flyx holds of
    # its drafter.  Then with a sliding window of 8, shorter than the prompts and their 48 tokens: with the target
    # (golden drafts from the reference's window) and alone.
    layers = json.load(open(os.path.join(tdir, "config.json"), encoding="utf-8"))
    layers = layers.get("text_config", layers)["num_hidden_layers"]
    zdir, wdir = os.path.join(out, "hf_draft_zlab"), os.path.join(out, "hf_draft_window")
    zlab_drafter(ddir, zdir, layers)
    convert_draft(zdir, os.path.join(out, "tiny_draft.flyx"), log=lambda *_: None)
    same_drafter(path, os.path.join(out, "tiny_draft.flyx"))
    zlab_drafter(ddir, wdir, layers, window=8)
    wpath = os.path.join(out, "tiny_window.flyx")
    convert(tdir, wpath, wdir, log=lambda *_: None)
    write_golden(wpath, os.path.join(out, "golden_window.json"), prompts, vocab)
    convert_draft(wdir, os.path.join(out, "tiny_draft_window.flyx"), log=lambda *_: None)
    # Chains of 15 widen the 8-row block. With 7 native rows, two widened chains of 7 fit in one batch.
    for block in (8, 7):
        bdir = os.path.join(out, f"hf_draft_block{block}")
        zlab_drafter(ddir, bdir, layers, block=block)
        convert(tdir, os.path.join(out, f"tiny_block{block}.flyx"), bdir, log=lambda *_: None)
    # The same model with NVFP4 linears (as NVIDIA ModelOpt stores them).
    qdir = os.path.join(out, "hf_target_nvfp4")
    make_nvfp4_dir(tdir, qdir)
    qpath = os.path.join(out, "tiny_nvfp4.flyx")
    # Its layers are 64 wide, below the W4A4 kernels' tile of 128 values of K: it declares bf16 inputs, which the
    # golden outputs below take.
    convert(qdir, qpath, ddir, log=lambda *_: None, activations="bf16")
    # NVFP4 linears take bf16-rounded inputs (the engine's activation policy, which
    # the reference reproduces); float32 vs float64 activations can still round
    # differently, moving logits by ~1e-3, so these cases avoid near-ties.
    more = [[7, 3, 99, 12], [1, 2, 3], [300, 301, 302, 303, 304], [42], [11, 500, 7], [64, 65, 66, 67, 68, 69]]
    write_golden(qpath, os.path.join(out, "golden_nvfp4.json"), prompts + more, vocab, min_margin=0.05, keep=4)
    # A second architecture through models/ (Qwen3): no drafter, golden from transformers.
    q3dir = os.path.join(out, "hf_qwen3")
    make_qwen3(q3dir, tdir, vocab, gen)
    q3path = os.path.join(out, "tiny_qwen3.flyx")
    convert(q3dir, q3path, None, log=lambda *_: None)
    n3 = write_golden_hf(q3dir, os.path.join(out, "golden_qwen3.json"), prompts + more)
    # GDN heads of 128 (Qwen3.5's): prefill takes the chunked recurrence on bf16
    # tensor cores, so these cases avoid near-ties as the NVFP4 ones do.
    gdir = os.path.join(out, "hf_gdn128")
    os.makedirs(gdir, exist_ok=True)
    for f in os.listdir(tdir):
        if f.endswith((".json", ".jinja", ".txt")) and not f.startswith(("config", "generation_config")):
            shutil.copy(os.path.join(tdir, f), gdir)
    make_target(gdir, vocab, gen, gdn_heads=(1, 2), gdn_dims=(128, 128))
    gpath = os.path.join(out, "tiny_gdn128.flyx")
    convert(gdir, gpath, ddir, log=lambda *_: None)
    write_golden(gpath, os.path.join(out, "golden_gdn128.json"), prompts + more, vocab, min_margin=0.05, keep=4)
    # NVFP4 linears that take NVFP4 inputs in passes of more than 16 rows (the W4A4 kernels): 128
    # wide, with a value head of 32 and an MLP of 256, so every linear's K is a whole number of the kernels' tiles
    # of 128.  No golden outputs: gguf_e2e_test holds the CUDA backend to the float64 one on this file.
    wdir, wq = os.path.join(out, "hf_w4a4"), os.path.join(out, "hf_w4a4_nvfp4")
    os.makedirs(wdir, exist_ok=True)
    for f in os.listdir(tdir):
        if f.endswith((".json", ".jinja", ".txt")) and not f.startswith(("config", "generation_config")):
            shutil.copy(os.path.join(tdir, f), wdir)
    make_target(wdir, vocab, gen, gdn_dims=(32, 32), hidden=128, inter=256)
    make_nvfp4_dir(wdir, wq)
    convert(wq, os.path.join(out, "tiny_w4a4.flyx"), None, log=lambda *_: None, activations="nvfp4")
    # The same weights in a file that says bf16 inputs: --activations nvfp4 gives it the first file's NVFP4 ones.
    convert(wq, os.path.join(out, "tiny_w4a4_a16.flyx"), None, log=lambda *_: None, activations="bf16")
    # Llama's attention without q/k norms (MiniCPM5's layers), with MiniCPM5's pre-tokenizer: no
    # drafter, golden from transformers.  Made last, so the random weights above stay what they were.
    ldir = os.path.join(out, "hf_llama")
    os.makedirs(ldir, exist_ok=True)
    _, lcases = make_tokenizer(ldir, minicpm5=True)
    shutil.copy(os.path.join(tdir, "chat_template.jinja"), ldir)
    with open(os.path.join(out, "tokenizer_cases_minicpm5.json"), "w", encoding="utf-8") as f:
        json.dump(lcases, f, ensure_ascii=False, indent=0)
    make_llama(ldir, vocab, gen)
    convert(ldir, os.path.join(out, "tiny_llama.flyx"), None, log=lambda *_: None)
    nl = write_golden_hf(ldir, os.path.join(out, "golden_llama.json"), prompts + more)
    write_dv_cases(os.path.join(out, "dv_cases.json"))
    make_llama_dv(out, ldir)
    nlora = make_llama_lora(out, ldir, prompts + more)
    # The tiny Qwen3.5 with a DSpine drafter in place of DFlash2's (its MTP layer kept), and with a sliding window of
    # 8; the drafter alone as a drafter file.  Its own generator: the files above stay what they were.
    sgen = torch.Generator().manual_seed(4321)
    sdir, swdir = os.path.join(out, "hf_dspine"), os.path.join(out, "hf_dspine_window")
    mask = tok.token_to_id("<|mask|>")
    make_dspine(sdir, vocab, mask, layers, sgen)
    spath = os.path.join(out, "tiny_dspine.flyx")
    convert(tdir, spath, sdir, log=lambda *_: None)
    write_golden(spath, os.path.join(out, "golden_dspine.json"), prompts + more, vocab, keep=4)
    make_dspine(swdir, vocab, mask, layers, sgen, window=8)
    swpath = os.path.join(out, "tiny_dspine_window.flyx")
    convert(tdir, swpath, swdir, log=lambda *_: None)
    write_golden(swpath, os.path.join(out, "golden_dspine_window.json"), prompts + more, vocab, keep=4)
    convert_draft(sdir, os.path.join(out, "tiny_draft_dspine.flyx"), log=lambda *_: None)
    same_drafter(spath, os.path.join(out, "tiny_draft_dspine.flyx"))
    # The tiny Qwen3 with a DSpine drafter of its own (taps after its layers 1, 2 and 3 of 4): held to transformers'
    # tokens (golden_qwen3.json) with it drafting, as a second architecture's block drafter.
    q3s = os.path.join(out, "hf_qwen3_dspine")
    make_dspine(q3s, vocab, mask, 4, sgen, taps=(1, 2, 3))
    convert(q3dir, os.path.join(out, "tiny_qwen3_dspine.flyx"), q3s, log=lambda *_: None)
    # A DSpine drafter of one layer: no stacked early reads (early_read_weights [0, code, hidden]), its one cache
    # binary16 beside fp8 target caches.
    s1dir, s1path = os.path.join(out, "hf_dspine1"), os.path.join(out, "tiny_dspine1.flyx")
    make_dspine(s1dir, vocab, mask, layers, sgen, layers=1)
    convert(tdir, s1path, s1dir, log=lambda *_: None)
    write_golden(s1path, os.path.join(out, "golden_dspine1.json"), prompts + more, vocab, keep=4)
    convert_draft(s1dir, os.path.join(out, "tiny_draft_dspine1.flyx"), log=lambda *_: None)
    cases = {name: json.load(open(os.path.join(out, golden), encoding="utf-8"))["cases"]
             for name, golden in (("", "golden_llama.json"), ("tiny_llama_dv_lora.flyx", "golden_llama_dv_lora.json"),
                                  ("tiny_llama_dv31_lora.flyx", "golden_llama_dv_lora.json"))}
    nd = write_dv_rounds(out, cases)
    # The tiny Llama's linears in NVFP4 (ModelOpt's layout, bf16 inputs): rounds of more than 16 rows on its NVFP4
    # GEMMs (gemm_fp4_rows).  The reference has no Llama of NVFP4 weights: golden_llama_nvfp4.json gives the prompts
    # alone, and golden_test holds the file to its own plain decoding.  No random numbers: the files above stay.
    lq = os.path.join(out, "hf_llama_nvfp4")
    make_nvfp4_dir(ldir, lq)
    convert(lq, os.path.join(out, "tiny_llama_nvfp4.flyx"), None, log=lambda *_: None, activations="bf16")
    with open(os.path.join(out, "golden_llama.json"), encoding="utf-8") as f:
        lprompts = [c["prompt"] for c in json.load(f)["cases"]]
    with open(os.path.join(out, "golden_llama_nvfp4.json"), "w", encoding="utf-8") as f:
        json.dump({"eos": [2], "plain_only": True, "cases": [{"prompt": p} for p in lprompts]}, f, indent=0)
    # The same NVFP4 linears with tiny_llama_lora.flyx's V-LoRA (bf16 adapters beside NVFP4 weights), held to its own
    # plain decoding as the file without it is.
    convert(lq, os.path.join(out, "tiny_llama_nvfp4_lora.flyx"), None, log=lambda *_: None, activations="bf16",
            lora_dir=os.path.join(out, "dv_tiny", "export_vlora"))
    with open(os.path.join(out, "golden_llama_nvfp4_lora.json"), "w", encoding="utf-8") as f:
        json.dump({"eos": [2], "plain_only": True, "cases": [{"prompt": p} for p in lprompts]}, f, indent=0)
    with open(os.path.join(out, STAMP), "w", encoding="utf-8") as f:
        f.write(made_from + "\n")
    print(f"wrote {out}: tiny.flyx, tiny_nvfp4.flyx, golden.json, golden_nvfp4.json ({len(prompts)} prompts), "
          f"tokenizer_cases.json ({len(cases)}), chat_cases.json ({n_chat}), tiny_qwen3.flyx, "
          f"golden_qwen3.json ({n3} prompts), tiny_llama.flyx, golden_llama.json ({nl} prompts), "
          f"tokenizer_cases_minicpm5.json ({len(lcases)}), "
          f"tiny_gdn128.flyx, golden_gdn128.json, tiny_draft.flyx, tiny_block8.flyx, tiny_block7.flyx, "
          f"tiny_window.flyx, golden_window.json, "
          f"tiny_draft_window.flyx, tiny_w4a4.flyx, tiny_w4a4_a16.flyx, dv_cases.json, tiny_llama_dv.flyx, "
          f"tiny_llama_dv_pool.flyx, tiny_llama_dv32.flyx, tiny_llama_dv31.flyx, tiny_llama_dv31_pool.flyx, "
          f"tiny_llama_dv31_wide.flyx, tiny_llama_dv31_wide_pool.flyx, golden_llama_dv.json ({nd} rounds), "
          f"tiny_llama_nvfp4.flyx, golden_llama_nvfp4.json, tiny_llama_lora.flyx, golden_llama_lora.json and "
          f"golden_llama_dv_lora.json ({nlora[0]} and {nlora[1]} prompts), tiny_llama_dv_lora.flyx, "
          f"tiny_llama_dv31_lora.flyx, "
          f"tiny_llama_nvfp4_lora.flyx, golden_llama_nvfp4_lora.json, tiny_dspine.flyx, golden_dspine.json, "
          f"tiny_dspine_window.flyx, golden_dspine_window.json, tiny_draft_dspine.flyx, tiny_qwen3_dspine.flyx, "
          f"tiny_dspine1.flyx, golden_dspine1.json, tiny_draft_dspine1.flyx")


def make_llama_dv(out: str, ldir: str):
    """The tiny Llama with a tiny DV drafter (8 slots, adapter rank 32 and selector rank 16: ranks a reader that mixes
    them up gets wrong; a beam tree of 15 drafts, the tokenizer's <|mask|> as MASK), its random weights as
    tests/python/reference/dv_test.py makes them, through the converter's whole path: bifill-dv's checkpoints
    (dv_tiny/canvas.pt, selector.pt), python/convert/export_dv.py (dv_tiny/export/), flyx_convert --dv
    (tiny_llama_dv.flyx); and the same drafter with a pool tree, ST 15/3/1 (dv_tiny/export_pool/,
    tiny_llama_dv_pool.flyx: its tree fields are where another drafter's block has others, so a reader that takes its
    block for another kind's gets numbers a beam tree's zeros hide).  Its own generator: nothing above changes.  The
    selector's A, H and B are a quarter of random_dv's, so that its scores (their bilinear term near 16) and alpha
    log p (a few) both decide the trees golden_test holds the CUDA rounds' to: at random_dv's scale the bilinear term
    near 1e3 decides them alone (alpha 0 moves 2 of 48 of the reference's trees; at a quarter, 38).  The pool file's
    fill is 0.4, where a random canvas keeps most of its top tokens (probabilities near 0.15 to 0.99) as its next
    inputs, so that its rounds run the canvas input's token path; the beam file's, bifill-dv's 0.9, keeps none.  The
    pool file's canvas is causal (dv_tiny/canvas_causal.pt: a slot sees the slots before it alone).
    tiny_llama_dv32.flyx (dv_tiny/export_32/): the beam file's canvas with 32 slots, MiniCPM5's (more than 16: the
    CUDA backend's prefill GEMMs, two chunks of the canvas input, 32 rows of canvas attention; not causal, so that the
    slots past the 15 a tree reaches change it), and a beam of 8 with coverage over its 15 depths.
    tiny_llama_dv31.flyx and tiny_llama_dv31_pool.flyx (dv_tiny/export_31, export_31_pool): the same canvas of 32 slots
    drafting MiniCPM5-1B's DV files' trees, 31 drafts a round over 31 depths (rounds of 32 verify rows, wide rows):
    a beam of 8 with coverage (DYN 31:8:cov, the res2 file's) and a pool of 7 ranks (ST 31/7/1, the dp9k file's),
    bifill-dv's fill 0.9.  On this canvas the selector's tables are steep: a tree of 31 grows deep and narrow, the
    beam's nodes past the fourth and the pool's ranks past the third hardly ever among the 31 best (the CUDA backend's
    first trees after fresh starts, drafting a beam of 7 or 4 or a pool of 6, 3 or 1 ranks from these files, keep 72%
    to 77% of the float64 reference's drafts, as with the files' own trees), and past 16 slots the trees turn on the
    canvas's bf16 inputs (their drafts are held, not the trees).
    tiny_llama_dv31_wide.flyx and tiny_llama_dv31_wide_pool.flyx (dv_tiny/export_31w, export_31w_pool) hold those
    trees whole: the same two trees of 31 on 16 slots (15 depths; the canvas's GEMMs keep float32's sums there), from
    a selector whose scores are an eighth of the one above (dv_tiny/selector_wide.pt: H and alpha divided by 8, exactly
    a temperature of 8), whose flat tables fill the beam's 8 nodes a depth and reach the pool's 7th rank: of the
    float64 reference's 184 first trees after fresh starts a beam of 7 leaves 43 as they are, coverage left out 100, a
    pool of 6 ranks 120, while the reference in float32 keeps 183 and 184."""
    import export_dv
    sys.path.insert(0, os.path.join(ROOT, "tests", "python", "reference"))
    from dv_test import random_dv
    cfg, _ = dv.hf_llama(ldir)
    W = random_dv(cfg, rank=32, sel_rank=16, gen=torch.Generator().manual_seed(32))
    adapter = {"norm": W["dv.fnormc"], "mask_emb": W["dv.mask_emb"], "res_in.weight": W["dv.res_in"],
               "res_out.weight": W["dv.res_out"]}
    for layer in range(cfg.layers):
        p = f"layers.{layer}."
        adapter[p + "ln1"], adapter[p + "ln2"] = W[f"dv.L{layer}.ln1c"], W[f"dv.L{layer}.ln2c"]
        for n, _ in dv._LINEARS:
            adapter[p + n + ".A"], adapter[p + n + ".B"] = W[f"dv.L{layer}.{n}.lora_a"], W[f"dv.L{layer}.{n}.lora_b"]
    canvas = {"cfg": {"n": 8, "rank": 32, "residual": True, "res_k": 16, "canvas_causal": False, "mixers": [],
                      "tau": 0.9}, "adapter": adapter}
    selector = {"sel": {"A.weight": W["dv.sel.A"] / 4, "H.weight": W["dv.sel.H"] / 4, "B.weight": W["dv.sel.B"] / 4,
                        "alpha": W["dv.sel.alpha"].reshape(())}, "r": 16}
    ddir = os.path.join(out, "dv_tiny")
    os.makedirs(ddir, exist_ok=True)
    torch.save(canvas, os.path.join(ddir, "canvas.pt"))
    torch.save(selector, os.path.join(ddir, "selector.pt"))
    wide = dict(selector, sel=dict(selector["sel"], **{"H.weight": selector["sel"]["H.weight"] / 8,
                                                       "alpha": selector["sel"]["alpha"] / 8}))
    torch.save(wide, os.path.join(ddir, "selector_wide.pt"))
    causal = dict(canvas, cfg=dict(canvas["cfg"], canvas_causal=True))
    torch.save(causal, os.path.join(ddir, "canvas_causal.pt"))
    for pt, sel, slots, tree, fill, export, name in (
            ("canvas.pt", "selector.pt", 8, "beam:15:4:cov", "0.9", "export", "tiny_llama_dv.flyx"),
            ("canvas_causal.pt", "selector.pt", 8, "pool:15:3:1", "0.4", "export_pool", "tiny_llama_dv_pool.flyx"),
            ("canvas.pt", "selector.pt", 32, "beam:15:8:cov", "0.9", "export_32", "tiny_llama_dv32.flyx"),
            ("canvas.pt", "selector.pt", 32, "beam:31:8:cov", "0.9", "export_31", "tiny_llama_dv31.flyx"),
            ("canvas.pt", "selector.pt", 32, "pool:31:7:1", "0.9", "export_31_pool", "tiny_llama_dv31_pool.flyx"),
            ("canvas.pt", "selector_wide.pt", 16, "beam:31:8:cov", "0.9", "export_31w", "tiny_llama_dv31_wide.flyx"),
            ("canvas.pt", "selector_wide.pt", 16, "pool:31:7:1", "0.9", "export_31w_pool",
             "tiny_llama_dv31_wide_pool.flyx")):
        export_dv.main(["--canvas", os.path.join(ddir, pt), "--selector", os.path.join(ddir, sel),
                        "--slots", str(slots), "--tree", tree, "--fill", fill, "--mask-token", "<|mask|>",
                        "-o", os.path.join(ddir, export)])
        convert(ldir, os.path.join(out, name), None, log=lambda *_: None, dv_dir=os.path.join(ddir, export))


def random_vlora(cfg, rank: int, seed: int) -> dict:
    """A V-LoRA for the tiny Llama in bifill-dv's raft.py checkpoint format ({'lora': per layer q, k, v, o, gate, up,
    down of {'A' [rank, in], 'B' [out, rank]} float32, 'cfg': {'r'}}), its product about half the linear's, so that
    it moves the tokens (a missing adapter, a wrong one or a wrong scale changes them)."""
    gen = torch.Generator().manual_seed(seed)
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv * cfg.head_dim
    dims = {"q": (q, cfg.hidden), "k": (kv, cfg.hidden), "v": (kv, cfg.hidden), "o": (cfg.hidden, q),
            "gate": (cfg.inter, cfg.hidden), "up": (cfg.inter, cfg.hidden), "down": (cfg.hidden, cfg.inter)}
    lora = []
    for _ in range(cfg.layers):
        for n, _blk in dv._LINEARS:
            out, inp = dims[n]
            lora.append({"A": torch.randn(rank, inp, generator=gen) / math.sqrt(inp),
                         "B": 0.5 * torch.randn(out, rank, generator=gen) / math.sqrt(rank)})
    return {"lora": lora, "cfg": {"r": rank, "name": f"tiny_r{rank}"}, "step": 0}


def make_llama_lora(out: str, ldir: str, prompts) -> tuple[int, int]:
    """The tiny Llama with a V-LoRA (the target's own adapters, unmerged, on every linear of every pass: ABI 43)
    through the converter's whole path: bifill-dv's checkpoint format (dv_tiny/vlora16.pt, vlora32.pt; random_vlora),
    python/convert/export_dv.py --vlora (dv_tiny/export_vlora/: rank 16 with --vlora-scale 0.5, so that the scale's
    path is held too; dv_tiny/export_dv_lora/: tiny_llama_dv.flyx's canvas and selector, 8 slots, beam:15:4:cov, beside
    a V-LoRA of rank 32 and bifill-dv's scale 1; dv_tiny/export_dv31_lora/: tiny_llama_dv31.flyx's DV, 32 slots and
    trees of 31 drafts, beam:31:8:cov, beside the same V-LoRA: rounds of 32 verify rows whose adapter GEMMs run beside
    the canvas's grouped ones, MiniCPM5-1B's g_b2h line), flyx_convert --lora and --dv: tiny_llama_lora.flyx,
    tiny_llama_dv_lora.flyx and tiny_llama_dv31_lora.flyx; and their golden tokens from transformers in float64 with
    the adapters merged (golden_llama_lora.json, golden_llama_dv_lora.json, which both DV files with the V-LoRA of rank
    32 share; write_golden_hf).  The V-LoRAs change the plain tokens.  Returns the prompts each golden file kept.  Its
    own generators: nothing above changes."""
    import export_dv
    from safetensors.torch import load_file
    cfg, _ = dv.hf_llama(ldir)
    ddir = os.path.join(out, "dv_tiny")
    os.makedirs(ddir, exist_ok=True)
    for rank, seed in ((16, 16), (32, 33)):
        torch.save(random_vlora(cfg, rank, seed), os.path.join(ddir, f"vlora{rank}.pt"))
    export_dv.main(["--vlora", os.path.join(ddir, "vlora16.pt"), "--vlora-scale", "0.5",
                    "-o", os.path.join(ddir, "export_vlora")])
    export_dv.main(["--canvas", os.path.join(ddir, "canvas.pt"), "--selector", os.path.join(ddir, "selector.pt"),
                    "--slots", "8", "--tree", "beam:15:4:cov", "--fill", "0.9", "--mask-token", "<|mask|>",
                    "--vlora", os.path.join(ddir, "vlora32.pt"), "-o", os.path.join(ddir, "export_dv_lora")])
    export_dv.main(["--canvas", os.path.join(ddir, "canvas.pt"), "--selector", os.path.join(ddir, "selector.pt"),
                    "--slots", "32", "--tree", "beam:31:8:cov", "--fill", "0.9", "--mask-token", "<|mask|>",
                    "--vlora", os.path.join(ddir, "vlora32.pt"), "-o", os.path.join(ddir, "export_dv31_lora")])
    # Both DV files with the V-LoRA of rank 32 are held to golden_llama_dv_lora.json's tokens: their adapters' bytes.
    with open(os.path.join(ddir, "export_dv_lora", "vlora.safetensors"), "rb") as a, \
            open(os.path.join(ddir, "export_dv31_lora", "vlora.safetensors"), "rb") as b:
        if a.read() != b.read():
            raise RuntimeError("export_dv31_lora: its V-LoRA is not export_dv_lora's, whose golden tokens it takes")
    convert(ldir, os.path.join(out, "tiny_llama_lora.flyx"), None, log=lambda *_: None,
            lora_dir=os.path.join(ddir, "export_vlora"))
    for export, name in (("export_dv_lora", "tiny_llama_dv_lora.flyx"),
                         ("export_dv31_lora", "tiny_llama_dv31_lora.flyx")):
        convert(ldir, os.path.join(out, name), None, log=lambda *_: None, dv_dir=os.path.join(ddir, export))
    base = [c["tokens"] for c in json.load(open(os.path.join(out, "golden_llama.json"), encoding="utf-8"))["cases"]]
    kept = []
    for export, scale, golden in (("export_vlora", 0.5, "golden_llama_lora.json"),
                                  ("export_dv_lora", 1.0, "golden_llama_dv_lora.json")):
        tensors = load_file(os.path.join(ddir, export, "vlora.safetensors"))
        kept.append(write_golden_hf(ldir, os.path.join(out, golden), prompts, lora=(tensors, scale)))
        tokens = [c["tokens"] for c in json.load(open(os.path.join(out, golden), encoding="utf-8"))["cases"]]
        if not set(map(tuple, tokens)).isdisjoint(set(map(tuple, base))):
            raise RuntimeError(f"{golden}: the V-LoRA leaves some case's tokens the plain model's")
    return kept[0], kept[1]


# golden_test's DV rounds (tests/e2e/golden_test.cpp, dv_rounds): every third round from the second (rounds 1, 4, 7,
# ... counting the first, DV's star of MASKs, as round 0) verifies a chain instead of DV's tree, the CUDA backend's
# test chain (set_oracle): its first a drafts plain decoding's tokens, the others not (each one past its token), a
# taking DV_CHAINS in turn (a file of DV_CHAINS_BY_FILE its own); so the canvas slides by 1 + a after them, 16 (past
# every slot) included.  golden_test builds the same chains.  A draft whose position is past the case's tokens is
# token 0 (the backend's chain past the oracle).
DV_CHAINS = (1, 3, 6, 2, 7, 4, 15, 5, 0)
DV_CHAIN_EVERY, DV_CHAIN_FIRST = 3, 1
# Trees of 31 drafts on 32 slots: a chain of all 31 accepted early enough in a case of 48 tokens to be taken whole, so
# that the canvas slides past every slot (32) and the round after it drafts from a canvas with none; chains of none
# and few accepted around it, so that a case still has ten trees of DV's own (40 in all, golden_test's floor 36).
DV_CHAINS_31 = (0, 0, 31, 0, 1, 0, 2)
# The same on 16 slots: a chain of 15 accepted slides the canvas by 16, every slot.
DV_CHAINS_31_WIDE = (0, 0, 15, 0, 1, 0, 2)
DV_CHAINS_BY_FILE = {"tiny_llama_dv31.flyx": DV_CHAINS_31, "tiny_llama_dv31_pool.flyx": DV_CHAINS_31,
                     "tiny_llama_dv31_lora.flyx": DV_CHAINS_31, "tiny_llama_dv31_wide.flyx": DV_CHAINS_31_WIDE,
                     "tiny_llama_dv31_wide_pool.flyx": DV_CHAINS_31_WIDE}


def dv_chains(prompt_len: int, plain: list, width: int, vocab: int, chains=DV_CHAINS):
    """The reference's oracle for those rounds (flyx_ref/dv.py Dv.generate): the chain's (drafts, parents, depths),
    else DV's own tree."""
    calls = [0]

    def oracle(pos, drafts, parents, depths):
        k = calls[0]
        calls[0] += 1
        if k % DV_CHAIN_EVERY != DV_CHAIN_FIRST:
            return drafts
        a = chains[(k // DV_CHAIN_EVERY) % len(chains)]
        chain = []
        for t in range(width):
            at = pos + 1 + t - prompt_len  # plain decoding's token at the draft's position
            chain.append(0 if at >= len(plain) else plain[at] if t < a else (plain[at] + 1) % vocab)
        return chain, list(range(-1, width - 1)), list(range(1, width + 1))
    return oracle


def dv_fresh(model, prompt: list, tokens: list) -> list:
    """DV's first two trees after a fresh start at each point of a case: from position len(prompt) + k,
    k < len(tokens) - 2, the prompt and tokens[:k] prefilled, tokens[k] the anchor, the round of DV's star (one canvas
    pass from MASK), the tree of the round after it (first: from that pass alone) and of the one after that (second:
    from the canvas it slid, its guesses and expected embeddings as inputs).  Each: the position its round starts at,
    its drafts and parents."""
    out = []
    for k in range(len(tokens) - 2):
        rounds = model.generate(prompt + tokens[:k], model.dv.width + 3)["rounds"]
        out.append({w: {"start": r["pos"], "drafts": r["drafts"], "parents": r["parents"]}
                    for w, r in (("first", rounds[1]), ("second", rounds[2]))})
    return out


def write_dv_rounds(out: str, cases: dict) -> int:
    """golden_llama_dv.json: per DV file, flyx_ref/dv.py in float64 on the file as stored (from_flyx), rounding what
    the CUDA backend rounds (dv.round_kv: binary16 K and V; past 16 slots, the canvas's linears' inputs in bf16, its
    prefill GEMMs), so that only the order of float32's sums is left between the two: per case the rounds of DV with
    the chains above (files: each one's position, its anchor's, drafts, parents and drafts accepted; the reference must
    emit the case's tokens, transformers', which the CUDA backend's rounds emit too) and DV's first tree after a fresh
    start at each point (fresh, dv_fresh); and the same fresh starts with FP8 caches (fp8: the e4m3 reference's own
    greedy tokens, which FP8 caches may change, and its trees along them).  cases: a file's golden cases (its
    transformers tokens: tiny_llama_dv_lora.flyx's and tiny_llama_dv31_lora.flyx's with their V-LoRA), the others'
    under "".

    A file's cases are independent of each other and of the other files', and nearly all of this script's time (a
    fresh start at every point of every case, twice over; the script took the hosted CPU runner 41 minutes with them
    one after another), so each case is a process's own job (dv_case_rounds), as many at once as there are processors,
    and their rows are put together in the files' and the cases' order: the bytes one process writes."""
    rows = {"chains": list(DV_CHAINS), "every": DV_CHAIN_EVERY, "first": DV_CHAIN_FIRST,
            "file_chains": {k: list(v) for k, v in DV_CHAINS_BY_FILE.items()}, "files": {}, "fresh": {}, "fp8": {}}
    jobs = [(out, name, c) for name in DV_ROUND_FILES for c in cases.get(name, cases[""])]
    workers = min(len(jobs), os.cpu_count() or 1)
    if workers > 1:
        import multiprocessing
        # spawn everywhere: a fork of a process whose torch has started its threads may wait on them forever
        with multiprocessing.get_context("spawn").Pool(workers, initializer=torch.set_num_threads,
                                                      initargs=(1,)) as pool:
            done = pool.map(dv_case_rounds, jobs, chunksize=1)
    else:
        done = [dv_case_rounds(job) for job in jobs]
    for name in DV_ROUND_FILES:
        rows["files"][name], rows["fresh"][name], rows["fp8"][name] = [], [], []
    n = 0
    for (_, name, _), (runs, fresh, fp8_run, count) in zip(jobs, done):
        rows["files"][name].append(runs)
        rows["fresh"][name].append(fresh)
        rows["fp8"][name].append(fp8_run)
        n += count
    with open(os.path.join(out, "golden_llama_dv.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=0)
    return n


DV_ROUND_FILES = ("tiny_llama_dv.flyx", "tiny_llama_dv_pool.flyx", "tiny_llama_dv32.flyx", "tiny_llama_dv31.flyx",
                  "tiny_llama_dv31_pool.flyx", "tiny_llama_dv31_wide.flyx", "tiny_llama_dv31_wide_pool.flyx",
                  "tiny_llama_dv_lora.flyx", "tiny_llama_dv31_lora.flyx")


def dv_case_rounds(job: tuple) -> tuple:
    """One case of one DV file for golden_llama_dv.json (write_dv_rounds): its rounds, its fresh starts, the same with
    FP8 caches, and how many rounds that is.  A function of the file and the case alone (the reference from the file
    anew, no random numbers), so it gives the same rows in a process of its own, whatever ran there before."""
    out, name, c = job
    cfg, W, dvc = dv.from_flyx(os.path.join(out, name))
    canvas_round = torch.bfloat16 if dvc.slots > 16 else None
    model = dv.Dv(cfg, W, dvc, torch.float64, kv_round=torch.float16, canvas_round=canvas_round)
    fp8 = dv.Dv(cfg, W, dvc, torch.float64, kv_round="e4m3", canvas_round=canvas_round)
    r = model.generate(c["prompt"], len(c["tokens"]),
                       oracle=dv_chains(len(c["prompt"]), c["tokens"], dvc.width, cfg.vocab,
                                        DV_CHAINS_BY_FILE.get(name, DV_CHAINS)))
    if r["ids"] != c["tokens"]:
        raise RuntimeError(f"{name}: DV's float64 reference does not emit the golden tokens")
    runs = [{k: rd[k] for k in ("pos", "drafts", "parents", "accepted")} for rd in r["rounds"]]
    fresh = dv_fresh(model, c["prompt"], c["tokens"])
    tokens8 = fp8.plain(c["prompt"], len(c["tokens"]))
    fp8_run = {"tokens": tokens8, "fresh": dv_fresh(fp8, c["prompt"], tokens8)}
    return runs, fresh, fp8_run, len(r["rounds"]) + 2 * len(fresh) + 2 * len(fp8_run["fresh"])


def write_chat_cases(tok_dir: str, out_json: str) -> int:
    """Every conversation of chat_cases.py through each template fixture, rendered by transformers."""
    from transformers import PreTrainedTokenizerFast
    sys.path.insert(0, os.path.join(ROOT, "tests", "core", "render"))
    import chat_cases
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tok_dir, "tokenizer.json"))
    rows = []
    for name, fmt in chat_cases.TEMPLATES.items():
        src = open(os.path.join(chat_cases.HERE, "templates", name), encoding="utf-8").read()
        for case, msgs, tools in chat_cases.CASES:
            for thinking in (True, False):
                rows.append({"template": name, "format": fmt, "case": case, "messages": msgs, "tools": tools,
                             "thinking": thinking, **chat_cases.expected(tok, src, msgs, tools, thinking)})
    json.dump(rows, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return len(rows)


def splitmix_uniform(seed: int, n: int):
    """u_i = (splitmix64(seed * 2^32 + i) >> 11) * 2^-53 for i < n: doubles in [0, 1) that
    tests/ops/draft/dv_tree_test.cpp makes again bit for bit, so its random tables need not be written out."""
    import numpy as np
    u64 = np.uint64
    z = u64(seed) * u64(1 << 32) + np.arange(n, dtype=np.uint64) + u64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> u64(30))) * u64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> u64(27))) * u64(0x94D049BB133111EB)
    return ((z ^ (z >> u64(31))) >> u64(11)).astype(np.float64) * 2.0 ** -53


def dv_tree_table(seed: int, depths: int, k: int, ties: bool):
    """A tree case's inputs as dv_tree_test.cpp makes them: the table, log-probabilities in [-8, 0] (ties: multiples
    of 1/4), and each slot's candidates' log-probabilities, multiples of 1/16 in (-19, -3] (every third slot's below
    -20, so that its top-k mass falls under cov's floor of 1e-6)."""
    import numpy as np
    n = depths * k * k
    u = splitmix_uniform(seed, n + depths * k)
    t = -np.floor(u[:n] * 8) / 4 if ties else -8 * u[:n]
    logp = (-np.floor(u[n:] * 256) / 16 - 3).reshape(depths, k)
    logp[2::3] -= 17
    return torch.from_numpy(t.reshape(depths, k, k).copy()), torch.from_numpy(logp.copy())


def write_dv_cases(out_json):
    """DV's pieces from the Python reference (flyx_ref/dv.py, float64), for the float64 references in ops/: trees
    from 1000 random tables of each kind (beam, pool) plus edge cases, selector tables, canvas inputs, statistics and
    slides.  The trees' tables and log-probabilities are not written out: dv_tree_test.cpp makes them again from each
    case's seed.  cov's term is, as the reference computes it from the log-probabilities."""
    gen = torch.Generator().manual_seed(77)
    ri = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=gen))  # noqa: E731
    mask = -7

    def tree_case(kind, seed, width, depths, k, ties, live_depths, cov=False, beam=0, ranks=0):
        table, logp = dv_tree_table(seed, depths, k, ties)
        cand = torch.arange(depths * k).reshape(depths, k) + 1000
        live = torch.arange(depths) < live_depths
        case = {"tree": kind, "seed": seed, "width": width, "depths": depths, "k": k, "ties": ties,
                "live_depths": live_depths, "cov": cov, "beam": beam, "ranks": ranks, "mask": mask}
        if kind == "beam":
            drafts, parents, _ = dv.beam_tree(dv.beam_edges(table, logp, live, cov), cand, width, beam, mask)
            if cov:   # cov's term a slot, as beam_edges adds it to every edge into the slot (0 + term: exact)
                zero, every = torch.zeros(depths, 1, 1, dtype=logp.dtype), torch.ones(depths, dtype=torch.bool)
                case["coverage"] = dv.beam_edges(zero, logp, every, True)[:, 0, 0].tolist()
        else:
            drafts, parents, _ = dv.pool_tree(table, cand, live, width, ranks, mask)
        assert all(p < i for i, p in enumerate(parents.tolist()))
        return dict(case, drafts=drafts.tolist(), parents=parents.tolist())

    trees = []
    for c in range(1000):
        k = 16 if c % 4 else ri(1, 16)
        depths, beam = ri(1, 31), ri(1, 20)
        width = ri(1, min(31, depths * beam))
        live = depths if c % 3 else ri(0, depths)
        trees.append(tree_case("beam", 1000 + c, width, depths, k, c % 5 == 0, live, cov=c % 2 == 0, beam=beam))
    for c in range(1000):
        k = 16 if c % 4 else ri(2, 16)
        depths, ranks = ri(1, 31), ri(1, min(7, k - 1))
        width = ri(1, min(31, sum(1 + d * ranks for d in range(1, depths + 1))))
        live = depths if c % 3 else ri(0, depths)
        trees.append(tree_case("pool", 5000 + c, width, depths, k, c % 5 == 0, live, ranks=ranks))
    # DYN 31:8:cov on 32 slots with 4 past the canvas; beams wider than a depth's children (padded: MASK drafts);
    # every slot past the canvas; ties everywhere; ST 31/7/1 with slots past the canvas, in ties, nearly all dead
    edges = [tree_case("beam", 1, 31, 31, 16, False, 27, cov=True, beam=8),
             tree_case("beam", 2, 8, 1, 4, False, 1, beam=8),
             tree_case("beam", 3, 20, 3, 4, True, 3, beam=12),
             tree_case("beam", 4, 31, 31, 16, False, 0, cov=True, beam=8),
             tree_case("beam", 5, 31, 31, 16, True, 31, beam=8),
             tree_case("pool", 6, 31, 31, 16, False, 25, ranks=7),
             tree_case("pool", 7, 31, 31, 16, True, 31, ranks=7),
             tree_case("pool", 8, 15, 2, 8, False, 1, ranks=7),
             tree_case("pool", 9, 31, 31, 16, False, 0, ranks=7)]
    for e in edges:
        e["edge"] = True

    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    tables = []
    for c in range(20):
        V, D, rank, S, k = 40, 24, 8, 6, 16
        E, A, H, B, h = r(V, D), r(rank, D) / 5, r(rank, D) / 5, r(rank, D) / 5, r(S, D)
        cand = torch.stack([torch.randperm(V, generator=gen)[:k] for _ in range(S)])
        logp = (torch.rand(S, k, generator=gen, dtype=torch.float64) * 1.2).clamp(max=1.0).log()
        alpha, temp, anchor = 0.85 + 0.1 * c, 1.0 if c % 2 else 0.7, ri(0, V - 1)
        sel = dv.Selector(E, A, H, B, torch.tensor(alpha, dtype=torch.float64), temp)
        table = dv.selector_table(sel, anchor, cand, logp, h)
        tables.append({"vocab": V, "hidden": D, "rank": rank, "slots": S, "k": k, "embed": E.flatten().tolist(),
                       "A": A.flatten().tolist(), "H": H.flatten().tolist(), "B": B.flatten().tolist(),
                       "h": h.flatten().tolist(), "cand": cand.flatten().tolist(), "logp": logp.flatten().tolist(),
                       "alpha": alpha, "temp": temp, "anchor": anchor, "table": table.flatten().tolist()})

    canvases = []
    for c in range(20):
        V, D, N, k, width = 50, 12, 8, 16, 5
        cmask = V - 1
        E, mask_emb, res_in, res_out = r(V, D), r(D), r(2 * D, D) / 3, r(D, D) / 3
        tokens = torch.randint(0, V, (N,), generator=gen)
        tokens[torch.rand(N, generator=gen) < 0.4] = cmask
        z = r(N, D) * (torch.rand(N, 1, generator=gen) < 0.7)
        logits = r(N, V) * 3
        if c % 2:
            logits = (logits * 4).round() / 4          # ties within the top k and at its edge
        hid = r(N, width)
        n, fill = ri(1, N), [0.9, 0.3, 0.05, 1.0][c % 4]
        x = dv.canvas_input(tokens, z, E, mask_emb, res_in, res_out, cmask)
        st = dv.canvas_stats(logits, E, k)
        sl = dv.slide(st, hid, n, fill, cmask)
        canvases.append({"vocab": V, "hidden": D, "slots": N, "k": k, "width": width, "mask": cmask,
                         "embed": E.flatten().tolist(), "mask_emb": mask_emb.tolist(),
                         "res_in": res_in.flatten().tolist(), "res_out": res_out.flatten().tolist(),
                         "tokens": tokens.tolist(), "z": z.flatten().tolist(), "x": x.flatten().tolist(),
                         "logits": logits.flatten().tolist(), "ids": st["ids"].flatten().tolist(),
                         "p": st["p"].flatten().tolist(), "lse": st["lse"].tolist(),
                         "mass": st["p"].sum(-1).tolist(), "zn": st["zn"].flatten().tolist(),
                         "hid": hid.flatten().tolist(), "n": n, "fill": fill,
                         "slid": {"cv": sl["cv"].tolist(), "z": sl["z"].flatten().tolist(),
                                  "cand": sl["cand"].flatten().tolist(), "logp": sl["logp"].flatten().tolist(),
                                  "live": sl["ok"].int().tolist(), "hid": sl["hid"].flatten().tolist()}})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"trees": trees + edges, "tables": tables, "canvases": canvases}, f, separators=(",", ":"))


PENALTY = (1.5, 0.25)  # presence, frequency: the golden cases with penalties


def write_golden(path, out_json, prompts, vocab, min_margin=0.0, keep=None):
    target, draft, mtp, header, _ = load(path, with_mtp=True)
    golden = {"eos": header["generation"]["eos_ids"], "block_drafter": header["draft"]["kind"], "cases": []}
    for pr in prompts:
        if keep is not None and len(golden["cases"]) >= keep:
            break
        plain, margins = greedy_plain(target, pr, 48, stop_ids=())
        if min(margins) < min_margin:
            continue
        spec, rounds = greedy_spec(target, draft, pr, 48, stop_ids=())
        assert plain == spec, (plain, spec)
        # Oracle drafts: the true continuation, corrupted where pos % 7 == 3, so
        # rounds accept a mix of 0..15 drafts and exercise commit and rollback.
        arr = list(pr) + plain
        arr = [t if (i < len(pr) or i % 7 != 3) else (t + 1) % vocab for i, t in enumerate(arr)]
        oracle = lambda pos, d, arr=arr: arr[pos] if pos < len(arr) else 0  # noqa: E731
        ospec, orounds = greedy_spec(target, draft, pr, 48, stop_ids=(), oracle=oracle)
        assert ospec == plain
        mspec, mrounds = greedy_spec_mtp(target, mtp, pr, 48, width=3)
        assert mspec == plain
        mo, morounds = greedy_spec_mtp(target, mtp, pr, 48, width=3, oracle=oracle)
        assert mo == plain
        # The same prompt with penalties on the tokens the reply already has
        # (ops/sampling/sampling.h).  Null when a choice is too close for
        # float32 against float64: the engine is then only held to itself.
        pen, pen_margins = greedy_plain(target, pr, 48, presence=PENALTY[0], frequency=PENALTY[1])
        penalty = {"presence": PENALTY[0], "frequency": PENALTY[1], "tokens": pen, "min_margin": min(pen_margins),
                   "differs": pen != plain} if min(pen_margins) >= max(min_margin, 0.01) else None
        golden["cases"].append({"prompt": pr, "tokens": plain, "min_margin": min(margins), "penalty": penalty,
                                "rounds": [{k: rd[k] for k in ("pos", "anchor", "drafts", "n")} for rd in rounds],
                                "oracle": arr, "oracle_n": [rd["n"] for rd in orounds],
                                "mtp_rounds": [{k: rd[k] for k in ("pos", "anchor", "drafts", "n")} for rd in mrounds],
                                "mtp_oracle_n": [rd["n"] for rd in morounds]})
    assert golden["cases"], "no prompt with a clear enough margin"
    assert any(c["penalty"] and c["penalty"]["differs"] for c in golden["cases"]), "the penalties change no case"
    json.dump(golden, open(out_json, "w", encoding="utf-8"), indent=0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny_assets")
