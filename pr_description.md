# [Feature] Add HALOps as an opt-in MXFP8 GEMM backend for Blackwell

## Motivation

[HALOps](https://github.com/PaddlePaddle/HALOps) is the PaddlePaddle team's high-performance kernel library for NVIDIA Blackwell (SM100). This PR integrates its W8A8 MXFP8 dense GEMM (`dense_gemm`) and fused `group_wise_quant` activation quantizer.

vLLM already serves MXFP8 dense linears through FlashInfer, Marlin, Torch and the b12x backends. HALOps is added as another one, selected explicitly with `--linear-backend halops`. Defaults are unchanged: `auto` keeps HALOps last in the CUDA MXFP8 priority list, and the package is imported lazily.

The change also completes the MXFP8 path used in the benchmarks below. MXFP8 experts are served by the existing DeepGEMM MegaMoE kernel through a new online-quantization mode, so a BF16 checkpoint runs without pre-quantized experts; weights are stored with canonical per-32 UE8M0 scales.

The `halops` wheel is published on PaddlePaddle's CUDA 13.2 index rather than PyPI, so it cannot be a wheel dependency (`pip install vllm` would fail for everyone without that index). It is an optional extra:

```bash
uv pip install "vllm[halops]" \
  -f https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/
```

## Modifications

- `vllm/model_executor/kernels/linear/mxfp8/halops.py` (new): `HALOpsMxfp8LinearKernel` with a lazy `halops` loader and install hint, SfAtom weight-scale swizzle at load time, and `group_wise_quant` + `dense_gemm` wrapped in one custom op with a fake impl (CUDA-graph safe). K and N must be 128-aligned; grouped BMM is rejected.
- `vllm/model_executor/kernels/linear/__init__.py`: register the kernel in `_POSSIBLE_MXFP8_KERNELS[CUDA]` (last) and `_LINEAR_BACKEND_KERNEL_MAP["halops"]`.
- `vllm/config/kernel.py`: `"halops"` in the `LinearBackend` literal and its docstring.
- `vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py`: `online_quant` mode; the weight loader quantizes BF16 expert shards to MXFP8 with canonical per-32 UE8M0 scales and writes them into the existing kernel layout.
- `vllm/models/deepseek_v4/nvidia/model.py`: resolve `MEGA_MOE_MXFP8` with `online_quant=True` when the config requests dynamic MXFP8 experts; the `expert_dtype` gate applies only to the non-online path.
- `setup.py`, `requirements/halops.txt`, `docs/features/quantization/halops.md`: optional extra, pinned standalone requirement, and a backend page with supported configurations.
- `tests/model_executor/kernels/test_halops_linear.py` (CPU): registration and selection, `is_supported()` rejection paths, `can_implement()` limits, weight preparation, custom-op boundary, and the `apply_weights` call shape.

Unit tests:

```bash
python -m pytest tests/model_executor/kernels/test_halops_linear.py -v   # 18 passed
```

## Usage

Serving DeepSeek-V4-Flash BF16 with online MXFP8 quantization, HALOps dense GEMM and MXFP8 MegaMoE on 4×GB200:

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization mxfp8 \
  --quantization-config '{"ignore": ["*.wo_a"]}' \
  --linear-backend halops \
  --moe-backend deep_gemm_mega_moe \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 30082
```

`*.wo_a` stays BF16: an FP8 `wo_a` selects the model's block-128 `fp8_einsum` path, which does not accept per-32 MXFP8 scales.

## Accuracy Tests

DeepSeek-V4-Flash, full GSM8K test set (1319 questions), 5-shot, greedy, `--max-tokens 2048`, TP4 + EP. The two arms differ only in `--linear-backend`; MoE is MXFP8 MegaMoE in both.

| arm | `--linear-backend` | GSM8K (3 runs) |
|---|---|---:|
| HALOps | `halops` | 0.884 / 0.889 / 0.8787 |
| DeepGEMM | `deep_gemm` | 0.873 / 0.8757 / 0.8802 |

The DeepGEMM dense MXFP8 kernel used as the baseline was written for this comparison and is not part of this PR.

## Speed Tests and Profiling

4×GB200 (SM100), DeepSeek-V4-Flash, TP4 + EP, online MXFP8 quantization. Attention dense MXFP8 GEMM is the only variable: MoE is `deep_gemm_mega_moe` in both arms and they differ only in `--linear-backend`.

Dataset level (`vllm bench serve`, random 1024 in / 256 out, `--ignore-eos`, prefix caching disabled, median of 3 runs):

| concurrency | HALOps out tok/s | DeepGEMM out tok/s | ratio | mean TTFT, HALOps / DeepGEMM (ms) | mean TPOT, HALOps / DeepGEMM (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 113.4 | 117.7 | 0.964 | 169.6 / 178.4 | 8.18 / 7.83 |
| 8 | 749.9 | 766.5 | 0.978 | 312.8 / 335.7 | 9.47 / 9.17 |
| 64 | 3807.5 | 3789.7 | 1.005 | 539.4 / 572.4 | 14.75 / 14.70 |

4k in / 1k out, 2 paired repeats:

| concurrency | HALOps out tok/s | DeepGEMM out tok/s | ratio |
|---:|---:|---:|---:|
| 1 | 117.35 / 117.62 | 121.98 / 122.14 | 0.962–0.963 |
| 8 | 770.12 / 773.57 | 786.99 / 787.00 | 0.979–0.983 |
| 32 | 2267.42 / 2284.26 | 2270.51 / 2268.92 | 0.999–1.007 |
| 64 | 3612.99 / 3686.30 | 3615.29 / 3614.85 | 0.999–1.020 |


## Checklist

- [x] Format your code according to [vLLM's pre-commit hooks](https://docs.vllm.ai/en/latest/contributing/) (`ruff check` / `ruff format --check` pass).
- [x] Add unit tests ([testing guide](https://docs.vllm.ai/en/latest/contributing/)): `tests/model_executor/kernels/test_halops_linear.py`, 18 passed.
- [x] Update documentation: `docs/features/quantization/halops.md`.
- [x] Provide accuracy and speed benchmark results ([benchmarking guide](https://docs.vllm.ai/en/latest/contributing/)).
- [x] Follow the vLLM code style [guidance](https://docs.vllm.ai/en/latest/contributing/).
- [x] AI assistance was used during the creation of this PR.
