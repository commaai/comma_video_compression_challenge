# Study toolkit

The artifacts behind [`../WRITEUP.md`](../WRITEUP.md).

## Self-contained (recommended)
- `beat_top_colab.ipynb` — quantization experiments (§3.1) on a CUDA Colab runtime.
- `student_retrain_colab.ipynb` — smaller-decoder distillation (§3.3) on CUDA.

Both clone the repo, fetch models/video, and download the teacher weights themselves —
just set the runtime to GPU and Run All.

## Reference scripts
`beat_top.py` (faithful in-process scorer + quant/distill/polish), `ptq_sweep.py` (§3.1),
`selector_search.py` (§3.2), `palettize.py` (§3.4), `student_train.py` (§3.3).

These were developed against a scratch `work/` directory at the repo root (they
`sys.path.insert(0, "work")` and cache `work/base_decoder.pt`, `work/gt_targets.pt`). To
run them, copy the `.py` files into a `work/` dir at the repo root and invoke from there,
e.g. `python work/ptq_sweep.py mps`. The notebooks above are the dependency-free path.
