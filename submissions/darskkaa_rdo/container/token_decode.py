"""The minimal decoder extension.

Decodes a candidate archive using ONLY data contained in the
archive itself -- the per-frame warp signal is read from the archive's
trailer (container.payload_format.split_inner_payload +
container.warp_signal.decode_warp_signal), never from an externally supplied
file. The "core" (compressed_models + compact_residual + token_stream) is
handed UNCHANGED to the inherited, unmodified `read_residual_archive()` --
no inherited container-parsing code is touched.

The only decoder-loop change from the inherited `decode_production_tokens`
(residual_archive.py:603-713) is: before calling
`model.prepare_frame_context`, apply `container.context_warp.apply_warp` to
the `previous` tensor using that frame's warp ID from the signal. The
corrector's own boundary/run/agreement state (`_boundary_buckets`, `end_frame`)
continues to receive the literal, UNWARPED previous frame -- exactly as
the original search and validation did.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


from container.context_warp import apply_warp  # noqa: E402
from container.payload_format import split_inner_payload, read_archive, write_archive  # noqa: E402
from container.warp_signal import decode_warp_signal  # noqa: E402
from container.archive_writer import EXTERNAL_ATTR_PR141  # noqa: E402


POST_PURGE_HOOKS = []

def decode_tokens(archive_path: Path, pr140_runtime_dir: Path, device_name: str = "cuda"):
    """Decode a candidate archive end-to-end, using only archive data.

    `pr140_runtime_dir` points at a directory containing the inherited,
    UNMODIFIED `runtime/` package (e.g. external/pr140) -- used only for its
    container-parsing (`read_residual_archive`) and model-loading code, which
    this function does not alter.

    Returns (decoded_tokens: torch.Tensor[N,H,W] uint8, info: dict).
    """
    import torch

    sys.path.insert(0, str(pr140_runtime_dir))
    # Force a fresh import bound to THIS runtime dir (PR140 vs PR141 trees
    # both define a top-level `runtime` package; avoid stale sys.modules
    # from a previous call in the same process).
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    for _hook in POST_PURGE_HOOKS:  # optional decoder shims (empty by default => behaviour unchanged)
        _hook()

    from runtime.residual_archive import (  # noqa: E402
        read_residual_archive,
        _boundary_buckets,
        _sparse_class,
        _probability_table,
        NUM_CLASSES,
    )
    from runtime.ihs2 import materialize_ihs1  # noqa: E402
    from runtime.residual_archive import _rr8_select_corrector  # noqa: E402
    from runtime.f26_inflate import _load_renderer, _configure_device  # noqa: E402
    from runtime.hpac_inference import optimize_sparse_evaluator  # noqa: E402
    from runtime.entropy.rc64 import NativeDecoder  # noqa: E402

    p_bytes = read_archive(archive_path)
    core, warp_signal = split_inner_payload(p_bytes)

    renderer_dir = pr140_runtime_dir / "cpr1"
    renderer = _load_renderer(renderer_dir)
    device, device_report = _configure_device(torch, device_name, 4)

    warp_ids = decode_warp_signal(warp_signal, count=renderer.N)
    assert len(warp_ids) == renderer.N, f"decoded {len(warp_ids)} warp ids, expected {renderer.N}"
    # Optional HPAC t-2 temporal branch (EXT record 0x30), see container/hpac2.py.
    from decoder import extfmt
    _, ext_payload = extfmt.split_trailer(warp_signal, renderer.N)
    prev2_record = extfmt.parse_payload(ext_payload)["prev2"] if ext_payload else None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_archive = Path(tmp) / "core_only.zip"
        write_archive(tmp_archive, core, EXTERNAL_ATTR_PR141)
        parts = read_residual_archive(tmp_archive)  # inherited, UNMODIFIED parsing logic

    base_hpac = materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    if prev2_record is not None:
        import importlib
        from container import hpac2 as hpac2_module
        hpac_integer = importlib.import_module("hpac_integer")
        kwargs = dict(
            num_pairs=renderer.N, num_classes=renderer.NUM_CLASSES, patch=renderer.HPAC_PATCH,
            delta=renderer.HPAC_DELTA, channels=renderer.HPAC_CHANNELS, frame_dim=renderer.HPAC_FILM_DIM,
            norm_mode="none", activation="relu", use_frame_scale=True, weight_bound=127, activation_bound=127,
            use_weight_scales=True, weight_exponent_min=-6, use_spm=True, use_norm_gates=False,
        )
        if prev2_record[0] == hpac2_module.EXT_TAG_SHARE:
            hpac2 = hpac2_module.make_hpacs_class(hpac_integer)(**kwargs)
            hpac2.load_state_dict(model.state_dict(), strict=False)
            hpac2_module.unpack_record_share(prev2_record, hpac2)
        else:
            hpac2 = hpac2_module.make_hpac2_class(hpac_integer)(k2=prev2_record[1], **kwargs)
            hpac2.load_state_dict(model.state_dict(), strict=False)
            hpac2_module.unpack_record(prev2_record, hpac2.conv_past2)
        model = hpac2.eval().to(device)
    masks = renderer.group_masks(device)
    sparse = _sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    optimize_sparse_evaluator(sparse)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append((torch.from_numpy(flat_positions).to(device), flat_positions))

    corrector = _rr8_select_corrector(renderer.EVAL_H * renderer.EVAL_W)
    decoder = NativeDecoder(Path(os.environ["CPR1_RC64_LIBRARY"]), parts.token_stream)
    previous = torch.zeros((1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long, device=device)
    previous2 = previous
    decoded_tokens = torch.empty((renderer.N, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.uint8)

    t_start = time.time()
    with torch.inference_mode():
        for frame in range(renderer.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            warp_id = warp_ids[frame]

            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8)

            # THE ONLY DECODER-LOOP CHANGE from the inherited decode_production_tokens:
            # warp the tensor fed into prepare_frame_context. `boundary` above and
            # `corrector.end_frame` below continue to use the literal, unwarped
            # `previous` -- unchanged from the inherited loop.
            warped_previous = apply_warp(previous, warp_id)
            if prev2_record is not None:
                context = model.prepare_frame_context(index, warped_previous, apply_warp(previous2, warp_id))
            else:
                context = model.prepare_frame_context(index, warped_previous)

            corrector.begin_frame(boundary)
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                corrected = base_logits + parts.table.values[feature]
                probability = _probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                state = corrector.group_state(probability, predicted, flat_positions)
                row64 = corrector.coding_row(state)
                symbols = decoder.decode(row64).astype(np.int64)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
            decoded_tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            corrector.end_frame(decoded_tokens[frame].numpy().reshape(-1))
            previous2 = previous
            previous = current

    elapsed = time.time() - t_start
    del model
    info = {
        "decoded_token_sha256": hashlib.sha256(decoded_tokens.numpy().tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "warp_ids_decoded": len(warp_ids),
        "warp_choice_counts": {w: warp_ids.count(w) for w in sorted(set(warp_ids))},
        "parts_semantic_blob_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
        "parts_carrier_blob_sha256": hashlib.sha256(parts.carrier_blob).hexdigest(),
        "parts_hpac_blob_sha256": hashlib.sha256(parts.hpac_blob).hexdigest(),
        "parts_residual_payload_sha256": hashlib.sha256(parts.residual_payload).hexdigest(),
        "parts_token_stream_bytes": len(parts.token_stream),
        "parts_token_stream_sha256": hashlib.sha256(parts.token_stream).hexdigest(),
    }
    return decoded_tokens, info
