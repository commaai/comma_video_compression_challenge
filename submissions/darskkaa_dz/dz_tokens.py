import importlib
import os
import struct
import sys
from pathlib import Path
import numpy as np
import dz_mc
from dz_bitcoder import Decoder, Models, put_int
HERE = Path(__file__).resolve().parent

def build_model(renderer, canonical_ihs1, prev2_record, device):
    from container import hpac2 as hpac2_module
    model = renderer.load_hpac(canonical_ihs1, device)
    hpac_integer = importlib.import_module('hpac_integer')
    kwargs = dict(num_pairs=renderer.N, num_classes=renderer.NUM_CLASSES, patch=renderer.HPAC_PATCH, delta=renderer.HPAC_DELTA, channels=renderer.HPAC_CHANNELS, frame_dim=renderer.HPAC_FILM_DIM, norm_mode='none', activation='relu', use_frame_scale=True, weight_bound=127, activation_bound=127, use_weight_scales=True, weight_exponent_min=-6, use_spm=True, use_norm_gates=False)
    if prev2_record[0] == hpac2_module.EXT_TAG_SHARE:
        full = hpac2_module.make_hpacs_class(hpac_integer)(**kwargs)
        full.load_state_dict(model.state_dict(), strict=False)
        hpac2_module.unpack_record_share(prev2_record, full)
    else:
        full = hpac2_module.make_hpac2_class(hpac_integer)(k2=prev2_record[1], **kwargs)
        full.load_state_dict(model.state_dict(), strict=False)
        hpac2_module.unpack_record(prev2_record, full.conv_past2)
    return full.eval().to(device)

def attach_mc(model, codes, device):
    import torch
    hpac_integer = importlib.import_module('hpac_integer')
    conv = hpac_integer.IntegerConv2d(model.num_classes, model.ch, 3, padding=1, weight_bound=127, use_weight_scales=True, exponent_min=-6)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        if codes is not None:
            weight, bias, exponent = codes
            conv.weight.copy_(torch.from_numpy(np.asarray(weight, dtype=np.float32)))
            conv.bias.copy_(torch.from_numpy(np.asarray(bias, dtype=np.float32)))
            conv.exponent.copy_(torch.from_numpy(np.asarray(exponent, dtype=np.float32)))
    model.conv_mc = conv.to(device)
    return model

def frame_context(model, index, previous_warped, previous2_warped, composite):
    from torch.nn import functional as F
    hpac_integer = importlib.import_module('hpac_integer')
    shift, past, scale, spm = model.prepare_frame_context(index, previous_warped, previous2_warped)
    batch, height, width = composite.shape
    one_hot = F.one_hot(composite, num_classes=model.num_classes).permute(0, 3, 1, 2).float()
    full = model._from_patches(past, batch, height // model.P, width // model.P) + model.conv_mc(one_hot)
    past = model._to_patches(hpac_integer.requantize(full, 0, -model.activation_bound, model.activation_bound))
    return (shift, past, scale, spm)
PARAM_NAMES = ('tau', 'v0', 'u0', 'dx', 'dy')
RHO_NUM = (0, 1, 2, 3, 4)

def _mc_pass(coder, models, weight, bias, exponent, params, rho, encode):
    C = weight.shape[0]
    for c in range(C):
        ctx = ('e',)
        if encode:
            put_int(coder, models, ctx, int(exponent[c]), True)
            put_int(coder, models, ('b',), int(bias[c]), True)
        else:
            exponent[c] = put_int(coder, models, ctx, 0, False)
            bias[c] = put_int(coder, models, ('b',), 0, False)
        for i in range(weight.shape[1]):
            for k in range(9):
                wctx = ('w', i, min(abs(int(exponent[c])), 6))
                if encode:
                    put_int(coder, models, wctx, int(weight[c, i, k // 3, k % 3]), True)
                else:
                    weight[c, i, k // 3, k % 3] = put_int(coder, models, wctx, 0, False)
    N, K = params.shape
    for t in range(1, N):
        for k in range(K):
            pred = rho[k] * int(params[t - 1, k]) >> 2 if t > 1 else 0
            if encode:
                put_int(coder, models, ('p', k), int(params[t, k]) - pred, True)
            else:
                params[t, k] = pred + put_int(coder, models, ('p', k), 0, False)

def decode_mc(blob):
    C, N, rho = struct.unpack_from('<BHB', blob)
    K = len(PARAM_NAMES)
    base = np.frombuffer(blob[4:4 + 2 * K], dtype='<i2').astype(np.int64)
    weight = np.zeros((C, 5, 3, 3), np.int64)
    bias = np.zeros(C, np.int64)
    exponent = np.zeros(C, np.int64)
    params = np.zeros((N, K), np.int64)
    _mc_pass(Decoder(blob[4 + 2 * K:]), Models(), weight, bias, exponent, params, [rho] * K, False)
    params[1:] += base
    return (weight, bias, exponent, params)

def decode_tokens(archive_path, runtime_dir, mc_blob, device_name='cuda'):
    import hashlib
    import tempfile
    import time
    import torch
    from container import token_decode as vendor
    from container.context_warp import apply_warp
    from container.payload_format import split_inner_payload, read_archive, write_archive
    from container.warp_signal import decode_warp_signal
    from container.archive_writer import EXTERNAL_ATTR_PR141
    sys.path.insert(0, str(runtime_dir))
    for name in list(sys.modules):
        if name == 'runtime' or name.startswith('runtime.'):
            del sys.modules[name]
    for hook in vendor.POST_PURGE_HOOKS:
        hook()
    from runtime.residual_archive import read_residual_archive, _boundary_buckets, _sparse_class, _probability_table, NUM_CLASSES, _rr8_select_corrector
    from runtime.ihs2 import materialize_ihs1
    from runtime.f26_inflate import _load_renderer, _configure_device
    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.entropy.rc64 import NativeDecoder
    from decoder import extfmt
    core, warp_signal = split_inner_payload(read_archive(archive_path))
    renderer_dir = Path(runtime_dir) / 'cpr1'
    renderer = _load_renderer(renderer_dir)
    device, _ = _configure_device(torch, device_name, 4)
    warp_ids = decode_warp_signal(warp_signal, count=renderer.N)
    _, ext_payload = extfmt.split_trailer(warp_signal, renderer.N)
    prev2_record = extfmt.parse_payload(ext_payload)['prev2']
    with tempfile.TemporaryDirectory() as tmp:
        shell = Path(tmp) / 'core_only.zip'
        write_archive(shell, core, EXTERNAL_ATTR_PR141)
        parts = read_residual_archive(shell)
    weight, bias, exponent, params = decode_mc(mc_blob)
    model = build_model(renderer, materialize_ihs1(parts.hpac_blob, renderer), prev2_record, device)
    attach_mc(model, (weight, bias, exponent), device)
    sparse = _sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    optimize_sparse_evaluator(sparse)
    plans = []
    for mask in renderer.group_masks(device):
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        plans.append((torch.from_numpy(flat).to(device), flat))
    pixels = renderer.EVAL_H * renderer.EVAL_W
    corrector = _rr8_select_corrector(pixels)
    decoder = NativeDecoder(Path(os.environ['CPR1_RC64_LIBRARY']), parts.token_stream)
    previous = torch.zeros((1, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.long, device=device)
    previous2 = previous
    tokens = torch.empty((renderer.N, renderer.EVAL_H, renderer.EVAL_W), dtype=torch.uint8)
    start = time.time()
    with torch.inference_mode():
        for frame in range(renderer.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            warp_id = warp_ids[frame]
            if frame:
                boundary = _boundary_buckets(previous[0].to(device='cpu', dtype=torch.uint8).numpy()).reshape(-1)
                composite = dz_mc.composite(previous[0], *params[frame].tolist())[None]
            else:
                boundary = np.full(pixels, 4, dtype=np.uint8)
                composite = previous
            context = frame_context(model, index, apply_warp(previous, warp_id), apply_warp(previous2, warp_id), composite)
            corrector.begin_frame(boundary)
            for group, (device_positions, flat) in enumerate(plans):
                base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                corrected = base_logits + parts.table.values[boundary[flat].astype(np.int64) * NUM_CLASSES + predicted]
                state = corrector.group_state(_probability_table(corrected, renderer.HPAC_LOGIT_PRECISION), predicted, flat)
                symbols = decoder.decode(corrector.coding_row(state)).astype(np.int64)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
            tokens[frame] = current[0].to(device='cpu', dtype=torch.uint8)
            corrector.end_frame(tokens[frame].numpy().reshape(-1))
            previous2 = previous
            previous = current
    return (tokens, dict(decoded_token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), decode_runtime_seconds=time.time() - start))
