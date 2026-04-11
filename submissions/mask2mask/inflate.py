#!/usr/bin/env python
import io
import marshal
import os
import sys
import tempfile
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------
# FP4 Dequantization Tools
# -----------------------------
class FP4Codebook:
	pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

	@staticmethod
	def dequantize_from_nibbles(nibbles: torch.Tensor, scales: torch.Tensor, orig_shape):
		flat_n = int(torch.tensor(orig_shape).prod().item())
		block_size = nibbles.numel() // scales.numel()

		nibbles = nibbles.view(-1, block_size)
		signs = (nibbles >> 3).to(torch.int64)
		mag_idx = (nibbles & 0x7).to(torch.int64)

		levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
		q = levels[mag_idx]
		q = torch.where(signs.bool(), -q, q)
		dq = q * scales[:, None].to(torch.float32)
		return dq.view(-1)[:flat_n].reshape(orig_shape)

def unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
	flat = packed.reshape(-1)
	hi = (flat >> 4) & 0x0F
	lo = flat & 0x0F
	out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
	out[0::2] = hi
	out[1::2] = lo
	return out[:count]

def get_decoded_state_dict(payload_data, device: torch.device):
	data = torch.load(io.BytesIO(payload_data), map_location=device)
	state_dict = {}

	for name, rec in data["quantized"].items():
		if rec["weight_kind"] == "fp4_packed":
			padded_count = rec["packed_weight"].numel() * 2
			nibbles = unpack_nibbles(rec["packed_weight"].to(device), padded_count)
			w = FP4Codebook.dequantize_from_nibbles(
				nibbles, rec["scales_fp16"].to(device), rec["weight_shape"]
			)
		else:
			w = rec["weight_fp16"].to(device).float()

		state_dict[f"{name}.weight"] = w.float()
		if rec.get("bias_fp16") is not None:
			state_dict[f"{name}.bias"] = rec["bias_fp16"].to(device).float()

	for name, tensor in data["dense_fp16"].items():
		state_dict[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)

	return state_dict

# -----------------------------
# Inference Helpers & Main
# -----------------------------
def load_encoded_mask_video(path: str) -> torch.Tensor:
	container = av.open(path)
	frames = []
	for frame in container.decode(video=0):
		img = frame.to_ndarray(format="gray")
		cls_img = np.round(img / 63.0).astype(np.uint8)
		cls_img = np.clip(cls_img, 0, 4)
		frames.append(cls_img)
	container.close()
	return torch.from_numpy(np.stack(frames)).contiguous()

def main():
	if len(sys.argv) < 4:
		print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
		sys.exit(1)

	data_dir = Path(sys.argv[1])
	out_dir = Path(sys.argv[2])
	file_list_path = Path(sys.argv[3])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]

	model_br = data_dir / "model.pt.br"
	mask_br = data_dir / "mask.mp4.br"
	arch_br = data_dir / "arch.br"

	# open/load minimally obfuscated architecture
	with open(arch_br, "rb") as f:
		compressed_bytecode = f.read()
	
	code_obj = marshal.loads(brotli.decompress(compressed_bytecode))
	exec(code_obj, globals())
	
	generator = AsymmetricPairGenerator().to(device)

	# 2. Load Weights
	with open(model_br, "rb") as f:
		weights_data = brotli.decompress(f.read())
	
	generator.load_state_dict(get_decoded_state_dict(weights_data, device), strict=True)
	generator.eval()

	with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
		with open(mask_br, "rb") as f:
			tmp_mp4.write(brotli.decompress(f.read()))
		tmp_mp4_path = tmp_mp4.name

	mask_frames_all = load_encoded_mask_video(tmp_mp4_path)
	os.remove(tmp_mp4_path)

	out_h, out_w = 874, 1164
	cursor = 0

	with torch.inference_mode():
		for file_name in files:
			base_name = os.path.splitext(file_name)[0]
			raw_out_path = out_dir / f"{base_name}.raw"
			
			video_masks = mask_frames_all[cursor : cursor + 1200]
			cursor += 1200
			
			usable_len = (video_masks.shape[0] // 2) * 2
			pairs = video_masks[:usable_len].view(-1, 2, video_masks.shape[-2], video_masks.shape[-1])
			
			with open(raw_out_path, "wb") as f_out:
				batch_size = 4 
				pbar = tqdm(range(0, pairs.shape[0], batch_size), desc=f"Decoding {file_name}")
				
				for i in pbar:
					batch_pairs = pairs[i : i + batch_size].to(device)
					
					in_mask1 = batch_pairs[:, 0].long()
					in_mask2 = batch_pairs[:, 1].long()

					fake1, fake2 = generator(in_mask1, in_mask2)

					fake1_up = F.interpolate(fake1, size=(out_h, out_w), mode="bilinear", align_corners=False)
					fake2_up = F.interpolate(fake2, size=(out_h, out_w), mode="bilinear", align_corners=False)

					batch_comp = torch.stack([fake1_up, fake2_up], dim=1)
					batch_comp = einops.rearrange(batch_comp, "b t c h w -> (b t) h w c")

					output_bytes = batch_comp.clamp(0, 255).round().to(torch.uint8)
					f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
	main()