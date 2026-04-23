#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import av
import torch
import torch.nn as nn
import torch.nn.functional as F


CAMERA_SIZE = (1164, 874)
SEGMAP_INPUT_SIZE = (512, 384)
CLASS_TARGETS = [0, 255, 64, 192, 128]
LUT_SIGMA = 15.0


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, block_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, block_hidden, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(block_hidden, hidden, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        return self.act2(out + residual)


class SegMap(nn.Module):
    def __init__(
        self,
        hidden: int,
        block_hidden: int,
        num_blocks: int,
        max_frame_index: int,
        affine_max_zoom_delta: float,
        affine_max_aspect_delta: float,
        affine_max_shear: float,
        affine_max_translation: float,
        latent_input_scale: float,
    ):
        super().__init__()
        self.h = SEGMAP_INPUT_SIZE[1]
        self.w = SEGMAP_INPUT_SIZE[0]
        self.hidden = hidden
        self.block_hidden = block_hidden
        self.num_blocks = num_blocks
        self.shared_latent_channels = 3
        self.shared_latent_height = 30
        self.shared_latent_width = 40
        self.latent_canvas_scale = 1.25
        self.max_zoom_delta = affine_max_zoom_delta
        self.max_aspect_delta = affine_max_aspect_delta
        self.max_shear = affine_max_shear
        self.max_translation = affine_max_translation
        self.latent_input_scale = latent_input_scale
        self.shared_latent_base = nn.Parameter(
            torch.empty(
                1,
                self.shared_latent_channels,
                self.shared_latent_height,
                self.shared_latent_width,
            )
        )
        self.frame_affine_embedding = nn.Embedding(max_frame_index, 6)
        self.layer_in = nn.Conv2d(5 + self.shared_latent_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, block_hidden) for _ in range(num_blocks)]
        )
        self.layer_out = nn.Conv2d(hidden, 3, kernel_size=1)

    def _build_affine_latent_channel(
        self, frame_indices: torch.Tensor, output_height: int, output_width: int
    ) -> torch.Tensor:
        batch_size = frame_indices.shape[0]
        canvas_height = math.ceil(output_height * self.latent_canvas_scale)
        canvas_width = math.ceil(output_width * self.latent_canvas_scale)
        shared_latent = F.interpolate(
            self.shared_latent_base,
            size=(canvas_height, canvas_width),
            mode="bicubic",
            align_corners=False,
        ).expand(batch_size, -1, -1, -1)
        affine_delta = self.frame_affine_embedding(frame_indices)
        zoom = 1.0 + self.max_zoom_delta * torch.tanh(affine_delta[:, 0:1])
        aspect = self.max_aspect_delta * torch.tanh(affine_delta[:, 1:2])
        shear_x = self.max_shear * torch.tanh(affine_delta[:, 2:3])
        shear_y = self.max_shear * torch.tanh(affine_delta[:, 3:4])
        trans_x = self.max_translation * torch.tanh(affine_delta[:, 4:5])
        trans_y = self.max_translation * torch.tanh(affine_delta[:, 5:6])
        scale_x = zoom + aspect
        scale_y = zoom - aspect
        theta = torch.cat(
            [scale_x, shear_x, trans_x, shear_y, scale_y, trans_y], dim=1
        ).view(-1, 2, 3)
        grid = F.affine_grid(
            theta,
            size=(batch_size, self.shared_latent_channels, output_height, output_width),
            align_corners=False,
        )
        return F.grid_sample(
            shared_latent,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        affine_latent = self._build_affine_latent_channel(
            frame_indices, x.shape[-2], x.shape[-1]
        )
        feat = self.layer_in(torch.cat([x, affine_latent * self.latent_input_scale], dim=1))
        for block in self.blocks:
            feat = block(feat)
        return torch.sigmoid(self.layer_out(feat)) * 255.0


def decode_tensor_payload(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(torch.float32)
    codec = value["codec"]
    bits = int(value["bits"])
    levels = (1 << bits) - 1
    if codec == "linear_q_per_tensor_v1":
        min_val = value["min"].to(torch.float32).view(1)
        max_val = value["max"].to(torch.float32).view(1)
        shape = tuple(int(v) for v in value["shape"].tolist())
        if value["data"].numel() == 0:
            return torch.full(shape, float(min_val.item()), dtype=torch.float32)
        q = value["data"].to(torch.float32)
        return min_val + q * ((max_val - min_val) / levels)
    if codec == "linear_q_per_affine_column_v1":
        shape = tuple(int(v) for v in value["shape"].tolist())
        mins = value["min"].to(torch.float32)
        maxs = value["max"].to(torch.float32)
        out = torch.empty(shape, dtype=torch.float32)
        for col in range(shape[1]):
            column = value["data"][col]
            if column.numel() == 0:
                out[:, col] = mins[col]
            else:
                q = column.to(torch.float32)
                out[:, col] = mins[col] + q * ((maxs[col] - mins[col]) / levels)
        return out
    raise ValueError(f"unsupported tensor codec: {codec}")


def reconstruct_weight(payload: dict, state: dict, prefix: str) -> torch.Tensor:
    qint = state[f"{prefix}.weight_qint"].to(torch.float32)
    exponents = state[f"{prefix}.weight_exponents"].to(torch.float32)
    if payload.get("weight_tensor_layout") == "HWOI":
        qint = qint.permute(2, 3, 0, 1).contiguous()
    return qint * (2 ** exponents)


def create_gaussian_softmax_lut() -> torch.Tensor:
    x = torch.arange(256, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(CLASS_TARGETS, dtype=torch.float32).unsqueeze(0)
    squared_diff = (x - targets) ** 2
    logits = torch.exp(-squared_diff / (2 * LUT_SIGMA**2))
    return F.softmax(logits, dim=1)


def load_segmap(checkpoint_path: Path, device: torch.device) -> SegMap:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("learned_fullres_residual", False):
        raise ValueError("learned_fullres_residual export is not supported by this submission")
    if payload.get("lowfreq_frame_channel", False):
        raise ValueError("lowfreq_frame_channel export is not supported by this submission")
    state = payload["inference_state_dict"]
    model = SegMap(
        hidden=int(payload["hidden"]),
        block_hidden=int(payload.get("block_hidden") or payload["hidden"]),
        num_blocks=int(payload["num_blocks"]),
        max_frame_index=int(payload["max_frame_index"]),
        affine_max_zoom_delta=float(payload.get("affine_max_zoom_delta", 0.12)),
        affine_max_aspect_delta=float(payload.get("affine_max_aspect_delta", 0.03)),
        affine_max_shear=float(payload.get("affine_max_shear", 0.03)),
        affine_max_translation=float(payload.get("affine_max_translation", 0.08)),
        latent_input_scale=float(payload.get("latent_input_scale", 1.0)),
    ).to(device)
    with torch.no_grad():
        model.shared_latent_base.copy_(decode_tensor_payload(state["shared_latent_base"]).to(device))
        model.frame_affine_embedding.weight.copy_(
            decode_tensor_payload(state["frame_affine_embedding.weight"]).to(device)
        )
        model.layer_in.weight.copy_(reconstruct_weight(payload, state, "layer_in").to(device))
        model.layer_in.bias.copy_(decode_tensor_payload(state["layer_in.bias"]).to(device))
        model.layer_out.weight.copy_(reconstruct_weight(payload, state, "layer_out").to(device))
        model.layer_out.bias.copy_(decode_tensor_payload(state["layer_out.bias"]).to(device))
        for block_idx, block in enumerate(model.blocks):
            prefix = f"blocks.{block_idx}.conv1"
            block.conv1.weight.copy_(reconstruct_weight(payload, state, prefix).to(device))
            block.conv1.bias.copy_(decode_tensor_payload(state[f"{prefix}.bias"]).to(device))
            prefix = f"blocks.{block_idx}.conv2"
            block.conv2.weight.copy_(reconstruct_weight(payload, state, prefix).to(device))
            block.conv2.bias.copy_(decode_tensor_payload(state[f"{prefix}.bias"]).to(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.inference_mode()
def inflate_to_raw(data_dir: Path, video_name: str, dst_path: Path) -> None:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = load_segmap(data_dir / "segmap_inference.pt", device)
    lut = create_gaussian_softmax_lut().to(device)
    src_video = data_dir / video_name
    container = av.open(str(src_video))
    stream = container.streams.video[0]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    target_w, target_h = CAMERA_SIZE
    model_w, model_h = SEGMAP_INPUT_SIZE
    with open(dst_path, "wb") as out_handle:
        for idx, frame in enumerate(container.decode(stream)):
            gray = torch.from_numpy(frame.to_ndarray(format="gray").copy()).to(
                device=device, dtype=torch.float32
            )
            gray = (
                F.interpolate(
                    gray.unsqueeze(0).unsqueeze(0),
                    size=(model_h, model_w),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .round()
                .clamp(0, 255)
                .long()
            )
            probability_map = F.embedding(gray, lut).permute(2, 0, 1).contiguous().unsqueeze(0)
            frame_indices = torch.tensor([2 * idx, 2 * idx + 1], device=device, dtype=torch.long)
            restored_batch = model(probability_map.repeat(2, 1, 1, 1), frame_indices)
            for restored in restored_batch:
                fullres = F.interpolate(
                    restored.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )
                rgb = (
                    fullres.clamp(0, 255)
                    .round()
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                out_handle.write(rgb.tobytes())
    container.close()


if __name__ == "__main__":
    inflate_to_raw(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
