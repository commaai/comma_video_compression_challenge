"""
The frame model, which redraws both frames of a pair from its map + motion values.

1. Turn the map into 7 grids: 5 yes/no class grids and 2 position grids.
2. Shrink to quarter size, refine there with 4 corrections, then grow back to full size.
3. Frame 2 head: draw frame 2 from the grids (what SegNet grades).
4. Frame 1 head: adjust the grids with the motion values, then draw frame 1 (what PoseNet reads the motion from).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MAP_HEIGHT, MAP_WIDTH = 384, 512          # SegNet's map size
CAMERA_HEIGHT, CAMERA_WIDTH = 874, 1164   # comma's frame size


class SeparableConv(nn.Module):
    """A depthwise separable conv: looks at each pixel's 3 x 3 neighbours in 2 steps. About 7x fewer weights than a normal conv."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # each channel looks at its own 3 x 3 neighbours
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )

        # blend the channels at each pixel
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SegCorrection(nn.Module):
    """Refines each pixel from its neighbours by adding a learned correction."""

    def __init__(self, channels):
        super().__init__()
        self.conv_a = SeparableConv(channels, channels)
        self.conv_b = SeparableConv(channels, channels)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x):
        correction = self.norm(self.conv_a(F.silu(x)))
        correction = self.conv_b(F.silu(correction))
        return x + correction


class PoseCorrection(nn.Module):
    """Refines each pixel like SegCorrection, with the pair's motion scaling and shifting each channel."""

    def __init__(self, channels, pose_code_size):
        super().__init__()
        self.conv_a = SeparableConv(channels, channels)
        self.conv_b = SeparableConv(channels, channels)
        self.norm = nn.GroupNorm(4, channels)
        self.pose_to_sliders = nn.Linear(pose_code_size, 2 * channels)

    def forward(self, x, pose_code):
        # one multiplier and one offset per channel, using the pair's motion
        sliders = self.pose_to_sliders(pose_code)[:, :, None, None]
        multiplier, offset = sliders.chunk(2, dim=1)

        correction = self.norm(self.conv_a(F.silu(x)))
        correction = correction * (1 + multiplier) + offset
        correction = self.conv_b(F.silu(correction))
        return x + correction


class FrameModel(nn.Module):
    """Draws both frames of a pair from its map and motion values. 64,358 weights."""

    def __init__(self, channels=32, pose_size=6, pose_code_size=32):
        super().__init__()
        C = channels

        # shrink: full -> half -> quarter size
        self.shrink1 = nn.Conv2d(5 + 2, C, kernel_size=3, stride=2, padding=1)
        self.shrink2 = SeparableConv(C, 2 * C, stride=2)

        # 4 corrections at quarter size
        self.seg_corrections = nn.Sequential(
            SegCorrection(2 * C),
            SegCorrection(2 * C),
            SegCorrection(2 * C),
            SegCorrection(2 * C),
        )

        # grow back: quarter -> half -> full size
        self.look1 = SeparableConv(2 * C, C)
        self.look2 = SeparableConv(2 * C, C)
        self.look3 = SeparableConv(C, C)

        # frame 2 head: one last correction, then turn the grids into RGB
        self.draw_frame2 = nn.Sequential(
            SegCorrection(C),
            nn.Conv2d(C, 3, kernel_size=1),
        )

        # frame 1 head: the motion values adjust the grids twice, then turn them into RGB
        self.pose_to_code = nn.Sequential(
            nn.Linear(pose_size, pose_code_size),
            nn.SiLU(),
            nn.Linear(pose_code_size, pose_code_size),
        )
        self.pose_correction1 = PoseCorrection(C, pose_code_size)
        self.pose_correction2 = PoseCorrection(C, pose_code_size)
        self.draw_frame1 = nn.Conv2d(C, 3, kernel_size=1)

        # row and column grids so the model knows where each pixel is
        rows = torch.linspace(-1, 1, MAP_HEIGHT).view(MAP_HEIGHT, 1).expand(MAP_HEIGHT, MAP_WIDTH)
        cols = torch.linspace(-1, 1, MAP_WIDTH).view(1, MAP_WIDTH).expand(MAP_HEIGHT, MAP_WIDTH)
        self.register_buffer("position", torch.stack([rows, cols])[None], persistent=False)

    def forward(self, seg_map, pose):
        """Draw both frames of each pair."""

        # split the map into 5 yes/no class grids (road, lane, sky, car, hood), then add the 2 position grids
        class_grids = F.one_hot(seg_map.long(), 5).permute(0, 3, 1, 2).float()
        position_grids = self.position.expand(len(seg_map), -1, -1, -1)
        grids = torch.cat([class_grids, position_grids], dim=1)

        # shrink twice, then refine at quarter size
        half_size = F.silu(self.shrink1(grids))
        quarter_size = F.silu(self.shrink2(half_size))
        quarter_size = self.seg_corrections(quarter_size)

        # grow back to full size, gluing the half-size detail back on
        full_size = F.silu(self.look1(quarter_size))
        full_size = F.interpolate(full_size, scale_factor=2, mode="bilinear", align_corners=False)
        full_size = torch.cat([full_size, half_size], dim=1)
        full_size = F.silu(self.look2(full_size))
        full_size = F.interpolate(full_size, scale_factor=2, mode="bilinear", align_corners=False)
        full_size = F.silu(self.look3(full_size))

        # frame 2: one last correction, then turn the grids into RGB (0 to 255)
        frame2 = torch.sigmoid(self.draw_frame2(full_size)) * 255

        # frame 1: the motion values adjust the grids twice, then turn them into RGB (0 to 255)
        pose_code = self.pose_to_code(pose)
        grids_with_motion = self.pose_correction1(full_size, pose_code)
        grids_with_motion = self.pose_correction2(grids_with_motion, pose_code)
        frame1 = torch.sigmoid(self.draw_frame1(grids_with_motion)) * 255

        return frame1, frame2


def to_camera(frame):
    """Stretch a drawn frame from 384x512 up to 874x1164 for comma's grader."""
    camera_frame = F.interpolate(frame, size=(CAMERA_HEIGHT, CAMERA_WIDTH), mode="bilinear", align_corners=False)
    return camera_frame.clamp(0, 255).round()


if __name__ == "__main__":
    model = FrameModel()
    print("Weights:", sum(p.numel() for p in model.parameters()))

    frame1, frame2 = model(torch.randint(0, 5, (2, MAP_HEIGHT, MAP_WIDTH)), torch.randn(2, 6))
    print("frame1:", tuple(frame1.shape), " frame2:", tuple(frame2.shape), " camera:", tuple(to_camera(frame2).shape))
