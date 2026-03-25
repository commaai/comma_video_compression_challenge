#!/usr/bin/env python
import torch, timm, einops, shutil
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path
from safetensors.torch import load_file
from collections import namedtuple
from frame_utils import rgb_to_yuv6, camera_size, segnet_model_input_size

Head = namedtuple('Head', ['name', 'hidden', 'out'])
HERE = Path(__file__).resolve().parent
def get_viewer():
  from PIL import ImageShow
  if shutil.which("eog"): return ImageShow.EogViewer()
  if shutil.which("xdg-open"): return ImageShow.XDGViewer()
  return None
segnet_sd_path = HERE / 'models/segnet.safetensors'
posenet_sd_path = HERE / 'models/posenet.safetensors'

BN_EPS = 0.001
BN_MOM = 0.01
VISION_FEATURES = 2048
SUMMARY_FEATURES = 512
IN_CHANS = 6 * 2
ACT_LAYER = 'gelu_tanh'
HEADS = [Head('pose', 32, 12)]

class AllNorm(nn.Module):
  def __init__(self, num_features: int, eps: float = BN_EPS, momentum: float = BN_MOM, affine: bool = True):
    super().__init__()
    self.bn = nn.BatchNorm1d(1, eps, momentum, affine)
  def forward(self, x):
    return self.bn(x.view(-1, 1)).view(x.shape)

class ResBlock(nn.Module):
  def __init__(self, feats, expansion=2, norm=AllNorm):
    super().__init__()
    self.block_a = nn.Sequential(nn.Linear(feats, feats*expansion), norm(feats*expansion), nn.ReLU(inplace=True), nn.Linear(feats*expansion, feats), norm(feats))
    self.block_b = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(feats, feats*expansion), norm(feats*expansion), nn.ReLU(inplace=True), nn.Linear(feats*expansion, feats), norm(feats))
    self.final_relu = nn.ReLU(inplace=False)
  def forward(self, x):
    a_out = x + self.block_a(x)
    return self.final_relu(a_out + self.block_b(a_out))

class Hydra(nn.Module):
  def __init__(self, num_features: int, heads: list[Head]=HEADS):
    super().__init__()
    self.resblock = ResBlock(num_features)
    self.relu = nn.ReLU(inplace=True)
    self.heads = heads
    self.in_layer = nn.ModuleDict({k.name: nn.Linear(num_features, k.hidden) for k in heads})
    self.res_layer = nn.ModuleDict({h.name: nn.Sequential(nn.Linear(h.hidden, h.hidden), nn.ReLU(inplace=True), nn.Linear(h.hidden, h.hidden)) for h in heads})
    self.final_layer = nn.ModuleDict({h.name: nn.Linear(h.hidden, h.out) for h in heads})
  def forward(self, x):
    x = self.resblock(x)
    in_layer = {k: self.relu(v(x)) for k,v in self.in_layer.items()}
    res_layer = {k: self.relu(in_layer[k] + v(in_layer[k])) for k,v in self.res_layer.items()}
    ret = {k: v(res_layer[k]) for k,v in self.final_layer.items()}
    return ret

class PoseNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.register_buffer('_mean', torch.tensor([255 / 2] * IN_CHANS).view(1, IN_CHANS, 1, 1), persistent=True)
    self.register_buffer('_std', torch.tensor([255 / 4] * IN_CHANS).view(1, IN_CHANS, 1, 1), persistent=True)
    self.vision = timm.create_model('fastvit_t12', pretrained=False, num_classes=VISION_FEATURES, in_chans=IN_CHANS, act_layer=timm.layers.get_act_layer(ACT_LAYER))
    self.summarizer = nn.Sequential(nn.Linear(VISION_FEATURES, SUMMARY_FEATURES), nn.ReLU(inplace=True), ResBlock(SUMMARY_FEATURES))
    self.hydra = Hydra(num_features=SUMMARY_FEATURES, heads=HEADS)

  def preprocess_input(self, x):
    batch_size, seq_len, *_ = x.shape
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len, c=3)
    x = torch.nn.functional.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')
    return einops.rearrange(rgb_to_yuv6(x), '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6)

  def forward(self, x):
    vision_out = self.vision((x - self._mean) / self._std)
    summary = self.summarizer(vision_out)
    ret = self.hydra(summary)
    return ret

  def compute_distortion(self, out1, out2):
    distortion_heads = ['pose']
    return sum((out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(dim=tuple(range(1, out1[h.name].ndim))) for h in self.hydra.heads if h.name in distortion_heads) # MSE

  @torch.inference_mode()
  def debug_run(self, x, idx=0, keys=['pose']):
    from PIL import Image
    import os, tempfile
    f, filename = tempfile.mkstemp('.gif')
    os.close(f)
    x = self.preprocess_input(x)
    out = self(x)
    c = 0 # y00 animated for the seq_len consecutive frames - change c to see other yuv channels
    imgs = einops.rearrange(x, 'b (t c) h w -> b t c h w', t=seq_len, c=6)[idx, :, c, ...].to(dtype=torch.uint8).cpu().numpy()
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(filename, format="GIF", save_all=True, append_images=imgs[1:],  loop=0, duration=int(1000 / 1), optimize=True, disposal=2)
    viewer = get_viewer()
    if viewer: viewer.show_file(filename)
    else: print(f"saved to {filename}")
    print({h.name: out[h.name][idx,..., : h.out // 2] for h in self.hydra.heads if h.name in keys})

class SegNet(smp.Unet):
  def __init__(self):
    super().__init__('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)

  def preprocess_input(self, x):
    x = x[:, -1, ...] # Use only last frame
    return torch.nn.functional.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')

  def compute_distortion(self, out1, out2):
    diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
    return diff.mean(dim=tuple(range(1, diff.ndim)))  # accuracy

  @torch.inference_mode()
  def debug_run(self, x, idx=0):
    from PIL import Image
    import os, tempfile
    f, filename = tempfile.mkstemp('.png')
    os.close(f)
    x = self.preprocess_input(x)
    out = self(x)
    img = 0.5 * x + 0.5 * out.argmax(dim=1, keepdim=True) * (255 / 5)
    img = img[idx].to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(filename)
    viewer = get_viewer()
    if viewer: viewer.show_file(filename)
    else: print(f"saved to {filename}")

class DistortionNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.posenet = PoseNet()
    self.segnet = SegNet()

  def load_state_dicts(self, posenet_sd_path, segnet_sd_path, device):
    posenet_sd = load_file(posenet_sd_path, device=str(device))
    segnet_sd = load_file(segnet_sd_path, device=str(device))

    self.posenet.load_state_dict(posenet_sd)
    self.segnet.load_state_dict(segnet_sd)

  def preprocess_input(self, x):
    batch_size, seq_len, *_ = x.shape
    x = einops.rearrange(x, 'b t h w c -> b t c h w', b=batch_size, t=seq_len, c=3).float()
    posenet_in = self.posenet.preprocess_input(x)
    segnet_in = self.segnet.preprocess_input(x)
    return posenet_in, segnet_in

  def forward(self, x):
    posenet_in, segnet_in = self.preprocess_input(x)
    return self.posenet(posenet_in), self.segnet(segnet_in)  # TODO run in bfloat16?

  @torch.inference_mode()
  def compute_distortion(self, x, y):
    posenet_out_x, segnet_out_x = self(x)
    posenet_out_y, segnet_out_y = self(y)
    return self.posenet.compute_distortion(posenet_out_x, posenet_out_y), self.segnet.compute_distortion(segnet_out_x, segnet_out_y)

if __name__ == "__main__":
  from frame_utils import DaliVideoDataset, AVVideoDataset, seq_len, camera_size
  batch_size = 8
  if torch.cuda.is_available():
    device = torch.device('cuda', 0)
  elif torch.backends.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')
  files = (HERE / 'public_test_video_names.txt').read_text().splitlines()
  uncompressed_data_dir = Path('./videos/')
  DsClaas = DaliVideoDataset if device.type == 'cuda' else AVVideoDataset
  ds = DsClaas(files, data_dir=uncompressed_data_dir, batch_size=batch_size, device=device)
  ds.prepare_data()
  segnet = SegNet().eval().to(device)
  segnet_sd = load_file(segnet_sd_path, device=str(device))
  segnet.load_state_dict(segnet_sd)
  posenet = PoseNet().eval().to(device)
  posenet_sd = load_file(posenet_sd_path, device=str(device))
  posenet.load_state_dict(posenet_sd)

  for (_,_,batch) in ds:
    assert batch.shape == (batch_size, seq_len, camera_size[1], camera_size[0], 3), f"unexpected batch shape: {batch.shape}"
    batch = einops.rearrange(batch, 'b t h w c -> b t c h w', b=batch_size, t=seq_len, c=3).float().to(device)
    segnet.debug_run(batch)
    posenet.debug_run(batch)
    break
