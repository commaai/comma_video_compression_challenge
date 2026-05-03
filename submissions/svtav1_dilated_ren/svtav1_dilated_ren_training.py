# %% [markdown]
# Generated from svtav1_dilated_ren_training.ipynb
# Open in VS Code/Jupyter-compatible editors to run cell-by-cell.

# %% [markdown]
# # svtav1_dilated_ren — training notebook
#
# SVT-AV1 v2.3.0, CRF 36, 544x408, film-grain=22, keyint=180. Decoder-side Residual Enhancement Network (REN) with dilated convolutions, trained with task-aware loss (SegNet KL + PoseNet MSE). Weights int8-quantized and bz2-compressed to ~21 KB inside archive.zip.
#
# Runs top to bottom on a fresh Colab T4 (~35 min total).

# %%
import subprocess, sys
def run(c): return subprocess.run(c, shell=True, capture_output=True, text=True)

run('apt-get remove -y svt-av1 2>/dev/null; true')
run('apt-get update -qq && apt-get install -y -qq git-lfs ffmpeg cmake yasm nasm build-essential')

print('Building SVT-AV1 v2.3.0 (~4 min)...')
run('rm -rf /tmp/SVT-AV1 && cd /tmp && git clone --depth=1 --branch v2.3.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git')
run('mkdir -p /tmp/SVT-AV1/Build && cd /tmp/SVT-AV1/Build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && make -j$(nproc) && make install && ldconfig')
r = subprocess.run(['SvtAv1EncApp', '--version'], capture_output=True, text=True)
print(r.stdout[:80], r.stderr[:80])
assert 'v2.3' in (r.stdout + r.stderr), 'wrong version'

run(f'{sys.executable} -m pip install -q einops timm safetensors "segmentation-models-pytorch>=0.3" av tqdm pillow pandas torchvision opencv-python-headless')
try: run(f'{sys.executable} -m pip install -q --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120')
except: pass
print('done')

# %%
import os, shutil, io, bz2, tempfile, zipfile
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import av
from tqdm.auto import tqdm

REPO = Path('/content/challenge')
if not REPO.exists():
    subprocess.run(f'git clone https://github.com/commaai/comma_video_compression_challenge.git {REPO}', shell=True, check=True)
subprocess.run('git lfs install && git lfs pull', shell=True, cwd=str(REPO), check=True)
for p in [REPO/'videos/0.mkv', REPO/'models/segnet.safetensors', REPO/'models/posenet.safetensors']:
    print(f'  {p.name}: {p.stat().st_size/1e6:.1f} MB')

sys.path.insert(0, str(REPO)); os.chdir(str(REPO))
from frame_utils import yuv420_to_rgb, camera_size
from modules import DistortionNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIDEO_IN = str(REPO / 'videos/0.mkv')
VIDEO_NAMES_FILE = REPO / 'public_test_video_names.txt'
MW, MH = 512, 384   # model input resolution
CW, CH = 1164, 874  # camera resolution
print(f'device: {DEVICE}')

# %%
dn = DistortionNet().eval().to(DEVICE)
dn.load_state_dicts(str(REPO/'models/posenet.safetensors'), str(REPO/'models/segnet.safetensors'), DEVICE)
for p in dn.parameters(): p.requires_grad_(False)
segnet, posenet = dn.segnet, dn.posenet
print('models loaded')

# %%
def encode_svtav1(out_mkv, vf, crf=36, fg=22, keyint=180):
    td = Path(tempfile.mkdtemp()); y4m = td/'i.y4m'; ivf = td/'o.ivf'
    r = subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
                        '-r','20','-i',VIDEO_IN,'-vf',vf,'-pix_fmt','yuv420p',
                        '-f','yuv4mpegpipe',str(y4m)], capture_output=True, text=True)
    if r.returncode != 0: shutil.rmtree(td); raise RuntimeError(r.stderr[-500:])
    r = subprocess.run(['SvtAv1EncApp','-i',str(y4m),'-b',str(ivf),
                        '--preset','0','--crf',str(crf),'--film-grain',str(fg),
                        '--keyint',str(keyint),'--scd','0','--rc','0'],
                       capture_output=True, text=True)
    if r.returncode != 0: shutil.rmtree(td); raise RuntimeError(r.stderr[-500:])
    r = subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error',
                        '-r','20','-i',str(ivf),'-c','copy',str(out_mkv)],
                       capture_output=True, text=True)
    shutil.rmtree(td)
    if r.returncode != 0: raise RuntimeError(r.stderr[-500:])

ENC_VF = 'scale=544:408:flags=lanczos'
ENC_CRF = 36

TMP = REPO / 'tmp_train.mkv'
encode_svtav1(TMP, ENC_VF, crf=ENC_CRF)
print(f'encoded: {TMP.stat().st_size:,} bytes')

def load_decoded_512(path):
    fs=[]; c=av.open(str(path))
    for fr in c.decode(c.streams.video[0]):
        t=yuv420_to_rgb(fr).float().permute(2,0,1).unsqueeze(0)
        t=F.interpolate(t, size=(MH,MW), mode='bicubic', align_corners=False).clamp(0,255).squeeze(0)
        fs.append(t)
    c.close(); return torch.stack(fs)

def load_original_full():
    fs=[]; c=av.open(VIDEO_IN)
    for fr in c.decode(c.streams.video[0]):
        fs.append(yuv420_to_rgb(fr).float().permute(2,0,1))
    c.close(); return torch.stack(fs)

print('loading frames...')
X = load_decoded_512(TMP)
Y = load_original_full()
N = X.shape[0]
print(f'X: {X.shape}  Y: {Y.shape}')

# %%
gt_seg, gt_pose = [], []
with torch.no_grad():
    for i in tqdm(range(N), desc='GT outputs'):
        f0=Y[i].to(DEVICE); f1=Y[min(i+1,N-1)].to(DEVICE)
        pair=torch.stack([f0,f1]).unsqueeze(0)
        s=segnet(segnet.preprocess_input(pair))
        gt_seg.append(F.softmax(s,dim=1).cpu())
        po=posenet(posenet.preprocess_input(pair))
        gt_pose.append({k: v[..., :v.shape[-1]//2].cpu() for k,v in po.items()})

def upscale(x):
    return F.interpolate(x.unsqueeze(0), size=(CH,CW), mode='bicubic', align_corners=False).clamp(0,255).squeeze(0)

bs, bp = 0.0, 0.0
with torch.no_grad():
    for i in tqdm(range(N), desc='baseline distortion'):
        f0=upscale(X[i]).to(DEVICE); f1=upscale(X[min(i+1,N-1)]).to(DEVICE)
        pair=torch.stack([f0,f1]).unsqueeze(0)
        s=segnet(segnet.preprocess_input(pair))
        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()
        po=posenet(posenet.preprocess_input(pair))
        gp={k:v.to(DEVICE) for k,v in gt_pose[i].items()}
        bp += sum(F.mse_loss(po[k][..., :po[k].shape[-1]//2], gp[k]) for k in po).item()
bs /= N; bp /= N

# normalize loss weights so each term starts at ~1.0
W_S = 1.0 / max(bs, 1e-8)
W_P = 1.0 / max(bp, 1e-8)
W_T = 0.01
print(f'baseline  seg={bs:.4f}  pose={bp:.5f}')
print(f'weights   W_S={W_S:.4f}  W_P={W_P:.2f}  W_T={W_T}')

# %%
class REN(nn.Module):
    """Residual Enhancement Network — dilated convs (dilation 1/2/4) for wider
    receptive field at same parameter count as a plain 4-conv REN."""
    def __init__(self, c=32):
        super().__init__()
        self.c0 = nn.Conv2d(3, c, 3, padding=1)
        self.c1 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(c, c, 3, padding=4, dilation=4)
        self.c3 = nn.Conv2d(c, 3, 3, padding=1)
    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        return x + self.c3(h)

ren = REN(32).to(DEVICE)
print(f'params: {sum(p.numel() for p in ren.parameters())}')

torch.manual_seed(42)
perm = torch.randperm(N - 1)
val_idx   = perm[:200].tolist()
train_idx = perm[200:].tolist()

EPOCHS = 25
opt = torch.optim.Adam(ren.parameters(), lr=5e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

def step(i, m):
    x0 = X[i].to(DEVICE) / 255.0
    x1 = X[min(i+1, N-1)].to(DEVICE) / 255.0
    r0 = m(x0.unsqueeze(0)) * 255.0
    r1 = m(x1.unsqueeze(0)) * 255.0
    f0 = F.interpolate(r0, size=(CH,CW), mode='bicubic', align_corners=False).clamp(0,255).squeeze(0)
    f1 = F.interpolate(r1, size=(CH,CW), mode='bicubic', align_corners=False).clamp(0,255).squeeze(0)
    pair = torch.stack([f0, f1]).unsqueeze(0)
    s  = segnet(segnet.preprocess_input(pair))
    sl = F.kl_div(F.log_softmax(s, dim=1), gt_seg[i].to(DEVICE), reduction='batchmean')
    po = posenet(posenet.preprocess_input(pair))
    gp = {k: v.to(DEVICE) for k,v in gt_pose[i].items()}
    pl = sum(F.mse_loss(po[k][..., :po[k].shape[-1]//2], gp[k]) for k in po)
    tl = (r0 - r1).abs().mean()
    return sl, pl, tl

best_val = float('inf'); best_sd = None
for ep in range(EPOCHS):
    ren.train(); np.random.shuffle(train_idx)
    for i in tqdm(train_idx, desc=f'ep{ep:02d}', leave=False):
        sl, pl, tl = step(i, ren)
        loss = W_S*sl + W_P*pl + W_T*tl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ren.parameters(), 1.0)
        opt.step()
    sch.step()
    ren.eval(); vs, vp = 0.0, 0.0
    with torch.no_grad():
        for i in val_idx:
            sl, pl, _ = step(i, ren); vs += sl.item(); vp += pl.item()
    vs /= len(val_idx); vp /= len(val_idx)
    proxy = W_S*vs + W_P*vp
    marker = ''
    if proxy < best_val:
        best_val = proxy
        best_sd = {k: v.detach().cpu().clone() for k,v in ren.state_dict().items()}
        marker = ' **'
    print(f'ep{ep:02d}  val seg={vs:.4f}  pose={vp:.5f}  proxy={proxy:.3f}{marker}')

ren.load_state_dict(best_sd); ren.eval()
print(f'\nbest val proxy: {best_val:.3f}')

# %%
ren.cpu()
qsd = {}
for k, v in ren.state_dict().items():
    if v.dtype == torch.float32 and v.numel() > 1:
        sc = v.abs().max() / 127.0
        qsd[k] = (v / sc).round().clamp(-128, 127).to(torch.int8)
        qsd[k + '.scale'] = sc
    else:
        qsd[k] = v
buf = io.BytesIO(); torch.save(qsd, buf)
REN_BZ2 = bz2.compress(buf.getvalue(), compresslevel=9)
print(f'raw: {len(buf.getvalue()):,} bytes  →  bz2: {len(REN_BZ2):,} bytes ({len(REN_BZ2)/1024:.1f} KB)')
ren.to(DEVICE)

# %%
SUB = REPO / 'submissions' / 'svtav1_dilated_ren'
SUB.mkdir(parents=True, exist_ok=True)
(SUB / '__init__.py').write_text('')
(SUB / 'ren.bz2').write_bytes(REN_BZ2)

AD = SUB / 'archive'
shutil.rmtree(AD, ignore_errors=True); AD.mkdir()
encode_svtav1(AD / '0.mkv', ENC_VF, crf=ENC_CRF)
shutil.copy(SUB / 'ren.bz2', AD / 'ren.bz2')
subprocess.run(['zip', '-r', str(SUB/'archive.zip'), '.'],
               check=True, capture_output=True, cwd=str(AD))

print(f'archive.zip: {(SUB/"archive.zip").stat().st_size:,} bytes')
with zipfile.ZipFile(SUB / 'archive.zip') as z:
    for info in z.infolist():
        print(f'  {info.filename:12s}  {info.file_size:>10,} bytes')

# %%
r = subprocess.run(
    ['bash', str(REPO/'evaluate.sh'), '--submission-dir', str(SUB), '--device', 'cuda'],
    capture_output=True, text=True, cwd=str(REPO)
)
print(r.stdout)
if r.returncode != 0:
    print('STDERR:', r.stderr[-1000:])

rt = SUB / 'report.txt'
if rt.exists():
    print(rt.read_text())
