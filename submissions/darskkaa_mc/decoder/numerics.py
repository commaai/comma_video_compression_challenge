"""Decoder numerics setup: OFFICIAL decoder numerics = cuDNN ENABLED, TF32 off (runtime.hpac_inference.configure_cuda_reproducibility keeps cuDNN default kernels).
Earlier engines disabled cuDNN, which shifted rendered pixels slightly.  Import this module after any other engine setup."""
import torch
def official_flags():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"; torch.backends.cudnn.conv.fp32_precision = "ieee"
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
official_flags()
