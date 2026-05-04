"""Muon optimizer (Keller Jordan, 2024). Newton-Schulz orthogonalized momentum.

We add explicit decoupled weight decay (researcher-recommended; Chen-Li-Liu
arXiv:2506.15054 — Muon's spectral-norm KKT story requires WD to be active).
4D conv weights are flattened to 2D for the NS step.

Used for hidden Conv2d weights (blocks + skips + refine + 1x1's). AdamW handles
the stem Linear + RGB heads + biases + latents.

Reference: https://github.com/KellerJordan/Muon
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if G.dtype == torch.float32 else G.clone()
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B_ = b * A + c * A @ A
        X = a * X + B_ @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                # Decoupled weight decay applied to the parameter directly,
                # before the orthogonalized update — matches AdamW convention.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                gu = g.add(buf, alpha=momentum) if nesterov else buf

                orig_shape = gu.shape
                if gu.ndim == 4:
                    g2d = gu.view(gu.size(0), -1)
                    g_ortho = zeropower_via_newtonschulz5(g2d, steps=ns_steps)
                    scale = max(1.0, (g2d.size(0) / g2d.size(1)) ** 0.5)
                    g_final = (g_ortho * scale).view(orig_shape)
                elif gu.ndim == 2:
                    g_ortho = zeropower_via_newtonschulz5(gu, steps=ns_steps)
                    scale = max(1.0, (gu.size(0) / gu.size(1)) ** 0.5)
                    g_final = g_ortho * scale
                else:
                    g_final = gu

                p.add_(g_final, alpha=-lr)
        return loss


def partition_params_for_muon(model):
    """Split params into (muon, adamw):
      - Muon: 2D+ weights NOT in stem and NOT in RGB heads
      - AdamW: stem Linear, rgb_0/rgb_1 weights, all biases, all 1D params
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            adamw_params.append(p)
            continue
        low = name.lower()
        if 'stem' in low or low.startswith('rgb') or '.rgb_' in low:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params
