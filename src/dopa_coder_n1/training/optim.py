from __future__ import annotations

import math

import torch


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    state_device: str = "cpu",
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embedding" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if state_device == "cpu":
        return CPUAdamW(groups, lr=lr, betas=(0.9, 0.95))
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


class CPUAdamW(torch.optim.Optimizer):
    """AdamW with optimizer state kept on CPU.

    This is a portable reference implementation for single-machine offload. It
    favors memory placement over speed.
    """

    def __init__(self, params, lr: float, betas: tuple[float, float], eps: float = 1e-8):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().to(device="cpu", dtype=torch.float32)
                state = self.state[param]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.long)
                    state["exp_avg"] = torch.zeros_like(grad, device="cpu")
                    state["exp_avg_sq"] = torch.zeros_like(grad, device="cpu")
                    state["master"] = param.detach().to(device="cpu", dtype=torch.float32).clone()
                state["step"] += 1
                step = int(state["step"].item())
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                master = state["master"]
                if weight_decay:
                    master.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * (bias_correction2**0.5) / bias_correction1
                master.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
                param.copy_(master.to(device=param.device, dtype=param.dtype))
        return loss


class CosineWithWarmup:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, total: int, min_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup = max(1, warmup)
        self.total = max(self.warmup + 1, total)
        self.min_ratio = min_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup:
            ratio = self.step_num / self.warmup
        else:
            progress = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            ratio = self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * ratio
