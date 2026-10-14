"""
ASAM — Adaptive Sharpness-Aware Minimization
Paper: "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning"
       Kwon et al., ICML 2021. https://arxiv.org/abs/2102.11600

Ported to pure PyTorch, no extra dependency needed.
Usage:
    base_opt  = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    optimizer = ASAM(base_opt, model, rho=0.5, eta=0.01)
    ...
    loss.backward()
    optimizer.ascent_step()
    loss2.backward()
    optimizer.descent_step()
"""

import torch


class ASAM:
    """
    Adaptive Sharpness-Aware Minimization optimizer wrapper.

    Args:
        optimizer : underlying optimizer (SGD, Adam, AdamW, …)
        model     : the model being trained
        rho       : perturbation radius (default 0.5)
        eta       : adaptive scaling lower bound (default 0.01)
    """

    def __init__(self, optimizer, model, rho: float = 0.5, eta: float = 0.01):
        self.optimizer = optimizer
        self.model     = model
        self.rho       = rho
        self.eta       = eta
        self._saved    = {}     # saved params for restore

    @torch.no_grad()
    def ascent_step(self):
        """
        Compute adaptive perturbation ε̂ and move parameters to θ + ε̂.
        Must be called AFTER loss.backward() (first backward).
        """
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            # Adaptive scaling: T_i = max(|θ_i|, η)
            t = p.abs().clamp(min=self.eta)
            # Adaptive gradient: g̃_i = T_i * ∇_i
            g_tilde = t * p.grad
            grads.append(g_tilde.norm(p=2))

        # Compute global norm of adaptive gradients
        grad_norm = torch.stack(grads).norm(p=2) + 1e-12

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t = p.abs().clamp(min=self.eta)
            # ε̂_i = (ρ / ||g̃||) * T_i * g_i
            eps = (self.rho / grad_norm) * t * p.grad
            # Save original parameter
            self._saved[n] = p.data.clone()
            # Move to perturbed point
            p.data.add_(eps)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        """
        Restore parameters to θ and apply the actual gradient update.
        Must be called AFTER second loss.backward().
        """
        for n, p in self.model.named_parameters():
            if n in self._saved:
                p.data = self._saved[n]
        self._saved.clear()
        self.optimizer.step()
        self.optimizer.zero_grad()

    # ── Delegate common optimizer methods ────────────────────────────────────
    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class SAM(ASAM):
    """
    Vanilla SAM (non-adaptive). Same interface as ASAM.
    Kept for ablation comparison.
    """

    @torch.no_grad()
    def ascent_step(self):
        grads = [p.grad.norm(p=2) for p in self.model.parameters()
                 if p.grad is not None]
        grad_norm = torch.stack(grads).norm(p=2) + 1e-12

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = (self.rho / grad_norm) * p.grad
            self._saved[n] = p.data.clone()
            p.data.add_(eps)

        self.optimizer.zero_grad()
