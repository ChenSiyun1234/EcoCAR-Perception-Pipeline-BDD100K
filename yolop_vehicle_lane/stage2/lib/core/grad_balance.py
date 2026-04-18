"""
Multi-task gradient balancing for Stage2.

Two options implemented (choose one via cfg.TRAIN.GRAD_BALANCE):

1. **PCGrad** (Yu et al., NeurIPS 2020) — project each task's gradient
   onto the normal of any other task's gradient with which it has
   negative cosine similarity, then sum.

2. **Uncertainty Weighting** (Kendall et al., CVPR 2018) — learn a
   per-task log-variance; total loss is
   `Σ_i exp(-log_var_i) * L_i + log_var_i`. Simpler, cheaper, and in
   most multi-task driving-perception literature reasonably robust.

Both wrap a list of per-task losses. Callers pass losses as an
ordered list (e.g. `[det_total, lane_total, distill_total]`).

Notes
-----
PCGrad here is implemented per-param-group and does a full backward
pass per task, so it's more expensive than summed-loss.backward().
It's the right tool when you see negative cosine similarity between
det and lane gradients in the preflight check.
"""

from typing import List

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """Learned log-variance weights. Treat the output as the single
    scalar loss to backprop.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == len(self.log_vars), (
            f'got {len(losses)} losses, expected {len(self.log_vars)}')
        terms = []
        for i, L in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            terms.append(precision * L + self.log_vars[i])
        return torch.stack(terms).sum()


class PCGrad:
    """Project conflicting task gradients before the optimizer step.

    Usage:
        pc = PCGrad(optimizer)
        pc.zero_grad()
        pc.pc_backward([loss_det, loss_lane, loss_distill])
        pc.step()
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _flatten_grad(self, grads):
        flat = []
        shapes = []
        for g in grads:
            if g is None:
                continue
            flat.append(g.view(-1))
            shapes.append(g.shape)
        return (torch.cat(flat), shapes) if flat else (None, [])

    def pc_backward(self, losses: List[torch.Tensor]):
        """Compute conflict-averse gradients then assign to parameters.

        Does one backward pass per task on a fresh grad buffer, stores
        all task gradients, projects each pair, sums, writes back to
        `.grad` attributes. Optimizer.step() is the user's job.
        """
        params = [p for group in self.param_groups for p in group['params']
                  if p.requires_grad]
        task_grads = []
        for i, L in enumerate(losses):
            self.optimizer.zero_grad(set_to_none=True)
            L.backward(retain_graph=(i < len(losses) - 1))
            task_grads.append([(p.grad.detach().clone() if p.grad is not None else None)
                               for p in params])

        # Project pairwise.
        projected = [list(tg) for tg in task_grads]  # copy
        n = len(task_grads)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                gi, _ = self._flatten_grad(projected[i])
                gj, _ = self._flatten_grad(task_grads[j])
                if gi is None or gj is None:
                    continue
                dot = torch.dot(gi, gj)
                if dot >= 0:
                    continue
                gj_sq = torch.dot(gj, gj).clamp(min=1e-12)
                # gi <- gi - (gi·gj / ||gj||²) * gj
                scale = dot / gj_sq
                gi_new = gi - scale * gj
                # Scatter back into per-parameter shapes.
                idx = 0
                for k, g in enumerate(projected[i]):
                    if g is None:
                        continue
                    n_el = g.numel()
                    projected[i][k] = gi_new[idx:idx + n_el].view_as(g)
                    idx += n_el

        # Sum task gradients into each parameter's .grad.
        for k, p in enumerate(params):
            agg = None
            for tg in projected:
                g = tg[k]
                if g is None:
                    continue
                agg = g if agg is None else (agg + g)
            p.grad = agg
