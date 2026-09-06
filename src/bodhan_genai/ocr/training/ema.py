"""Exponential moving average of the weights.

Detection training on a mixed corpus is noisy step to step — each batch is a different
blend of handwritten and printed pages — and the last checkpoint of a run is just
whichever point the noise left you at. An EMA shadow is consistently the better model to
evaluate and to ship, usually by a point or two of mAP, for the cost of one extra copy
of the weights.

The decay warms up: early on the shadow is dominated by a near-random init, so a fixed
0.9998 would take tens of thousands of steps to forget it. ``decay(step)`` ramps from
almost nothing to the target with the usual ``1 - exp(-step / warmup)`` schedule.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_MISSING = "EMA has no shadow for {name!r}; the model changed shape since the EMA was built"


class ModelEma:
    """A shadow copy of the model's floating-point parameters and buffers."""

    def __init__(self, model, *, decay: float = 0.9998, warmup: int = 2000, device=None) -> None:
        import torch

        self.decay = decay
        self.warmup = max(1, warmup)
        self.device = device
        self.updates = 0
        with torch.no_grad():
            self.shadow = {
                name: value.detach().clone().to(device or value.device)
                for name, value in self._state(model).items()
            }

    @staticmethod
    def _state(model) -> dict[str, torch.Tensor]:
        source = model.module if hasattr(model, "module") else model
        return {k: v for k, v in source.state_dict().items() if v.dtype.is_floating_point}

    def current_decay(self) -> float:
        return self.decay * (1 - math.exp(-self.updates / self.warmup))

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            self.updates += 1
            decay = self.current_decay()
            for name, value in self._state(model).items():
                shadow = self.shadow.get(name)
                if shadow is None:
                    raise KeyError(_MISSING.format(name=name))
                shadow.mul_(decay).add_(value.detach().to(shadow.device), alpha=1 - decay)

    def copy_to(self, model) -> None:
        """Write the shadow into ``model`` in place."""
        import torch

        target = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            for name, value in target.state_dict().items():
                if name in self.shadow:
                    value.copy_(self.shadow[name].to(value.device))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup, "updates": self.updates,
                "shadow": self.shadow}  # fmt: skip

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.warmup = state["warmup"]
        self.updates = state["updates"]
        self.shadow = state["shadow"]
