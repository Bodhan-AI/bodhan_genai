"""Reading order as pairwise precedence, and the loss that trains it.

The detector predicts an antisymmetric matrix ``S`` over its queries, where ``S[i, j] > 0``
means query *i* is read before query *j*. That is trained here with a **locality-weighted
Generalized Cross-Entropy**:

*   **GCE** rather than plain BCE because reading-order annotation is noisy — a page with
    two columns and a floating caption has more than one defensible order — and GCE's
    ``q`` bounds the gradient a confidently-wrong pair can contribute.
*   **Locality weighting** because getting adjacent blocks in the right order matters and
    getting block 2 against block 40 does not. Pairs are weighted ``exp(-|Δrank| / tau)``,
    so the loss spends its capacity where a human would notice the mistake.

Only the strict upper triangle is trained. PP-DocLayoutV3's GlobalPointer head masks
``a >= b`` to ``-1e4``, so the lower triangle holds no real score and training it just
penalizes entries the model cannot fix. For an antisymmetric head the upper triangle
determines the lower anyway, so nothing is lost either way.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def decode_order(scores: torch.Tensor) -> torch.Tensor:
    """Reading sequence from a pairwise score matrix.

    ``P(i before j) = sigmoid(S[i, j])`` for ``i < j`` and ``1 - sigmoid(S[j, i])``
    otherwise; ``votes[j]`` counts how many elements precede *j*, and sorting the votes
    ascending gives the sequence first-to-last.

    This matches PP-DocLayoutV3's own ``_get_order_seqs``, and is correct for both the
    triangular GlobalPointer head and a fully antisymmetric one — under this formula the
    two coincide. Decoding with the wrong convention is not an error, it silently
    produces a plausible but different order.
    """
    import torch

    probabilities = torch.sigmoid(scores)
    votes = probabilities.triu(1).sum(0) + (1.0 - probabilities.t()).tril(-1).sum(0)
    return torch.argsort(votes)


def pairwise_scores(queries: torch.Tensor, project_q, project_k) -> torch.Tensor:
    """``[B, N, d]`` decoder queries to an antisymmetric ``[B, N, N]`` score matrix."""
    a, k = project_q(queries), project_k(queries)
    m = a @ k.transpose(-1, -2)
    return (m - m.transpose(-1, -2)) / math.sqrt(a.shape[-1])


def locality_gce(
    scores: torch.Tensor,
    order: torch.Tensor,
    *,
    q: float = 0.7,
    tau: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Loss over ``scores[m, m]`` for matched queries with ground-truth ranks ``order[m]``.

    Returns a zero that still carries a gradient path when there is nothing to order, so
    a batch of single-block pages does not detach the order head from the graph.
    """
    import torch

    m = order.numel()
    if m < 2:
        return scores.sum() * 0.0

    precedes = (order.unsqueeze(1) < order.unsqueeze(0)).float()
    probability = torch.sigmoid(scores)
    correct = torch.where(precedes > 0.5, probability, 1 - probability).clamp(eps, 1)
    gce = (1 - correct.pow(q)) / q

    positions = order.argsort().argsort().float()  # dense ranks, robust to gaps in `order`
    distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    weight = torch.exp(-distance / tau)

    upper = torch.triu(torch.ones(m, m, dtype=torch.bool, device=scores.device), diagonal=1)
    return (gce * weight)[upper].sum() / weight[upper].sum().clamp_min(eps)
