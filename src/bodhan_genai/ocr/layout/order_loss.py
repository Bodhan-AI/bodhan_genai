"""Locality-weighted Generalized Cross-Entropy on the antisymmetric pairwise
precedence scores. Validated in learnings (recovers arbitrary order, tau=1.0).
"""

import math

import torch


def decode_order(S):
    """Reading order from a pairwise score matrix, matching PP-DocLayoutV3's _get_order_seqs.
    P(i before j) = sigmoid(S[i,j]) if i<j else 1-sigmoid(S[j,i]); votes[j]=#elements before j;
    argsort ascending -> reading sequence (first..last). Correct for BOTH the triangular ppdoc
    GlobalPointer AND an antisymmetric head (they coincide under this formula)."""
    sc = torch.sigmoid(S)
    votes = sc.triu(1).sum(0) + (1.0 - sc.t()).tril(-1).sum(0)
    return torch.argsort(votes)


def pairwise_scores(Q, ro_q, ro_k):
    """Q:[B,N,d] -> antisymmetric S:[B,N,N], S_ij>0 => query i precedes j."""
    A, K = ro_q(Q), ro_k(Q)  # [B,N,r]
    M = A @ K.transpose(-1, -2)  # M_ij = q_i^T Wq^T Wk q_j
    return (M - M.transpose(-1, -2)) / math.sqrt(A.shape[-1])


def locality_gce(S, order, q=0.7, tau=3.0, eps=1e-6):
    """S:[m,m] scores over matched queries, order:[m] their GT reading_order values."""
    m = order.numel()
    if m < 2:
        return S.sum() * 0.0  # nothing to order; keep graph
    P = (order.unsqueeze(1) < order.unsqueeze(0)).float()  # P_ab=1 if a precedes b
    p = torch.sigmoid(S)
    pc = torch.where(P > 0.5, p, 1 - p).clamp(eps, 1)
    gce = (1 - pc.pow(q)) / q
    pos = order.argsort().argsort().float()  # dense rank positions
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    W = torch.exp(-dist / tau)
    # only the strict UPPER triangle carries real scores (ppdoc GlobalPointer masks a>=b to -1e4);
    # training the masked lower triangle penalizes unfixable entries. Upper-tri also suffices for
    # an antisymmetric head. P[i,j] for i<j = (order_i < order_j).
    mask = torch.triu(torch.ones(m, m, dtype=torch.bool, device=S.device), diagonal=1)
    return (gce * W)[mask].sum() / W[mask].sum().clamp_min(eps)
