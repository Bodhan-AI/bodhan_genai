"""Single-image inference for IndicDocLayout -> boxes, labels and reading order.

Emits ``[{bbox: [y0, x0, y1, x1] normalised to 0-1000, label, reading_order, score}]``. That is
the layout viewer's schema, kept verbatim so predictions render with the same box and
reading-order visualisation as the dataset viewer; ``engine.layout`` converts it to pixel
``[x0, y0, x1, y1]``.

The decode mirrors the training-time evaluation exactly: sigmoid-max detections, and reading
order from a voting sort over the pairwise ``order_logits`` restricted to the kept boxes.
"""

import numpy as np
import torch

from .labels import ID2LABEL
from .order_loss import decode_order, pairwise_scores

_CACHE = {}


def cxcywh_to_xyxy(b):
    """Centre-form boxes -> corner form, preserving the input tensor's dtype and device."""
    c = b.clone()
    c[..., 0], c[..., 1] = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
    c[..., 2], c[..., 3] = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    return c


def get_model(ckpt, device="cuda"):
    """Load (and cache) an IndicDocLayout checkpoint.

    The checkpoint is saved as ``PPDocLayoutV3Trainable`` -- our subclass -- so that class has to
    be importable here even though nothing is trained at inference time.
    """
    key = (ckpt, device)
    if key not in _CACHE:
        from .modeling_ppdoc import PPDocLayoutV3Trainable

        _CACHE[key] = PPDocLayoutV3Trainable.from_pretrained(ckpt).to(device).eval()
    return _CACHE[key]


@torch.no_grad()
def infer(model, pil_img, conf=0.5, img_size=1024, device="cuda"):
    """Detect blocks on one page image."""
    im = pil_img.convert("RGB").resize((img_size, img_size))
    # np.array (not asarray): a PIL buffer is read-only, and torch warns on every page about
    # wrapping a non-writable array.
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0)  # [3,S,S] in [0,1]
    out = model(pixel_values=x[None].to(device))
    scores, labels = out.logits.sigmoid().max(-1)  # [1,N]
    boxes = cxcywh_to_xyxy(out.pred_boxes)[0]  # [N,4] normalised x0,y0,x1,y1

    if getattr(out, "order_logits", None) is not None:
        order_scores = out.order_logits[0]
    else:
        nq = model.config.num_queries
        order_scores = pairwise_scores(out.last_hidden_state[:, -nq:], model.ro_q, model.ro_k)[0]

    keep = (scores[0] > conf).nonzero().squeeze(-1)
    if keep.numel() == 0:
        return []
    kept_boxes, kept_scores, kept_labels = boxes[keep], scores[0][keep], labels[0][keep]

    # Reading order is decoded over the KEPT sub-block only: ranking against suppressed queries
    # would leave gaps in the sequence.
    sub = order_scores[keep][:, keep].cpu().float()
    sequence = decode_order(sub).tolist()  # positions, first -> last
    rank = [0] * len(sequence)
    for r, position in enumerate(sequence):
        rank[position] = r + 1

    content = []
    for j in range(keep.numel()):
        x0, y0, x1, y1 = kept_boxes[j].tolist()
        content.append(
            {
                "bbox": [  # viewer schema: [y0, x0, y1, x1], 0-1000
                    round(y0 * 1000, 1),
                    round(x0 * 1000, 1),
                    round(y1 * 1000, 1),
                    round(x1 * 1000, 1),
                ],
                "label": ID2LABEL[int(kept_labels[j])],
                "reading_order": rank[j],
                "score": round(float(kept_scores[j]), 3),
            }
        )
    return content
