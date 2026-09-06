"""Trainable PP-DocLayoutV3: detection loss plus the reading-order loss.

``transformers`` ships ``PPDocLayoutV3ForObjectDetection`` inference-only — its forward
raises if you pass labels. This subclass unblocks training by calling the inner model
with labels (which is also what builds the contrastive-denoising groups), reusing the
base RT-DETR detection loss over its outputs, and adding the locality-weighted GCE order
loss on the pretrained order logits.

The warm start matters: backbone, encoder, decoder, mask and **order** heads all come
from the document-pretrained checkpoint. Only the classification heads are re-initialized,
because PaddleX's taxonomy is not ours. Training the order head from scratch instead
throws away the one part of PP-DocLayoutV3 that is hard to reproduce.

Verified against transformers 5.13.1: the internals this relies on
(``RTDetrHungarianMatcher``, ``loss_function``, and the ``intermediate_logits`` /
``intermediate_reference_points`` / ``out_order_logits`` / ``enc_topk_*`` /
``denoising_meta_values`` outputs) are all present. They are internals, so a
transformers upgrade is a deliberate, re-tested step rather than a free one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bodhan_genai.ocr.training.order_loss import locality_gce

if TYPE_CHECKING:
    import torch

# RT-DETR loss and matcher settings the PP-DocLayoutV3 config does not carry, because it
# was only ever configured for inference. These are the RT-DETR defaults.
_RTDETR_LOSS_DEFAULTS: dict[str, Any] = {
    "use_focal_loss": True,
    "auxiliary_loss": True,
    "weight_loss_vfl": 1.0,
    "weight_loss_bbox": 5.0,
    "weight_loss_giou": 2.0,
    "matcher_class_cost": 2.0,
    "matcher_bbox_cost": 5.0,
    "matcher_giou_cost": 2.0,
    "matcher_alpha": 0.25,
    "matcher_gamma": 2.0,
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2.0,
    "eos_coefficient": 1e-4,
}


def _build_class() -> type:
    """Define the subclass lazily, so importing this module does not import torch."""
    import torch.nn as nn
    from transformers import PPDocLayoutV3ForObjectDetection
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher
    from transformers.utils import ModelOutput

    @dataclass
    class PPDocOutput(ModelOutput):
        loss: torch.FloatTensor | None = None
        logits: torch.FloatTensor | None = None
        pred_boxes: torch.FloatTensor | None = None
        order_logits: torch.FloatTensor | None = None
        last_hidden_state: torch.FloatTensor | None = None

    class PPDocLayoutV3Trainable(PPDocLayoutV3ForObjectDetection):
        """PP-DocLayoutV3 with a detection + reading-order training objective."""

        output_class = PPDocOutput

        def __init__(self, config):
            super().__init__(config)
            self.lambda_order = getattr(config, "lambda_order", 5.0)
            self._matcher = RTDetrHungarianMatcher(config)
            self.loss_type = "RTDetrForObjectDetection"

        @classmethod
        def build(cls, checkpoint, num_labels, id2label, label2id, lambda_order=5.0):
            from transformers import PPDocLayoutV3Config

            config = PPDocLayoutV3Config.from_pretrained(
                checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
            )
            config.lambda_order = lambda_order
            config.loss_type = "RTDetrForObjectDetection"
            # PP-DocLayoutV3's denoising path sizes its embedding at num_labels but pads
            # with num_labels, so it indexes off the end. It was never exercised because
            # HF blocks training entirely. Disabled rather than patched: it is a
            # convergence aid, not a requirement. Re-enable by widening
            # denoising_class_embed by one.
            config.num_denoising = 0
            for key, value in _RTDETR_LOSS_DEFAULTS.items():
                if not hasattr(config, key):
                    setattr(config, key, value)

            model = cls.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)
            # Re-init only the class heads: PaddleX's document classes are not ours.
            # Everything else — including the order head — keeps its pretrained weights.
            # Two distinct shapes identify a class head: the Linear that emits one logit
            # per class, and the denoising Embedding that carries an extra "no object"
            # row. Matching on shape rather than on name because the head is nested
            # differently across decoder layers and auxiliary outputs.
            for module in model.modules():
                if (isinstance(module, nn.Linear) and module.out_features == num_labels) or (
                    isinstance(module, nn.Embedding) and module.num_embeddings == num_labels + 1
                ):
                    module.reset_parameters()
            return model

        def _order_loss(self, order_logits, logits, pred_boxes, labels):
            """Order loss over the queries the Hungarian matcher assigned to real boxes.

            Scoring every query would train the order head against the ~300 - n
            unmatched queries, which have no ground-truth rank at all.
            """
            indices = self._matcher({"logits": logits, "pred_boxes": pred_boxes}, labels)
            total, matched_images = 0.0, 0
            for image, (source, target) in enumerate(indices):
                if source.numel() < 2:
                    continue
                order = labels[image]["reading_order"][target]
                scores = order_logits[image][source][:, source]
                total = total + locality_gce(scores, order)
                matched_images += 1
            if not matched_images:
                return order_logits.sum() * 0.0
            return total / matched_images

        def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            denoising = outputs.denoising_meta_values if self.training else None
            outputs_class = outputs.intermediate_logits
            outputs_coord = outputs.intermediate_reference_points
            logits, pred_boxes = outputs_class[:, -1], outputs_coord[:, -1]
            order_logits = outputs.out_order_logits[:, -1]  # [B, queries, queries]

            loss = None
            if labels is not None:
                loss, _, _ = self.loss_function(
                    logits,
                    labels,
                    self.device,
                    pred_boxes,
                    self.config,
                    outputs_class,
                    outputs_coord,
                    enc_topk_logits=outputs.enc_topk_logits,
                    enc_topk_bboxes=outputs.enc_topk_bboxes,
                    denoising_meta_values=denoising,
                )
                loss = loss + self.lambda_order * self._order_loss(
                    order_logits, logits, pred_boxes, labels
                )

            return PPDocOutput(
                loss=loss,
                logits=logits,
                pred_boxes=pred_boxes,
                order_logits=order_logits,
                last_hidden_state=outputs.last_hidden_state,
            )

    return PPDocLayoutV3Trainable


_CLASS = None


def trainable_class() -> type:
    """The ``PPDocLayoutV3Trainable`` class, built on first use."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_class()
    return _CLASS


def build_model(
    checkpoint: str,
    *,
    num_labels: int | None = None,
    lambda_order: float = 5.0,
):
    """Warm-start a trainable detector from a PP-DocLayoutV3 checkpoint."""
    from bodhan_genai.ocr.data.taxonomy import ID2LABEL, LABEL2ID, NUM_CLASSES

    num_labels = NUM_CLASSES if num_labels is None else num_labels
    return trainable_class().build(
        checkpoint,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        lambda_order=lambda_order,
    )
