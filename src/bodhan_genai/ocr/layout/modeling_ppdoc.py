"""IndicDocLayout: trainable PP-DocLayoutV3 (document-pretrained strong init).

HF ships PPDocLayoutV3ForObjectDetection inference-only (forward raises on labels).
This subclass unblocks training: it calls the inner model with labels (which builds the
contrastive-denoising groups), reuses the base RT-DETR detection loss on its outputs, and
adds our locality-weighted GCE order loss on its (pretrained) order_logits.
Backbone + decoder + order/mask heads start from the document-pretrained checkpoint;
only the class heads are re-init'd for our 37 education classes.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PPDocLayoutV3Config, PPDocLayoutV3ForObjectDetection
from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher
from transformers.utils import ModelOutput

from .order_loss import locality_gce


@dataclass
class PPDocOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    order_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


class PPDocLayoutV3Trainable(PPDocLayoutV3ForObjectDetection):
    def __init__(self, config):
        super().__init__(config)
        self.lambda_order = getattr(config, "lambda_order", 5.0)
        self._matcher = RTDetrHungarianMatcher(config)
        self.loss_type = "RTDetrForObjectDetection"  # base RT-DETR loss over its outputs

    @classmethod
    def build(cls, ckpt, num_labels, id2label, label2id, lambda_order=5.0):
        config = PPDocLayoutV3Config.from_pretrained(
            ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        config.lambda_order = lambda_order
        config.loss_type = "RTDetrForObjectDetection"
        # PP-DocLayoutV3's denoising path is buggy (embed size num_labels but pads with
        # num_labels -> index error); it was never run since HF blocks training. Disable it
        # (optional convergence aid). Re-enable later by resizing denoising_class_embed to +1.
        config.num_denoising = 0
        # RT-DETR loss/matcher fields the base config lacks
        defaults = {
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
        for k, v in defaults.items():
            if not hasattr(config, k):
                setattr(config, k, v)
        model = cls.from_pretrained(ckpt, config=config, ignore_mismatched_sizes=True)
        # re-init class heads (paddle doc classes -> our education classes); keep everything else
        for m in model.modules():
            if (isinstance(m, nn.Linear) and m.out_features == num_labels) or (
                isinstance(m, nn.Embedding) and m.num_embeddings == num_labels + 1
            ):
                m.reset_parameters()
        return model

    def _order_loss(self, order_logits, logits, pred_boxes, labels):
        idx = self._matcher({"logits": logits, "pred_boxes": pred_boxes}, labels)
        tot, n = 0.0, 0
        for b, (src, tgt) in enumerate(idx):
            if src.numel() < 2:
                continue
            order = labels[b]["reading_order"][tgt]
            S = order_logits[b][src][:, src]
            tot = tot + locality_gce(S, order)
            n += 1
        return tot / max(n, 1) if n else order_logits.sum() * 0.0

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        dn = outputs.denoising_meta_values if self.training else None
        outputs_class = outputs.intermediate_logits
        outputs_coord = outputs.intermediate_reference_points
        logits, pred_boxes = outputs_class[:, -1], outputs_coord[:, -1]
        order_logits = outputs.out_order_logits[:, -1]  # [B, num_queries, num_queries]
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
                denoising_meta_values=dn,
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
