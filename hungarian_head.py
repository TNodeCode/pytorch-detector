"""DETR-style detection head with Hungarian-matching set-based loss.

This module provides:

* :class:`SinusoidalPositionEncoding2D` – 2-D sinusoidal position encoding for
  spatial feature maps.
* :class:`HungarianMatcher` – optimal bipartite matching between predicted and
  ground-truth objects using the Hungarian algorithm
  (:func:`scipy.optimize.linear_sum_assignment`).
* :class:`HungarianDetectionHead` – transformer-decoder detection head that
  combines the matcher with a set-based training loss (classification,
  L1-box, and GIoU).
* :class:`DINOv2HungarianDetectionModel` – complete ``nn.Module`` that pairs a
  :class:`~feature_extractors.DINOv2ViT` backbone with a
  :class:`HungarianDetectionHead`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou

from feature_extractors import DINOv2ViT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    """Build a fully-connected MLP with ReLU activations between layers.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Width of every hidden layer.
        output_dim (int): Dimensionality of the final output.
        num_layers (int): Total number of linear layers (including the output
            layer).  Must be ≥ 1.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers: list[nn.Module] = []
    for i in range(num_layers):
        in_d = input_dim if i == 0 else hidden_dim
        out_d = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_d, out_d))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionEncoding2D(nn.Module):
    """2-D sinusoidal positional encoding for spatial feature maps.

    Generates a fixed (non-learned) positional encoding following the DETR
    convention: the spatial axes are encoded independently with sine/cosine
    functions, then concatenated along the channel dimension.

    Args:
        hidden_dim (int): Total number of encoding channels (must be even).
        temperature (float): Denominator base used in the sinusoidal
            frequency computation.  Default: 10 000.
    """

    def __init__(self, hidden_dim: int, temperature: float = 10000.0):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for 2-D sinusoidal encoding.")
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute the positional encoding for a spatial mask.

        Args:
            mask (torch.Tensor): Boolean tensor of shape ``(B, H, W)`` where
                ``True`` marks padding positions.

        Returns:
            torch.Tensor: Positional encoding of shape ``(B, H*W, hidden_dim)``.
        """
        B, H, W = mask.shape
        device = mask.device

        not_mask = ~mask
        # Cumulative sum gives a monotonically increasing coordinate per row/col
        y_embed = not_mask.cumsum(1, dtype=torch.float32)   # (B, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   # (B, H, W)
        # Normalise to [0, 2π]
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * 2.0 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * 2.0 * math.pi

        half = self.hidden_dim // 2
        dim_t = torch.arange(half, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2.0 * (dim_t // 2) / half)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, half)
        pos_y = y_embed[:, :, :, None] / dim_t

        # Interleave sin / cos
        pos_x = torch.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)   # (B, H, W, half)
        pos_y = torch.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)   # (B, H, W, half)

        pos = torch.cat([pos_y, pos_x], dim=3)  # (B, H, W, hidden_dim)
        return pos.flatten(1, 2)                 # (B, H*W, hidden_dim)


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Optimal bipartite matching of predictions to ground-truth objects.

    The matching cost is a weighted sum of three terms:

    * **Classification cost** – negative softmax probability assigned to the
      target class.
    * **L1 bounding-box cost** – ℓ¹ distance between predicted and target boxes
      in normalised *cxcywh* format.
    * **GIoU cost** – negative Generalised IoU between the matched boxes.

    Args:
        weight_class (float): Weight for the classification cost.  Default: 1.
        weight_bbox (float): Weight for the L1 bounding-box cost.  Default: 5.
        weight_giou (float): Weight for the GIoU cost.  Default: 2.
    """

    def __init__(
        self,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ):
        super().__init__()
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute the optimal matching for a batch of images.

        Args:
            pred_logits (torch.Tensor): Raw class logits of shape
                ``(B, num_queries, num_classes + 1)``.
            pred_boxes (torch.Tensor): Predicted boxes in normalised *cxcywh*
                format of shape ``(B, num_queries, 4)``.
            targets (list[dict]): One dict per image.  Each dict must contain:

                * ``'boxes'``      – ``(num_gt, 4)`` absolute *xyxy* coordinates.
                * ``'labels'``     – ``(num_gt,)`` integer class labels
                  (1-indexed; 0 is reserved for background).
                * ``'image_size'`` – ``(H, W)`` tuple of the input image
                  spatial dimensions.

        Returns:
            list[tuple[Tensor, Tensor]]: For each image a pair
            ``(src_idx, tgt_idx)`` of 1-D :class:`torch.long` tensors giving
            the matched query indices and the matched ground-truth indices
            respectively.
        """
        B = pred_logits.shape[0]
        indices = []

        for b in range(B):
            gt_boxes = targets[b]["boxes"]    # (num_gt, 4) absolute xyxy
            gt_labels = targets[b]["labels"]  # (num_gt,)  1-indexed
            img_h, img_w = targets[b]["image_size"]
            num_gt = len(gt_labels)

            if num_gt == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            # ---- Classification cost ----------------------------------------
            prob = pred_logits[b].softmax(-1)        # (Q, C+1)
            cost_class = -prob[:, gt_labels]          # (Q, num_gt)

            # ---- Normalise GT boxes to [0, 1] cxcywh -----------------------
            gt_boxes_norm = gt_boxes.clone().float()
            gt_boxes_norm[:, [0, 2]] = gt_boxes_norm[:, [0, 2]] / img_w
            gt_boxes_norm[:, [1, 3]] = gt_boxes_norm[:, [1, 3]] / img_h
            gt_cxcywh = box_convert(gt_boxes_norm, in_fmt="xyxy", out_fmt="cxcywh")

            # ---- L1 bounding-box cost ----------------------------------------
            cost_bbox = torch.cdist(pred_boxes[b], gt_cxcywh, p=1)  # (Q, num_gt)

            # ---- GIoU cost --------------------------------------------------
            pred_xyxy = box_convert(pred_boxes[b], in_fmt="cxcywh", out_fmt="xyxy")
            gt_xyxy = box_convert(gt_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
            cost_giou = -generalized_box_iou(pred_xyxy, gt_xyxy)     # (Q, num_gt)

            # ---- Solve assignment problem -----------------------------------
            C = (
                self.weight_class * cost_class
                + self.weight_bbox * cost_bbox
                + self.weight_giou * cost_giou
            ).cpu().numpy()

            row_idx, col_idx = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.long),
                torch.as_tensor(col_idx, dtype=torch.long),
            ))

        return indices


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------

class HungarianDetectionHead(nn.Module):
    """DETR-style detection head that uses Hungarian matching for supervision.

    The head accepts a single spatial feature map from a DINOv2 backbone,
    projects it to ``hidden_dim`` channels, adds 2-D sinusoidal positional
    encodings, and feeds the result as the *memory* of a standard
    :class:`torch.nn.TransformerDecoder`.  A set of ``num_queries`` learnable
    object queries act as the *target* sequence.  The decoded queries are
    independently passed through a classification head and a bounding-box
    regression MLP.

    **Training** – ground-truth objects are assigned to predicted queries via
    the :class:`HungarianMatcher`.  Three losses are computed:

    * Cross-entropy classification loss over all queries (matched queries carry
      their GT class; unmatched queries carry the background class 0).
    * L1 loss on the normalised *cxcywh* boxes of the matched pairs.
    * GIoU loss on the matched pairs.

    **Inference** – scores are derived from the softmax over foreground classes
    (background excluded).  Only predictions whose score exceeds
    ``score_threshold`` are returned.

    Args:
        in_channels (int): Number of channels in the input feature map produced
            by the backbone.
        num_classes (int): Number of foreground classes.  The head produces
            ``num_classes + 1`` class outputs (index 0 = background).
        hidden_dim (int): Transformer / embedding dimensionality.  Default: 256.
        num_queries (int): Number of learnable object queries.  Default: 100.
        nhead (int): Number of attention heads per transformer layer.
            Default: 8.
        num_decoder_layers (int): Depth of the transformer decoder.
            Default: 6.
        dim_feedforward (int): Feedforward dimension inside each transformer
            layer.  Default: 2048.
        dropout (float): Dropout probability in the transformer.  Default: 0.1.
        weight_class (float): Matcher cost weight for classification.
            Default: 1.
        weight_bbox (float): Matcher cost weight for L1 boxes.  Default: 5.
        weight_giou (float): Matcher cost weight for GIoU.  Default: 2.
        loss_weight_class (float): Coefficient for the classification loss term.
            Default: 1.
        loss_weight_bbox (float): Coefficient for the L1 box loss term.
            Default: 5.
        loss_weight_giou (float): Coefficient for the GIoU loss term.
            Default: 2.
        score_threshold (float): Minimum foreground score to keep a prediction
            during inference.  Default: 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        loss_weight_class: float = 1.0,
        loss_weight_bbox: float = 5.0,
        loss_weight_giou: float = 2.0,
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.score_threshold = score_threshold
        self.loss_weight_class = loss_weight_class
        self.loss_weight_bbox = loss_weight_bbox
        self.loss_weight_giou = loss_weight_giou

        # Project backbone features to hidden_dim
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Positional encoding for memory tokens
        self.pos_encoding = SinusoidalPositionEncoding2D(hidden_dim)

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = _build_mlp(hidden_dim, hidden_dim, 4, num_layers=3)

        # Matcher (used only during training)
        self.matcher = HungarianMatcher(
            weight_class=weight_class,
            weight_bbox=weight_bbox,
            weight_giou=weight_giou,
        )

    def forward(
        self,
        features: torch.Tensor,
        targets: list[dict] | None = None,
        image_sizes: list[tuple[int, int]] | None = None,
    ):
        """Run the detection head on a batch of feature maps.

        Args:
            features (torch.Tensor): Backbone feature map of shape
                ``(B, in_channels, H, W)``.
            targets (list[dict] | None): Ground-truth dicts (only needed during
                training).  Each dict must have ``'boxes'``, ``'labels'``, and
                ``'image_size'`` keys – see :class:`HungarianMatcher` for the
                expected formats.
            image_sizes (list[tuple[int, int]] | None): ``(H, W)`` of each
                original input image.  Required during inference to convert
                normalised predictions back to absolute pixel coordinates.

        Returns:
            dict | list[dict]:
                * **Training** – a dict with keys ``'loss_classification'``,
                  ``'loss_bbox'``, and ``'loss_giou'``.
                * **Inference** – a list of dicts each containing ``'boxes'``
                  (absolute *xyxy*), ``'labels'`` (1-indexed), and ``'scores'``.
        """
        B, _, H, W = features.shape

        # Project features and add positional encoding
        feat = self.input_proj(features)                             # (B, D, H, W)
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=features.device)
        pos = self.pos_encoding(mask)                                # (B, H*W, D)
        memory = feat.flatten(2).permute(0, 2, 1) + pos             # (B, H*W, D)

        # Expand learnable queries for the whole batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # Decode
        decoded = self.transformer_decoder(tgt=queries, memory=memory)     # (B, Q, D)

        # Predictions
        pred_logits = self.class_head(decoded)              # (B, Q, C+1)
        pred_boxes = self.box_head(decoded).sigmoid()       # (B, Q, 4) normalised cxcywh

        if self.training:
            return self._compute_loss(pred_logits, pred_boxes, targets)
        return self._post_process(pred_logits, pred_boxes, image_sizes)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        """Compute the set-based detection loss for a batch.

        Args:
            pred_logits: ``(B, Q, C+1)`` raw class logits.
            pred_boxes:  ``(B, Q, 4)`` normalised *cxcywh* predictions.
            targets: list of ground-truth dicts (see :meth:`forward`).

        Returns:
            dict with keys ``'loss_classification'``, ``'loss_bbox'``,
            ``'loss_giou'``.
        """
        B, Q, _ = pred_logits.shape
        device = pred_logits.device

        # Compute optimal matching
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # ---- Classification loss -------------------------------------------
        # Default: all queries predict background (class 0)
        target_labels = torch.zeros(B, Q, dtype=torch.long, device=device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_labels[b, src_idx] = targets[b]["labels"][tgt_idx].to(device)

        # Down-weight background to balance against foreground queries
        num_gt_total = max(1, sum(len(t["labels"]) for t in targets))
        bg_weight = num_gt_total / (B * Q)
        class_weights = torch.ones(self.num_classes + 1, device=device)
        class_weights[0] = bg_weight

        loss_cls = F.cross_entropy(
            pred_logits.reshape(B * Q, -1),
            target_labels.reshape(B * Q),
            weight=class_weights,
        )

        # ---- Box losses (matched pairs only) --------------------------------
        src_boxes_list: list[torch.Tensor] = []
        tgt_boxes_list: list[torch.Tensor] = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_boxes_list.append(pred_boxes[b][src_idx])

            img_h, img_w = targets[b]["image_size"]
            gt_boxes = targets[b]["boxes"][tgt_idx].float().to(device)
            gt_boxes_norm = gt_boxes.clone()
            gt_boxes_norm[:, [0, 2]] = gt_boxes_norm[:, [0, 2]] / img_w
            gt_boxes_norm[:, [1, 3]] = gt_boxes_norm[:, [1, 3]] / img_h
            tgt_boxes_list.append(
                box_convert(gt_boxes_norm, in_fmt="xyxy", out_fmt="cxcywh")
            )

        if src_boxes_list:
            src_boxes = torch.cat(src_boxes_list, dim=0)  # (M, 4)
            tgt_boxes = torch.cat(tgt_boxes_list, dim=0)  # (M, 4)
            num_matched = max(1, src_boxes.shape[0])

            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_matched

            src_xyxy = box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            tgt_xyxy = box_convert(tgt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            giou = generalized_box_iou(src_xyxy, tgt_xyxy)
            loss_giou = (1.0 - giou.diag()).sum() / num_matched
        else:
            # No matched pairs in this batch – return zero-gradient tensors
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        return {
            "loss_classification": self.loss_weight_class * loss_cls,
            "loss_bbox": self.loss_weight_bbox * loss_bbox,
            "loss_giou": self.loss_weight_giou * loss_giou,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _post_process(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        """Convert raw model output to detection results.

        Args:
            pred_logits: ``(B, Q, C+1)`` raw class logits.
            pred_boxes:  ``(B, Q, 4)`` normalised *cxcywh* predictions.
            image_sizes: ``(H, W)`` for each image in the batch.

        Returns:
            list of dicts each with keys ``'boxes'`` (absolute *xyxy*),
            ``'labels'`` (1-indexed), and ``'scores'``.
        """
        probs = pred_logits.softmax(-1)               # (B, Q, C+1)
        # Exclude background class (index 0) when computing scores
        fg_probs = probs[:, :, 1:]                    # (B, Q, C)
        scores, labels = fg_probs.max(-1)             # (B, Q)
        labels = labels + 1                           # restore 1-indexed class labels

        results = []
        for b in range(pred_logits.shape[0]):
            img_h, img_w = image_sizes[b]

            # Denormalise boxes from [0, 1] cxcywh → absolute xyxy
            boxes_abs = pred_boxes[b].clone()
            boxes_abs[:, [0, 2]] = boxes_abs[:, [0, 2]] * img_w
            boxes_abs[:, [1, 3]] = boxes_abs[:, [1, 3]] * img_h
            boxes_xyxy = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")

            keep = scores[b] >= self.score_threshold
            results.append({
                "boxes": boxes_xyxy[keep],
                "labels": labels[b][keep],
                "scores": scores[b][keep],
            })

        return results


# ---------------------------------------------------------------------------
# Complete model
# ---------------------------------------------------------------------------

class DINOv2HungarianDetectionModel(nn.Module):
    """Complete DINOv2 + :class:`HungarianDetectionHead` detection model.

    This ``nn.Module`` is the object stored in
    :attr:`DINOv2HungarianDetector.model`.  It wires together a
    :class:`~feature_extractors.DINOv2ViT` backbone (frozen by default) and
    a :class:`HungarianDetectionHead`.

    The model's ``forward`` method mirrors the interface expected by
    :class:`~detectors.AbstractDetector`:

    * **Training mode** – call with ``(images, targets)``; returns a loss dict.
    * **Eval mode** – call with ``(images,)``; returns a list of detection
      dicts (``'boxes'``, ``'labels'``, ``'scores'``).

    .. note::
        All images in a batch must have the *same* spatial dimensions because
        the DINOv2 ViT backbone maps each image to a fixed-size patch grid.
        Use a resize transform to ensure consistent dimensions during training
        and inference.

    Args:
        backbone (DINOv2ViT): Feature extractor.
        head (HungarianDetectionHead): Detection head.
    """

    def __init__(self, backbone: DINOv2ViT, head: HungarianDetectionHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        targets: list[dict] | None = None,
    ):
        """Run a full forward pass.

        Args:
            images: Either a list of ``(C, H, W)`` tensors (all the same size)
                or a pre-stacked ``(B, C, H, W)`` tensor.
            targets (list[dict] | None): Ground-truth dicts required during
                training.  Each dict should contain ``'boxes'`` and
                ``'labels'``; ``'image_size'`` is filled in automatically.

        Returns:
            dict | list[dict]: Loss dict during training; list of detection
            dicts during inference.
        """
        if isinstance(images, (list, tuple)):
            image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
            x = torch.stack(images, dim=0)
        else:
            image_sizes = [(images.shape[-2], images.shape[-1])] * images.shape[0]
            x = images

        # Attach image sizes to targets so the matcher can normalise GT boxes
        if targets is not None:
            for i, t in enumerate(targets):
                if "image_size" not in t:
                    t["image_size"] = image_sizes[i]

        features = self.backbone(x)   # (B, D, H_p, W_p)
        return self.head(features, targets=targets, image_sizes=image_sizes)
