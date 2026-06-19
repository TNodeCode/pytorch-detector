import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import Dinov2Config, Dinov2Model, ConvNextConfig, ConvNextModel
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class DINOv2ViT(nn.Module):
    """DINOv2 Vision Transformer feature extractor.

    Extracts features from intermediate transformer layers and reshapes them
    from (B, N, D) to spatial feature maps (B, D, H, W).

    Args:
        model_name (str | None): HuggingFace model identifier to load
            pretrained weights (e.g. ``'facebook/dinov2-base'``).  When
            ``None`` a randomly-initialised model using the default
            :class:`~transformers.Dinov2Config` is created.
        finetuning (bool): If True the backbone weights are trainable.
        output_patches (bool): If True include the raw patch embeddings as an
            additional output level.
        layers (list[int]): Indices of transformer encoder layers whose output
            to capture.  Default: [2, 5, 8, 11].
        layer_norm (bool): Apply a learned LayerNorm to each captured feature
            before reshaping.
    """

    def __init__(
        self,
        model_name: str = None,
        finetuning: bool = False,
        output_patches: bool = False,
        layers: list = None,
        layer_norm: bool = True,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 5, 8, 11]
        self.finetuning = finetuning
        self.layers = layers
        self.output_patches = output_patches

        if model_name is not None:
            self.model = Dinov2Model.from_pretrained(model_name)
            self.config = self.model.config
        else:
            self.config = Dinov2Config()
            self.model = Dinov2Model(self.config)

        self.layer_norm = layer_norm and (len(layers) + int(output_patches)) > 0
        if self.layer_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(self.config.hidden_size, eps=1e-5, elementwise_affine=True)
                for _ in range(len(layers) + int(output_patches))
            ])
        if not self.finetuning:
            self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Forward image through DINOv2 ViT.

        Args:
            x (torch.Tensor): Image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor or tuple[torch.Tensor]: Spatial feature map(s) of
                shape (B, D, H/patch, W/patch).  A tuple is returned when
                more than one layer is selected.
        """
        captured = OrderedDict()
        hooks = []

        if self.output_patches:
            # key= default-argument captures current string at definition time
            hooks.append(
                self.model.embeddings.register_forward_hook(
                    lambda m, i, o, key="embeddings": captured.update({key: o})
                )
            )
        for layer_idx in self.layers:
            # key=layer_idx default-argument captures the current int at definition time
            hooks.append(
                self.model.encoder.layer[layer_idx].register_forward_hook(
                    lambda m, i, o, key=layer_idx: captured.update({key: o})
                )
            )

        z = self.model(x)

        for h in hooks:
            h.remove()

        if not captured:
            captured[len(self.model.encoder.layer) - 1] = z.last_hidden_state

        feature_maps = OrderedDict()
        for idx, (k, feat) in enumerate(captured.items()):
            if self.layer_norm:
                feat = self.norms[idx](feat)
            # Remove the [CLS] token
            feat = feat[:, 1:, :]
            B, P, D = feat.shape
            h = w = int(P ** 0.5)
            feat = feat.permute(0, 2, 1).reshape(B, D, h, w)
            feature_maps[k] = feat

        if len(feature_maps) > 1:
            return tuple(feature_maps[k] for k in feature_maps)
        return feature_maps[next(iter(feature_maps))]


class DINOv2ConvNext(nn.Module):
    """ConvNext feature extractor for use in detection pipelines.

    Extracts hierarchical feature maps from a ConvNext backbone.  Each
    selected stage outputs a spatial map at a different resolution, making
    this extractor naturally suitable for FPN-based detectors.

    Args:
        model_name (str | None): HuggingFace model identifier to load
            pretrained weights (e.g. ``'facebook/convnext-base-224'``).  When
            ``None`` a randomly-initialised model using the default
            :class:`~transformers.ConvNextConfig` is created.
        finetuning (bool): If True the backbone weights are trainable.
        layers (list[int]): Indices of ConvNext stages to capture.
            Default: [0, 1, 2, 3].
    """

    def __init__(
        self,
        model_name: str = None,
        finetuning: bool = False,
        layers: list = None,
    ):
        super().__init__()
        if layers is None:
            layers = [0, 1, 2, 3]
        self.finetuning = finetuning
        self.layers = layers

        if model_name is not None:
            self.model = ConvNextModel.from_pretrained(model_name)
            self.config = self.model.config
        else:
            self.config = ConvNextConfig()
            self.model = ConvNextModel(self.config)

        if not self.finetuning:
            self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Forward image through ConvNext.

        Args:
            x (torch.Tensor): Image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor or tuple[torch.Tensor]: Feature map(s) from the
                selected stages.  A tuple is returned when more than one
                stage is selected.
        """
        captured = OrderedDict()
        hooks = []

        for i in self.layers:
            # key=i default-argument captures the current int at definition time
            hooks.append(
                self.model.encoder.stages[i].register_forward_hook(
                    lambda m, inp, out, key=i: captured.update({key: out})
                )
            )

        z = self.model(x)

        for h in hooks:
            h.remove()

        if not captured:
            captured[0] = z.last_hidden_state

        if len(captured) > 1:
            return tuple(captured[k] for k in captured)
        return captured[next(iter(captured))]


class DINOv2ViTBackbone(nn.Module):
    """DINOv2 ViT backbone with multi-scale projection for detection.

    Wraps :class:`DINOv2ViT` (single-scale ViT output) into a five-level
    feature pyramid compatible with torchvision detection heads such as
    :class:`~torchvision.models.detection.RetinaNet`.

    The ViT last-layer features are first projected to ``out_channels`` via a
    1×1 convolution, then downsampled four times with 3×3 strided convolutions
    to produce five scales.

    Args:
        model_name (str | None): Passed to :class:`DINOv2ViT`.  Provide a
            HuggingFace identifier (e.g. ``'facebook/dinov2-base'``) to load
            pretrained weights.
        out_channels (int): Number of channels in every output feature level.
        finetuning (bool): Passed to :class:`DINOv2ViT`.
    """

    def __init__(
        self,
        model_name: str = None,
        out_channels: int = 256,
        finetuning: bool = False,
    ):
        super().__init__()
        self.body = DINOv2ViT(model_name=model_name, finetuning=finetuning, layers=[], layer_norm=False)
        hidden = self.body.config.hidden_size

        self.proj = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.down1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feat = self.body(x)
        p0 = self.proj(feat)
        p1 = self.down1(p0)
        p2 = self.down2(p1)
        p3 = self.down3(p2)
        p4 = self.down4(p3)
        return OrderedDict([("0", p0), ("1", p1), ("2", p2), ("3", p3), ("4", p4)])


class DINOv2ConvNextBackbone(nn.Module):
    """ConvNext backbone with FPN for detection.

    Wraps :class:`DINOv2ConvNext` (all four stages) and applies a
    :class:`~torchvision.ops.FeaturePyramidNetwork` to produce a uniform
    ``out_channels``-wide five-level feature pyramid (four FPN levels plus one
    max-pool level).

    Args:
        model_name (str | None): Passed to :class:`DINOv2ConvNext`.  Provide a
            HuggingFace identifier (e.g. ``'facebook/convnext-base-224'``) to
            load pretrained weights.
        out_channels (int): Number of channels in every FPN output level.
        finetuning (bool): Passed to :class:`DINOv2ConvNext`.
    """

    def __init__(
        self,
        model_name: str = None,
        out_channels: int = 256,
        finetuning: bool = False,
    ):
        super().__init__()
        self.body = DINOv2ConvNext(model_name=model_name, finetuning=finetuning, layers=[0, 1, 2, 3])
        in_channels_list = list(self.body.config.hidden_sizes)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        stage_feats = self.body(x)
        # Map each feature to its actual stage index so the FPN in_channels
        # list aligns correctly with the captured feature maps.
        feat_dict = OrderedDict(
            (str(stage_idx), f)
            for stage_idx, f in zip(self.body.layers, stage_feats)
        )
        return self.fpn(feat_dict)

