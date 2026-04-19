"""
src/models/sam_decoder.py — SAM-based models for attempts #4 and #5.

Both models freeze the SAM ViT-B image encoder and train only a lightweight
decoder on top. This is the key idea for few-shot segmentation: the encoder
already contains a massive visual prior from SAM's 11M-image pretraining,
so even with only 24 training patches the model has strong features to work
with. Only the small decoder needs to be learned from scratch.

Attempt #4: SAMFrozenConvDecoder
    Ported from an earlier reference implementation. A simple projector (256 -> 128 channels) followed by 4 ConvTranspose2d
    upsampling stages, with Dropout2d on the first two. No skip connections.
    The decoder sees only the encoder's final feature map (64x64 for 1024
    input), so it has to reconstruct fine spatial detail from coarse features.

Attempt #5: SAMUNetDecoder
    A richer decoder that pulls intermediate features from ViT blocks 3, 6,
    9, 12 via forward hooks and uses them as U-Net-style skip connections.
    This gives the decoder access to multi-scale features (fine edges from
    early blocks, semantic understanding from late blocks), similar to how
    a U-Net decoder uses skip connections from the encoder's downsampling
    stages.

    The key challenge is reshaping the ViT features: each block outputs
    (B, N, C) where N = (H/16)^2 = 64*64 = 4096 tokens. We reshape this
    to (B, C, 64, 64) spatial feature maps. Since all ViT blocks operate
    at the same spatial resolution (1/16), the skip connections are all
    64x64 — unlike a CNN U-Net where each level is a different resolution.
    We project each skip to a smaller channel count and concatenate with
    the upsampled features at the appropriate decoder stage.

Input:  (B, 3, 1024, 1024) float32 RGB
Output: (B, 1, 1024, 1024) float32 logits (pre-sigmoid)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry


# Default checkpoint path — tries common locations in order.
# On Colab, download to the repo root; locally, place in the project directory.
_CANDIDATES = [
    Path("sam_vit_b_01ec64.pth"),                  # repo root / CWD
    Path(__file__).parent.parent.parent / "sam_vit_b_01ec64.pth",  # project root
]
DEFAULT_CHECKPOINT = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])


def _load_frozen_encoder(
    model_type: str = "vit_b",
    checkpoint: Optional[Path] = None,
) -> nn.Module:
    """Load the SAM image encoder with all parameters frozen.

    Args:
        model_type: SAM model variant ('vit_b', 'vit_l', 'vit_h').
        checkpoint: path to the pretrained checkpoint .pth file.

    Returns:
        The encoder module with requires_grad=False on all parameters.
    """
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            f"Download from https://github.com/facebookresearch/segment-anything"
        )
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    encoder = sam.image_encoder

    # Freeze every parameter in the encoder. This is the fundamental
    # assumption of attempts #4 and #5: the encoder is a fixed feature
    # extractor, only the decoder learns.
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


# ============================================================================
#  ATTEMPT #4: FROZEN ENCODER + SIMPLE CONV DECODER
# ============================================================================

class SAMFrozenConvDecoder(nn.Module):
    """SAM ViT-B encoder (frozen) + simple convolutional decoder.

    Ported from the reference code's SAMFrozenBinarySegmentation class.
    The decoder is a straightforward stack of ConvTranspose2d layers that
    upsample the encoder's 64x64 feature map back to 1024x1024.

    Architecture:
        encoder output: (B, 256, 64, 64)
        projector:      (B, 256, 64, 64) -> (B, 128, 64, 64)
        upsample:       (B, 128, 64, 64) -> (B, 128, 128, 128) -> bilinear
        stage 1:        (B, 128, 128, 128) -> (B, 128, 256, 256) + Dropout
        stage 2:        (B, 128, 256, 256) -> (B, 64, 512, 512) + Dropout
        stage 3:        (B, 64, 512, 512) -> (B, 32, 1024, 1024)
        head:           (B, 32, 1024, 1024) -> (B, 1, 1024, 1024)
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: Optional[Path] = None,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.encoder = _load_frozen_encoder(model_type, checkpoint)

        # 1x1 projection to reduce channel count before the decoder.
        self.projector = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder stages: each ConvTranspose2d doubles spatial resolution.
        # Dropout2d on the first two stages for regularization — these are
        # the highest-capacity layers and most prone to overfitting.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final 1x1 conv to produce single-channel logits.
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: (B, 3, 1024, 1024) -> (B, 256, 64, 64)
        with torch.no_grad():
            feats = self.encoder(x)

        # Projector: (B, 256, 64, 64) -> (B, 128, 64, 64)
        x = self.projector(feats)

        # Upsample 64 -> 128 with bilinear interpolation (the encoder
        # output is 64x64 and our first ConvTranspose expects 128x128).
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Decoder: 128 -> 256 -> 512 -> 1024
        x = self.decoder(x)

        # Final size adjustment — if the decoder output isn't exactly 1024
        # due to rounding in ConvTranspose2d, interpolate to match.
        if x.shape[-2:] != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

        return x


# ============================================================================
#  ATTEMPT #5: FROZEN ENCODER + U-NET DECODER WITH VIT SKIP CONNECTIONS
# ============================================================================

class _ViTFeatureHook:
    """Forward hook that captures a ViT block's output as a spatial feature map.

    SAM's ViT blocks output (B, H, W, C) where H = W = 64 for 1024x1024
    input with patch_size=16. This is different from standard ViT which
    outputs (B, N, C) — SAM keeps the spatial layout throughout. We simply
    permute to (B, C, H, W) for use as a spatial skip connection in the
    U-Net-style decoder.
    """

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        # SAM blocks output (B, H, W, C) — permute to (B, C, H, W)
        self.features = output.permute(0, 3, 1, 2)


class SAMUNetDecoder(nn.Module):
    """SAM ViT-B encoder (frozen) + U-Net-style decoder with skip connections.

    This model pulls intermediate features from ViT blocks at indices
    [3, 6, 9, 12] (1-indexed: blocks 3, 6, 9, and the final output) via
    forward hooks. These features serve as skip connections similar to a
    U-Net encoder's multi-scale outputs.

    Unlike a CNN U-Net where each encoder level has a different spatial
    resolution, ALL ViT block outputs have the same resolution (64x64).
    So the "skip connections" here are not multi-scale — they're
    multi-depth. Early blocks have fine-grained local features (edges,
    textures), late blocks have coarse semantic features (object-level).

    The decoder progressively upsamples from 64x64 to 1024x1024 (4 stages,
    each 2x), concatenating a skip connection at each stage.
    """

    # Which ViT blocks to tap for skip connections (0-indexed).
    # ViT-B has 12 blocks total. We pick 4 evenly spaced blocks
    # to get a spread from early (local) to late (semantic) features.
    SKIP_BLOCK_INDICES = [2, 5, 8, 11]  # 0-indexed: blocks 3, 6, 9, 12

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: Optional[Path] = None,
        skip_channels: int = 64,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.encoder = _load_frozen_encoder(model_type, checkpoint)

        # Register forward hooks on the selected ViT blocks to capture
        # intermediate features without modifying the encoder code.
        self.hooks = []
        for idx in self.SKIP_BLOCK_INDICES:
            hook = _ViTFeatureHook()
            self.encoder.blocks[idx].register_forward_hook(hook)
            self.hooks.append(hook)

        # The ViT-B embedding dimension is 768 for all blocks.
        vit_embed_dim = 768

        # 1x1 projections to reduce each skip's channel count from 768 to
        # skip_channels. Without this, concatenation would create huge
        # tensors (768 + decoder channels per stage).
        self.skip_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_embed_dim, skip_channels, 1),
                nn.BatchNorm2d(skip_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.SKIP_BLOCK_INDICES
        ])

        # The encoder's final output is 256 channels (after the neck).
        # Project to the decoder's working width.
        decoder_width = 128
        self.stem = nn.Sequential(
            nn.Conv2d(256, decoder_width, 1),
            nn.BatchNorm2d(decoder_width),
            nn.ReLU(inplace=True),
        )

        # Decoder stages: 4 stages, each upsamples 2x.
        # At each stage, the input is (upsampled features + skip connection).
        # Input channels = decoder_width_at_stage + skip_channels (from concat).
        # We process the deepest skip first (block 12) and work outward.
        stages = []
        in_ch = decoder_width
        out_channels = [128, 64, 32, 16]
        for i, out_ch in enumerate(out_channels):
            stages.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch + skip_channels, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                *([] if i >= 2 else [nn.Dropout2d(dropout_p)]),
            ))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        # Final 1x1 conv: 16 channels -> 1 logit.
        self.head = nn.Conv2d(out_channels[-1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward (frozen). The hooks capture intermediate features.
        with torch.no_grad():
            encoder_out = self.encoder(x)  # (B, 256, 64, 64)

        # Project skip connections from 768 to skip_channels.
        # Reversed so that the deepest block's features come first
        # (matching the decoder's coarse-to-fine direction).
        skips = [
            proj(hook.features)
            for proj, hook in zip(
                reversed(self.skip_projections),
                reversed(self.hooks),
            )
        ]

        # Stem: project encoder output to decoder width.
        x = self.stem(encoder_out)  # (B, 128, 64, 64)

        # Decoder stages: upsample + concat skip + conv.
        for stage, skip in zip(self.stages, skips):
            # All skips are 64x64 (ViT is single-resolution), but x grows
            # with each upsample. We need to upsample the skip to match x.
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = stage(x)

        x = self.head(x)

        # Final size adjustment if needed.
        if x.shape[-2:] != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

        return x
