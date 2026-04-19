"""
src/models/smp_wrapper.py — SMP models with decoder-only dropout injection.

Attempts #2 and #3: SMP U-Net and DeepLabV3+ with ImageNet-pretrained
ResNet encoders and Dropout2d injected into the decoder for regularization.

Why decoder-only dropout:
  With only 24 training patches, the decoder's randomly-initialized weights
  are where overfitting actually happens — the pretrained encoder already has
  strong, well-generalized features from ImageNet. Adding dropout to the
  encoder would disrupt its BatchNorm statistics and learned representations.
  Dropout in the decoder regularizes the small number of newly-trained
  parameters without touching the pretrained encoder.

  This matches the pattern used in the SAM decoder (Attempt #4), which
  applies Dropout2d(0.3) to the first two ConvTranspose stages.

The dropout is injected via forward hooks on the decoder's internal blocks,
which is less fragile than subclassing SMP's decoder (the internal block
structure can change between SMP versions, but the blocks attribute is
part of the public API).

Input:  (B, 3, 1024, 1024) float32 RGB
Output: (B, 1, 1024, 1024) float32 logits (pre-sigmoid)
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


def _inject_decoder_dropout(model: nn.Module, dropout_p: float) -> None:
    """Add Dropout2d after each decoder block via forward hooks.

    This walks `model.decoder.blocks` and registers a forward hook on each
    block that applies Dropout2d to the block's output. The dropout module
    is stored as an attribute on the block so that model.train()/model.eval()
    correctly switches it between training and inference modes.

    Args:
        model: an SMP model with a `.decoder.blocks` attribute.
        dropout_p: dropout probability (e.g. 0.3). If 0, no hooks are added.
    """
    if dropout_p <= 0:
        return

    if not hasattr(model.decoder, "blocks"):
        raise AttributeError(
            "model.decoder has no 'blocks' attribute — cannot inject dropout. "
            "This might mean the SMP version changed its decoder internals."
        )

    for i, block in enumerate(model.decoder.blocks):
        dropout = nn.Dropout2d(p=dropout_p)
        # Store the dropout module as an attribute on the block so it's
        # properly registered as a sub-module — this ensures that
        # model.train() / model.eval() toggles it correctly, and that
        # model.parameters() can see it if needed.
        block.add_module(f"injected_dropout_{i}", dropout)

        # The hook applies dropout AFTER the block's own forward pass.
        def make_hook(drop):
            def hook_fn(module, input, output):
                return drop(output)
            return hook_fn

        block.register_forward_hook(make_hook(dropout))


def make_smp_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    decoder_dropout: float = 0.3,
) -> nn.Module:
    """Create an SMP U-Net with ImageNet pretrained encoder and decoder dropout.

    This is attempt #2: the standard transfer-learning recipe. The encoder
    starts frozen (see train.py's two-stage logic), the decoder is randomly
    initialized and regularized with dropout.

    Args:
        encoder_name: backbone architecture (default 'resnet34').
        encoder_weights: pretrained weights (default 'imagenet').
        decoder_dropout: dropout probability for injected Dropout2d layers.
            Set to 0 to disable.

    Returns:
        An smp.Unet with classes=1, no activation, and decoder dropout.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    _inject_decoder_dropout(model, decoder_dropout)
    return model


def make_smp_deeplabv3plus(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    decoder_dropout: float = 0.3,
) -> nn.Module:
    """Create an SMP DeepLabV3+ with pretrained encoder and decoder dropout.

    This is attempt #3: same encoder and recipe as #2, but with a different
    decoder architecture. DeepLabV3+ uses atrous spatial pyramid pooling
    (ASPP) instead of the U-Net skip-connection decoder, which captures
    multi-scale context differently.

    Args:
        encoder_name: backbone architecture (default 'resnet34').
        encoder_weights: pretrained weights (default 'imagenet').
        decoder_dropout: dropout probability for injected Dropout2d layers.
            Set to 0 to disable. Note: DeepLabV3+ may have fewer decoder
            blocks than U-Net, so fewer dropout layers are injected.

    Returns:
        An smp.DeepLabV3Plus with classes=1, no activation, and decoder dropout.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    # DeepLabV3+ decoder structure differs from U-Net — it may not have
    # a `.blocks` attribute. In that case, we skip dropout injection and
    # rely on other regularization (augmentation, weight decay).
    if hasattr(model.decoder, "blocks"):
        _inject_decoder_dropout(model, decoder_dropout)
    return model
