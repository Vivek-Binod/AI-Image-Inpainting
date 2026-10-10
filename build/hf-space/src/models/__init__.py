from src.models.unet import InpaintUNet, composite, count_parameters

MODEL_NAMES = {"unet": "conv", "gated_unet": "gated"}


def build_model(name: str, base_channels: int = 32):
    """Build a model by name: 'unet' (ordinary convolutions) or 'gated_unet' (gated convolutions)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Expected one of {list(MODEL_NAMES)}.")
    return InpaintUNet(block=MODEL_NAMES[name], base_channels=base_channels)


__all__ = ["InpaintUNet", "build_model", "composite", "count_parameters", "MODEL_NAMES"]
