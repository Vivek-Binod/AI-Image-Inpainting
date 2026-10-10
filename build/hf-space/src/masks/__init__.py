from src.masks.generator import (MASK_TYPES, MaskConfig, apply_mask, center_square_mask,
                                 free_form_mask, generate_mask, rect_mask)


def mask_config_from_dict(d: dict | None) -> MaskConfig:
    """Build a MaskConfig from a (possibly partial) config dictionary."""
    return MaskConfig(**(d or {}))


__all__ = ["MASK_TYPES", "MaskConfig", "apply_mask", "center_square_mask", "free_form_mask",
           "generate_mask", "rect_mask", "mask_config_from_dict"]
