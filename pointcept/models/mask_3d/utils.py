import torch

def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(0, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(0, dtype=torch.int32)
    )
    return intersections / (unions + 1e-15)