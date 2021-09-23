import torch


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha: int = 4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames

        # Perform temporal sampling from the fast pathway.
        temporal_idx = 1 if frames.ndim == 4 else 2
        slow_pathway = torch.index_select(
            frames,
            temporal_idx,
            torch.linspace(
                0, frames.shape[temporal_idx] - 1, frames.shape[temporal_idx] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def spatio_temporal_collate_fn(batch):
    pass
