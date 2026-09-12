# -*- coding: utf-8 -*-
import torch

def random_masking(ratio: float, strategy):
    """Generate masks for reconstruction based on specified strategy."""

    def random_mask(x, ratio=ratio):
        """Randomly mask 2D content across [C, N] plane."""
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(C, N, device=device)

        num_elements = C * N
        num_keep = int(num_elements * (1 - ratio))

        ids_shuffle = torch.argsort(noise.reshape(-1))

        mask = torch.ones([C, N], device=device)
        mask_flat = mask.reshape(-1)
        mask_flat[ids_shuffle[:num_keep]] = 0
        mask = mask_flat.reshape(C, N)

        return mask

    def patch_mask(x, ratio=ratio):
        """Mask all channels of selected patches."""
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(N, device=device)

        num_elements = N
        num_keep = int(num_elements * (1 - ratio))

        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep]

        mask = torch.ones([N], device=device)
        mask[ids_keep] = 0

        mask = mask.unsqueeze(0).expand(C, -1)  # [C, N]

        return mask

    def channel_mask(x, ratio=ratio):
        """Mask all patches of selected channels."""
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(C, device=device)

        if isinstance(ratio, int):
            # Use as number of channels to mask
            num_keep = C - ratio
        else:
            # Use as ratio
            num_keep = int(C * (1 - ratio))

        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep]

        mask = torch.ones([C], device=device)
        mask[ids_keep] = 0

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    def he_mask(x, ratio=ratio):
        """Mask H&E channels (last 3 channels)."""
        B, C, N, D = x.shape
        device = x.device
        mask = torch.ones([C], device=device)
        mask[:-3] = 0

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    def mif_mask(x, ratio=ratio):
        """Mask MIF channels (all except last 3 channels)."""
        B, C, N, D = x.shape
        device = x.device
        mask = torch.ones([C], device=device)
        mask[-3:] = 0

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    def specified_mask(x, channels):
        """Mask specified channels."""
        assert isinstance(channels, list), "channels must be a list"
        B, C, N, D = x.shape
        device = x.device
        mask = torch.zeros([C], device=device)
        mask[channels] = 1

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    strategies = {
        "random": random_mask,
        "patch": patch_mask,
        "channel": channel_mask,
        "he": he_mask,
        "mif": mif_mask,
        "specified": specified_mask,
    }

    return strategies[strategy]
