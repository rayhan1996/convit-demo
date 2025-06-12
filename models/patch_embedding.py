import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Turn a 4‑channel image into a sequence of flattened patches, then
    project each patch to the Transformer embed_dim.

    Args
    ----
    in_channels : int
        Number of input channels (4 for the multispectral TIFFs).
    embed_dim   : int
        Dimension of the token/patch embedding (ViT default = 768).
    patch_size  : int
        Size of each square patch (e.g. 16 → 16×16 pixels).

    Output
    ------
    Tensor of shape (batch_size, num_patches, embed_dim)
        num_patches = (H // patch_size) * (W // patch_size)
    """
    def __init__(self, in_channels: int = 4,
                 embed_dim:   int = 768,
                 patch_size:  int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

        # A conv‑projection with kernel == stride == patch_size
        # splits the image into non‑overlapping patches and
        # linearly projects each patch to embed_dim.
        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 4, H, W)

        Returns
        -------
        Tensor, shape (B, num_patches, embed_dim)
        """
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, N)
        x = x.transpose(1, 2)      # (B, N, embed_dim)
        return x
