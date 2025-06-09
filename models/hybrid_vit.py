import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

from models.efficientnet4ch import EfficientNet4ch
from models.patch_embedding import PatchEmbedding


# ───────────────────── CNN‑to‑token adapter ─────────────────────
class CBlock(nn.Module):
    """Depth‑wise 3×3 conv  → BN → GELU → 1×1 projection to *embed_dim*."""
    def __init__(self, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.bn  = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()
        self.pw  = nn.Conv2d(in_ch, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (B, C, H, W)
        x = self.act(self.bn(self.dw(x)))
        return self.pw(x)                                     # (B, embed_dim, H, W)


# ───────────────────── Hybrid model (P2FE‑ViT) ─────────────────────
class P2FE_ViT(nn.Module):
    """
    Plug‑and‑Play Feature‑Enhanced ViT:
    EfficientNet‑B0 (4‑channel) + ViT‑B/16 encoder.
    """
    def __init__(
        self,
        patch_size : int = 16,
        embed_dim  : int = 768,
        num_classes: int = 11,
    ) -> None:
        super().__init__()

        # 1) CNN branch (feature extractor)
        self.cnn = EfficientNet4ch(num_classes=num_classes, pretrained=True)
        cnn_out_ch = 1280                                           # for EfficientNet‑B0

        # 2) CNN → token adapter
        self.cblock = CBlock(cnn_out_ch, embed_dim)

        # 3) Patch embedding for the raw 4‑channel image
        self.patch = PatchEmbedding(
            in_channels=4, embed_dim=embed_dim, patch_size=patch_size
        )  # 224×224 → 14×14 patches (196 tokens)

        # 4) Vision Transformer backbone
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.vit.heads = nn.Linear(embed_dim, num_classes)          # replace classifier

        # 4‑a) adjust positional embedding from 577 to 197 tokens
        with torch.no_grad():
            old_pos   = self.vit.encoder.pos_embedding              # (1,577,768)
            cls_pos   = old_pos[:, :1]                              # (1,1 ,768)
            patch_pos = old_pos[:, 1:]                              # (1,576,768)
            patch_pos = (
                patch_pos.view(1, 24, 24, embed_dim)
                .permute(0, 3, 1, 2)                                # (1,768,24,24)
            )
            patch_pos = nn.functional.interpolate(
                patch_pos, size=(14, 14), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, 196, embed_dim)
            new_pos   = torch.cat([cls_pos, patch_pos], dim=1)      # (1,197,768)
        self.vit.encoder.pos_embedding = nn.Parameter(new_pos, requires_grad=True)

    # ───────────────────── forward ─────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:             # (B, 4, 224, 224)
        # CNN features
        feats   = self.cnn.model.features(x)                        # (B,1280,7,7)
        cls_map = self.cblock(feats)                                # (B,768 ,7,7)
        cls_vec = cls_map.mean(dim=(2, 3))                          # (B,768) – global avg pooling

        # Patch tokens from raw image
        patch_tok = self.patch(x)                                   # (B,196,768)

        # CNN‑based positional bias for each patch token
        pos_map = nn.functional.interpolate(cls_map, (14, 14),
                                            mode="bilinear", align_corners=False)
        pos_tok = pos_map.flatten(2).transpose(1, 2)                # (B,196,768)

        # Assemble full token sequence (cls + patches) and add bias
        cls_tok = cls_vec.unsqueeze(1)                              # (B,1,768)
        tokens  = torch.cat([cls_tok, patch_tok], dim=1)            # (B,197,768)
        tokens += torch.cat([torch.zeros_like(cls_tok), pos_tok], dim=1)

        # ViT encoder and classifier head
        h   = self.vit.encoder(tokens)                              # (B,197,768)
        out = self.vit.heads(h[:, 0])                               # logits (B,num_classes)
        return out
