import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

from models.efficientnet4ch import EfficientNet4ch
from models.patch_embedding import PatchEmbedding

# ───────────────────────────────────────────────────────────────────────────────
# CBlock: depth-wise 3×3 conv  → BN → GELU → 1×1 projection to embed_dim
# ───────────────────────────────────────────────────────────────────────────────
class CBlock(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.bn  = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()
        self.pw  = nn.Conv2d(in_ch, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        x = self.act(self.bn(self.dw(x)))
        return self.pw(x)                                # (B, embed_dim, H, W)

# ───────────────────────────────────────────────────────────────────────────────
# ConViT: EfficientNet-B0 (4-ch)  +  ViT-B/16  with CNN-ViT fusion at head
# ───────────────────────────────────────────────────────────────────────────────
class ConViT(nn.Module):
    def __init__(
        self,
        patch_size : int = 16,
        embed_dim  : int = 768,
        num_classes: int = 3,     # 3 labels: Cloud / Shadow / LandCover
    ) -> None:
        super().__init__()

        # 1) CNN branch
        self.cnn = EfficientNet4ch(num_classes=num_classes, pretrained=True)
        cnn_out_ch = 1280  # EfficientNet-B0 last feature channels
        self.cnn_proj = nn.Linear(cnn_out_ch, embed_dim)  # 1280 → 768

        # 2) CBlock (CNN → token map)
        self.cblock = CBlock(cnn_out_ch, embed_dim)

        # 3) Patch embedding for raw 4-channel image
        self.patch = PatchEmbedding(
            in_channels=4, embed_dim=embed_dim, patch_size=patch_size
        )  # 224×224 → 14×14 → 196 tokens

        # 4) Vision Transformer backbone (ViT-B/16)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.vit.heads = nn.Identity()  # disable original classifier

        # Adjust ViT positional embedding (577 → 197 tokens)
        with torch.no_grad():
            old_pos   = self.vit.encoder.pos_embedding          # (1,577,768)
            cls_pos   = old_pos[:, :1]                          # (1,1,768)
            patch_pos = old_pos[:, 1:].view(1, 24, 24, embed_dim).permute(0,3,1,2)
            patch_pos = nn.functional.interpolate(
                patch_pos, size=(14,14), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0,2,3,1).reshape(1, 196, embed_dim)
            new_pos   = torch.cat([cls_pos, patch_pos], dim=1)  # (1,197,768)
        self.vit.encoder.pos_embedding = nn.Parameter(new_pos, requires_grad=True)

        # 5) Fusion classifier (ViT cls + CNN vec)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )

    # ───────────────────────────────── forward ────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 4, 224, 224)
        # CNN features (B,1280,7,7)
        feats   = self.cnn.model.features(x)

        # CBlock map for positional bias (B,768,7,7)
        cls_map = self.cblock(feats)
        cls_vec = cls_map.mean(dim=(2, 3))             # global avg → (B,768)

        # CNN global vector for fusion (B,1280) → project → (B,768)
        cnn_vec = feats.mean(dim=(2, 3))
        cnn_tok = self.cnn_proj(cnn_vec)

        # Patch tokens from raw image (B,196,768)
        patch_tok = self.patch(x)

        # CNN-based positional bias for each patch (B,196,768)
        pos_map = nn.functional.interpolate(cls_map, (14,14), mode="bilinear",
                                            align_corners=False)
        pos_tok = pos_map.flatten(2).transpose(1, 2)

        # Assemble full token sequence (1 cls + 196 patches) and add bias
        cls_tok = cls_vec.unsqueeze(1)                       # (B,1,768)
        tokens  = torch.cat([cls_tok, patch_tok], dim=1)     # (B,197,768)
        tokens += torch.cat([torch.zeros_like(cls_tok), pos_tok], dim=1)

        # ViT encoder
        h       = self.vit.encoder(tokens)                   # (B,197,768)
        vit_cls = h[:, 0]                                    # (B,768)

        # Concatenate ViT cls with CNN vector and classify
        fusion  = torch.cat([vit_cls, cnn_tok], dim=1)       # (B,1536)
        out     = self.classifier(fusion)                    # (B,num_classes)
        return out
