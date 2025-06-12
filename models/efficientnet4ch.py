
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNet4ch(nn.Module):
    """
    EfficientNet‑B0 adapted to 4‑channel input and multi‑label output.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()

        # 1) Load backbone
        if pretrained:
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.model = models.efficientnet_b0(weights=None)

        # 2) Replace first conv layer (3 → 4 channels)
        # original Conv2d(3, 32, kernel=3, stride=2, padding=1)

        old_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # --- copy RGB weights and INITIALISE the 4th channel with the mean of R‑G‑B ---
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight          # copy first 3 channels
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)  # 4th channel = mean(R,G,B)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        # ---------------------------------------------------------------------------

        self.model.features[0][0] = new_conv                 # plug in the new conv layer

        # 3) Freeze *all* layers by default
        for p in self.model.parameters():
            p.requires_grad = False

        # 4) Un‑freeze classifier (“head”)
        for p in self.model.classifier.parameters():
            p.requires_grad = True

        # 5) Replace classifier for our number of classes
        in_features = self.model.classifier[1].in_features    # 1280 for EfficientNet‑B0
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
