import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MlpBlock(nn.Module):
    """A simple two-layer MLP with GELU activation."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MixerBlock(nn.Module):
    """One Mixer layer: token-mixing + channel-mixing, each with skip connection."""
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.token_mixing  = MlpBlock(num_patches, tokens_mlp_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        # x shape: (batch, num_patches, hidden_dim)

        # Token-mixing: operate along the patch (spatial) dimension
        y = self.norm1(x)
        y = y.transpose(1, 2)          # → (batch, hidden_dim, num_patches)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)          # → (batch, num_patches, hidden_dim)
        x = x + y                      # skip connection

        # Channel-mixing: operate along the feature dimension
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y                      # skip connection

        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        hidden_dim=512,
        num_layers=8,
        tokens_mlp_dim=256,
        channels_mlp_dim=2048,
        dropout=0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2

        # Split image into patches and project to hidden_dim
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),  # flatten spatial dims
        )

        self.mixer_layers = nn.Sequential(*[
            MixerBlock(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)          # (B, num_patches, hidden_dim)
        x = self.dropout(x)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)               # global average pooling over patches
        return self.head(x)