from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch

class AutoencoderKLWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.module = vae
        self.out_channels = vae.config.latent_channels
        self.patch_size = [1, 8, 8]

    def encode(self, x):
        # x is (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x

    def decode(self, x):
        # x is (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.decode(x / 0.18215).sample
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x
    
class MultiModalVAEWrapper(nn.Module):
    """
    Keep the pretrained RGB VAE frozen.
    Provide small learnable projection convs that map sat channels / radar to VAE latent channels.
    Works for image mode (non-video). Input shapes assumed (B, C, H, W).
    """
    def __init__(self, vae, sat_channels=4):
        super().__init__()
        self.module = vae  # pretrained VAE
        self.patch_size = [1, 8, 8]

        # project extra satellite channels (e.g. channel 4) to latent channels
        self.sat_proj = nn.Conv2d(sat_channels, self.latent_ch, kernel_size=1)

        # project radar (single-channel) to latent channels (used as target latent)
        self.radar_proj = nn.Conv2d(1, self.latent_ch, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv2d(sat_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1)
        )

    def encode_rgb(self, x):
        # x: (B,3,H,W)
        with torch.no_grad():
            out = self.module.encode(x).latent_dist.sample().mul_(0.18215)  # (B, latent_ch, H/8, W/8)
        return out

    def encode_sat(self, x):
        # x: (B, extra_ch, H, W) -> downsample then project
        # Downsample factor must match VAE latent downsampling (8)
        down = F.avg_pool2d(x, kernel_size=8)
        return self.extra_proj(down)

    def encode_radar(self, x):
        # x: (B,1,H,W) -> downsample then project
        down = F.avg_pool2d(x, kernel_size=8)
        return self.radar_proj(down)

    def decode_from_radar_latent(self, z):
        # optional: decode using base VAE (z expected in latent space)
        with torch.no_grad():
            # VAE expects inputs scaled by /0.18215 when decoding
            return self.module.decode(z / 0.18215).sample