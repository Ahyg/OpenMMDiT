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
    def __init__(self, vae, sat_chn=4, radar_chn=1, if_freeze=True):
        super().__init__()
        self.module = vae  # pretrained VAE
        self.if_freeze = if_freeze
        if self.if_freeze:
            for p in self.module.parameters():
                p.requires_grad = False

        # project satellite / radar channels to RGB channels
        self.sat_in = nn.Conv2d(sat_chn, 3, kernel_size=1)
        self.sat_out = nn.Conv2d(3, sat_chn, kernel_size=1)
        self.radar_in = nn.Conv2d(radar_chn, 3, kernel_size=1)
        self.radar_out = nn.Conv2d(3, radar_chn, kernel_size=1)

    def encode(self, x):
        # x: (B, 3, H, W) -> (B, latent_ch, H/8, W/8)
        with torch.no_grad() if self.if_freeze else torch.enable_grad():
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)  # (B, latent_ch, H/8, W/8)
        return x
    
    def decode(self, x):
        # x: (B, latent_ch, H/8, W/8) -> (B, 3, H, W)
        with torch.no_grad() if self.if_freeze else torch.enable_grad():
            x = self.module.decode(x / 0.18215).sample
        return x

    def encode_sat(self, x):
        # x: (B, sat_chn, H, W)
        x = self.sat_in(x)
        x = self.encode(x)
        return x

    def encode_radar(self, x):
        # x: (B, radar_chn, H, W)
        x = self.radar_in(x)
        x = self.encode(x)
        return x
    
    def decode_sat(self, x):
        # x: (B, latent_ch, H/8, W/8)
        x = self.decode(x)
        x = self.sat_out(x)
        return x

    def decode_radar(self, x):
        # x: (B, latent_ch, H/8, W/8)
        x = self.decode(x)
        x = self.radar_out(x)
        return x

