'''
Reference: https://github.com/milesial/Pytorch-UNet
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                                     Vanilla UNet Model                                              +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class UNet_simplified(nn.Module):
    def __init__(self, n_channels=1, cond_dim =1, noise_steps=1000, time_dim=256):
        super(UNet_simplified, self).__init__()
        self.n_channels   = n_channels
        self.cond_dim    = cond_dim
        self.noise_steps  = noise_steps
        self.time_dim     = time_dim

        self.input_conv = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32, cond_dim))
        # self.down2 = (Down(32, 64))
        self.down3 = (Down(32+32, 64 ,cond_dim))
        factor = 2
        self.down4 = (Down(64+32, 128 // factor ,cond_dim))
        self.up1 = (Up(128+64, 64 // factor,cond_dim))
        # self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64+64, 32 // factor,cond_dim))
        self.up4 = (Up(32+32, 16,cond_dim))
        self.outc = nn.Conv2d(in_channels=16+32, out_channels=n_channels, kernel_size=(1, 1))


    def forward(self, x, t, x_cond):
        t = self.pos_encoding(t, 256)

        x1 = self.input_conv(x)
        x2 = self.down1(x1, t, x_cond)
        # x3 = self.down2(x2)
        x4 = self.down3(x2, t, x_cond)
        x5 = self.down4(x4, t, x_cond)
        x = self.up1(x5, x4, t, x_cond)
        # x = self.up2(x, x3)
        x = self.up3(x, x2, t, x_cond)
        x = self.up4(x, x1, t, x_cond)

        logits = self.outc(x)
        return logits

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2,device=t.device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    # '''
    # USAGE:
    # ======================
    # try:
    #     train_model(...)
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                 'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                 'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     model.use_checkpointing()
    #     train_model(...)
    # ======================
    # '''
    # def use_checkpointing(self):
    #     self.inc = checkpoint(self.inc)
    #     self.down1 = checkpoint(self.down1)
    #     self.down2 = checkpoint(self.down2)
    #     self.down3 = checkpoint(self.down3)
    #     self.down4 = checkpoint(self.down4)
    #     self.up1 = checkpoint(self.up1)
    #     self.up2 = checkpoint(self.up2)
    #     self.up3 = checkpoint(self.up3)
    #     self.up4 = checkpoint(self.up4)
    #     self.outc = checkpoint(self.outc)

# ===========================================
#  Parts of the U-Net model 
# ===========================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False):
        """Double convolutions as applied in the unet paper architecture. """
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False,),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return F.gelu(self.double_conv(x))
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int, in_features, emb_dim: int = 256, conditional: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

        self.conditional = conditional
        if self.conditional:
            self.cond_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=in_features, out_features=32),
            )

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor, x_cond=None) -> torch.Tensor:
        """Downsamples input tensor, calculates embedding and adds embedding channel wise.

        If, `x.shape == [4, 64, 64, 64]` and `out_channels = 128`, then max_conv outputs [4, 128, 32, 32] by
        downsampling in h, w and outputting specified amount of feature maps/channels.

        `t_embedding` is embedding of timestep of shape [batch, time_dim]. It is passed through embedding layer
        to output channel dimentsion equivalent to channel dimension of x tensor, so they can be summbed elementwise.

        Since emb_layer output needs to be summed its output is also `emb.shape == [4, 128]`. It needs to be converted
        to 4D tensor, [4, 128, 1, 1]. Then the channel dimension is duplicated in all of `H x W` dimension to get
        shape of [4, 128, 32, 32]. 128D vector is sample for each pixel position is image. Now the emb_layer output
        is summed with max_conv output.
        """
        x = self.maxpool_conv(x)

        # Add time embedding
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + emb

        if self.conditional:
            # Add conditional embedding
            cond_emb = self.cond_emb_layer(x_cond.view(x_cond.shape[0], -1))
            cond_emb = cond_emb.view(cond_emb.shape[0], cond_emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, cond_emb],dim=1)
        return x
    
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int,   out_channels: int, in_features, emb_dim: int = 256, conditional: bool = True):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

        self.conditional = conditional
        if self.conditional:
            self.cond_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=in_features, out_features=32),
            )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, t_embedding: torch.Tensor, x_cond=None) -> torch.Tensor:
        x = self.up(x)
        diffY = x_skip.size()[2] - x.size()[2]
        diffX = x_skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)

        # Add time embedding
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + emb

        if self.conditional:
            # Add conditional embedding
            cond_emb = self.cond_emb_layer(x_cond.view(x_cond.shape[0], -1))
            cond_emb = cond_emb.view(cond_emb.shape[0], cond_emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x,cond_emb],dim=1)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 1000, apply_dropout: bool = True):
        """Section 3.5 of attention is all you need paper.

        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.

        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.

        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """Get precalculated positional embedding at timestep t. Outputs same as video implementation
        code but embeddings are in [sin, cos, sin, cos] format instead of [sin, sin, cos, cos] in that code.
        Also batch dimension is added to final output.
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                                    Transformer UNet Model                                           +
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class UNetTrans(nn.Module):
    def __init__(self,
            n_channels: int   = 1,
            out_channels: int = 1,
            noise_steps: int  = 1000,
            time_dim: int     = 256,
            features: list    = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.time_dim     = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps+1)

        self.input_conv = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32, conditional=False)
        self.sa1   = TransformerEncoderSA(32)
        self.down2 = Down(32, 64, conditional=False)
        self.sa2   = TransformerEncoderSA(64)
        self.down3 = Down(64, 64, conditional=False)
        self.sa3   = TransformerEncoderSA(64)

        self.bottleneck1 = DoubleConv(64, 128)
        self.bottleneck2 = DoubleConv(128, 128)
        self.bottleneck3 = DoubleConv(128, 64)

        self.up1 = Up(128, 32, conditional=False)
        self.sa4 = TransformerEncoderSA(32)
        self.up2 = Up(64, 16, conditional=False)
        self.sa5 = TransformerEncoderSA(16)
        self.up3 = Up(32, 16, conditional=False)
        self.sa6 = TransformerEncoderSA(16)
        self.out_conv = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=(1, 1))

        print('\n---UNet+Transformer Model---\n')
        print('# weights: ',np.sum([param.nelement() for param in self.parameters()]))
        print(self)

    def forward(self, x: torch.Tensor, t: torch.LongTensor, x_cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with image tensor and timestep reduce noise.

        Args:
            x: Image tensor of shape, [batch_size, channels, height, width].
            t: Time step defined as long integer. If batch size is 4, noise step 500, then random timesteps t = [10, 26, 460, 231].
        """
        t = self.pos_encoding(t)

        x1 = self.input_conv(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bottleneck1(x4)
        x4 = self.bottleneck2(x4)
        x4 = self.bottleneck3(x4)

        x  = self.up1(x4, x3, t)
        x  = self.sa4(x)
        x  = self.up2(x, x2, t)
        x  = self.sa5(x)
        x  = self.up3(x, x1, t)
        x  = self.sa6(x)

        x = self.out_conv(x)
        return x
    
# ===========================================
#  Additional Parts of the U-Net model 
# ===========================================

class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4):
        """A block of transformer encoder with mutli head self attention from vision transformers paper,
         https://arxiv.org/pdf/2010.11929.pdf.
        """
        super().__init__()
        self.num_channels = num_channels
        # self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([num_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self attention.

        Input feature map [4, 128, 32, 32], flattened to [4, 128, 32 x 32]. Which is reshaped to per pixel
        feature map order, [4, 1024, 128].

        Attention output is same shape as input feature map to multihead attention module which are added element wise.
        Before returning attention output is converted back input feature map x shape. Opposite of feature map to
        mha input is done which gives output [4, 128, 32, 32].
        """
        self.size1 = x.size(2)
        self.size2 = x.size(3)
        x = x.view(-1, self.num_channels, self.size1 * self.size2).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(-1, self.num_channels, self.size1, self.size2)