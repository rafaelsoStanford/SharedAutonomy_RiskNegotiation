
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

# Padding for images to be divisible by 2^depth
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x




class SelfAttention(nn.Module):
    """
    Transformer Structure:
    
    Attention is all you need paper (https://arxiv.org/abs/1706.03762): 
        See the diagram of the transformer architecture (example: the encoder)

    1. Multihead Attention 
    2-  Normalization
    3- Feed Forward Network 
    4-  Normalization
    """
    def __init__(self, channels):
        super().__init__()
        self.label_size = 1
        self.channels = channels        
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        self.emb =  nn.Embedding(10, 3) # Embedding layer: 10 labels, 3 channels

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        D = x.shape[-1] # Sequence length
        T = x.shape[-2]
        x = x.view(-1, self.channels, D * T).swapaxes(1, 2) # View(): reshapes the tensor to the desired shape
            # -1: infer this dimension from the other given dimension; Preserve number of batches
            # swapaxes(1, 2): swap the second and third dimension -> (B, C, len) -> (B, len, C)
        if y is not None:
            y = self.emb(y).view(-1, self.label_size, 3) # Embedding layer: 10 labels, 1 , 3 channels
            x = torch.cat([x, y], dim=1) # Concatenate the embedding to the input

        x_ln = self.ln(x) # Normalize input
        attention_value, _ = self.attention(x_ln, x_ln, x_ln) #Multihead attention: Pytorch Implementation
        attention_value = attention_value + x #Add residual connection (See paper; we add the input to the output of the multihead attention)
        attention_value = self.ff_self(attention_value) + attention_value #Second residual connection (see paper)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, D, T) #Swap back the second and third dimension and reshape to original image


class DoubleConvolution(nn.Module):
    """
    Structure taken from original UNet paper (https://arxiv.org/abs/1505.04597)
    Adjusted to fit implementation of DDPM (https://arxiv.org/abs/2006.11239) 

    Removed internal residual connections, coud not be bothered to implement them
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False) #Takes inputs of (B,Cin,H,W) where B is batch size, Cin is input channels, H is height, W is width  
        # Second 3x3 convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)


    def forward(self, x: torch.Tensor):

        # Apply the two convolution layers and activations
        x = self.first(x)   # (B,Cin,H,W) -> (B,Cout,H,W)
        x = self.norm(x)    # Group normalization
        x = self.act(x)     # GELU activation function (https://arxiv.org/abs/1606.08415)
        x = self.second(x)  # (B,Cin,H,W) -> (B,Cout,H,W)
        return self.norm(x) # Group normalization Final output shape (B,Cout,H,W)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a 2x2 max pooling operation with stride 2. 
    Two Double Convolution layers are applied to the feature map before each down-sampling operation. 
        (effectively doubling the number of channels)
    
    """

    def __init__(self, in_channels: int, out_channels: int, global_cond_dim: int ,embeddedTime_dim=256):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, ) #2x2 max pooling windows -> reduce size of feature map by 2
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)

        self.step_emb_layer = nn.Sequential( # Brutally make dimensions match using a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim, # IN: Dimension of embedded "denoising timestep" (B, 256)
                out_channels # OUT: Number of channels of the image (B, Channels_out)
            ),
        ) # Trainable layer: Not sure how okay this is, some repo's do it, some don't

        self.cond_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features= global_cond_dim, out_features=32),
            )
        

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_cond: torch.Tensor ):
        x = self.pool(x)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)

        # Time embedding added
        emb_t = self.step_emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # self.step_emb_layer(t) -> (B, C_out, 1, 1) 
                                                                                            #-> repeat to match image dimensions (B, C_out, img_s, img_s)
                                                                                            # -> same "time" value for all pixels
        x = x + emb_t

        # conditional embedding concatenation
        emb_cond = self.cond_emb_layer(x_cond.view(x_cond.shape[0], -1))
        emb_cond = emb_cond.view(emb_cond.shape[0], emb_cond.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return torch.cat([x, emb_cond],dim=1)



class UpSample(nn.Module):
    """
    ### Up-sample
    """
    def __init__(self, in_channels: int, out_channels: int, global_cond_dim: int, embeddedTime_dim=256):
        super().__init__()

        # Up-convolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.doubleConv1 = DoubleConvolution(in_channels, in_channels)
        self.doubleConv2 = DoubleConvolution(in_channels, out_channels)

        self.emb_layer = nn.Sequential( # Brutally make dimensions match unsing a linear layer
            nn.SiLU(),
            nn.Linear(
                embeddedTime_dim,
                out_channels),
        )
        self.cond_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features= global_cond_dim, out_features=32),
            )

    def forward(self, x: torch.Tensor, x_res: torch.Tensor,  t: torch.Tensor, x_cond : torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, x_res], dim=1)# Concatenate along the channel dimension; kept previous feature map and upsampled feature map
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        emb_t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x =  x + emb_t

        # conditional embedding concatenation
        emb_cond = self.cond_emb_layer(x_cond.view(x_cond.shape[0], -1))
        emb_cond = emb_cond.view(emb_cond.shape[0], emb_cond.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return torch.cat([x, emb_cond],dim=1)

class UNet_DDPM(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, global_cond_dim = None, time_dim=256):
        super(UNet_DDPM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.global_cond_dim = global_cond_dim

        # Define all layers used by U-net
        self.inc = DoubleConvolution(in_channels, 64)
        self.down1 = DownSample(64, 128, global_cond_dim) #set time_dim to 256 for all up and down sampling layers in init()
        #self.sa1 = SelfAttention(128) x2
        self.down2 = DownSample(128 + 32 , 256, global_cond_dim)
        #self.sa2 = SelfAttention(256) x3
        self.down3 = DownSample(256 + 32 , 256, global_cond_dim)
        #self.sa3 = SelfAttention(256) x4

        self.bot1 = DoubleConvolution(256 + 32 , 512) # x5
        self.bot2 = DoubleConvolution(512, 512) #x5
        self.bot3 = DoubleConvolution(512, 256 +32) #x5

        self.up1 = UpSample(2*(256+32), 256, global_cond_dim) #x5, x3
        #self.sa4 = SelfAttention(128)
        self.up2 = UpSample(2*32+(256+128), 128, global_cond_dim) #x: 128  x2 256
        #self.sa5 = SelfAttention(64)
        self.up3 = UpSample(32+(128 +64) , 64, global_cond_dim) #x x1
        #self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64 + 32 , out_channels, kernel_size=1)


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2,device=t.device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):

        if torch.is_tensor(t):
            t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Do the encoding
        
        x, padding = pad_to(x, 2**3)

        x1 = self.inc(x)
        
        x2 = self.down1(x1, t, y)
        #x2 = self.sa1(x2, y)
        x3 = self.down2(x2, t, y)
        #x3 = self.sa2(x3, y)
        x4 = self.down3(x3, t, y)
        #x4 = self.sa3(x4, y)

        x5 = self.bot1(x4)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x3, t, y) # include residual connections
        #x = self.sa4(x, y)
        x = self.up2(x, x2, t, y)
        #x = self.sa5(x, y)
        x = self.up3(x, x1, t, y)
        #x = self.sa6(x, y)

        x = self.outc(x)

        x = unpad(x , padding)

        return x

