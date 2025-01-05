'''
The encoder module is responsible for encoding the image into a feature vector conditioned on the label latent vector.
the feature vector will be used as the coeeficients of the Fourier series used to approximate the image.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention
import numpy as np


def modulate(x, shift, scale):
    '''
    Modulate the input tensor x with the given shift and scale.
    '''
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
    

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class ViT_block(nn.Module):
    '''
    Single block of the Vision Transformer.
    '''
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, int(6 * hidden_size)),
        )
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.module(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

class FinalLayer(nn.Module):
    '''
    Final layer of the encoder, which outputs the Fourier series coefficients.
    Final layer is a module that takes the output of the encoder and the label latent vector as input.
    
    Input: x, c
        x: (Batch, num_patches, hidden_size), the output of the encoder
        c: (Batch, hidden_size), the label latent vector
    Output: A0, An, Bn
        A0: (Batch, C), the DC component of the Fourier series
        An: {"An_x": An_x, "An_y": An_y, "An_xy": An_xy}, where An_x, An_y, An_xy are tensors of shape (Batch, C, N), N is the number of Fourier series terms
        Bn: {"Bn_x": Bn_x, "Bn_y": Bn_y, "Bn_xy": Bn_xy}, where Bn_x, Bn_y, Bn_xy are tensors of shape (Batch, C, N), N is the number of Fourier series terms
    '''
    def __init__(self, hidden_size, channel, num_fourier_terms):
        super().__init__()
        self.A0 = ConcatSquashLinear(hidden_size, 1, hidden_size)

        self.An_xy = ConcatSquashLinear(hidden_size, num_fourier_terms, hidden_size)
        self.Bn_xy = ConcatSquashLinear(hidden_size, num_fourier_terms, hidden_size)
        
        self.An_yx = ConcatSquashLinear(hidden_size, num_fourier_terms, hidden_size)
        self.Bn_yx = ConcatSquashLinear(hidden_size, num_fourier_terms, hidden_size)
        
        self.channel = channel
    
    def forward(self, x, c):
        x = x[:, :self.channel, :]
        c = c.unsqueeze(1)
        A0 = self.A0(c, x).squeeze(-1)

        An_xy = self.An_xy(c, x)
        Bn_xy = self.Bn_xy(c, x)
        
        An_yx = self.An_yx(c, x)
        Bn_yx = self.Bn_yx(c, x)
        
        return A0, {"An_xy": An_xy, "An_yx": An_yx}, {"Bn_xy": Bn_xy, "Bn_yx": Bn_yx}


class Encoder(nn.Module):
    '''
    Encoder module for encoding the image into tokens conditioned on the label latent vector.
    Use ViT as the encoder.
    The encoded tokens should be reconstructed by the Fourier series.
    '''
    def __init__(self, input_size=32, in_channels=4, 
                 num_classes=1000, num_fourier_terms=512,
                 hidden_size=1024, depth=12, num_heads=16, 
                 mlp_ratio=4.0, patch_size=4,
                 return_coefficients=False
                 ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_fourier_terms = num_fourier_terms
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.return_coefficients = return_coefficients
        
        # Embed 
        self.label_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            ViT_block(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        self.initilize_parameters()
    
    def initilize_parameters(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.module[-1].weight, 0)
            nn.init.constant_(block.module[-1].bias, 0)

    def forward(self, x, labels):
        B, C, H, W = x.shape
        x = self.x_embedder(x) + self.pos_embed # (B, C, H, W) -> (B, num_patches, hidden_size)
        c = self.label_embedder(labels) # (B, hidden_size)
        for block in self.blocks:
            x = block(x, c)
        return x
    

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                                   ViT Configs                                  #
#################################################################################
def Enc_XL_2(**kwargs):
    return Encoder(depth=24, num_heads=16, hidden_size=1152, patch_size=2, **kwargs)

def Enc_XL_4(**kwargs):
    return Encoder(depth=24, num_heads=16, hidden_size=1152, patch_size=4, **kwargs)

def Enc_XL_8(**kwargs):
    return Encoder(depth=24, num_heads=16, hidden_size=1152, patch_size=8, **kwargs)

def Enc_L_2(**kwargs):
    return Encoder(depth=16, num_heads=16, hidden_size=1024, patch_size=2, **kwargs)

def Enc_L_4(**kwargs):
    return Encoder(depth=16, num_heads=16, hidden_size=1024, patch_size=4, **kwargs)

def Enc_L_8(**kwargs):
    return Encoder(depth=16, num_heads=16, hidden_size=1024, patch_size=8, **kwargs)

def Enc_B_2(**kwargs):
    return Encoder(depth=12, num_heads=8, hidden_size=768, patch_size=2, **kwargs)

def Enc_B_4(**kwargs):
    return Encoder(depth=12, num_heads=8, hidden_size=768, patch_size=4, **kwargs)

def Enc_B_8(**kwargs):
    return Encoder(depth=12, num_heads=8, hidden_size=768, patch_size=8, **kwargs)

Enc_models = {
    "Enc_XL_2": Enc_XL_2,
    "Enc_XL_4": Enc_XL_4,
    "Enc_XL_8": Enc_XL_8,
    "Enc_L_2": Enc_L_2,
    "Enc_L_4": Enc_L_4,
    "Enc_L_8": Enc_L_8,
    "Enc_B_2": Enc_B_2,
    "Enc_B_4": Enc_B_4,
    "Enc_B_8": Enc_B_8,
}