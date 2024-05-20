import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, drop_path, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskDecoder(nn.Module):
    def __init__(self, n_cls, d_model, n_layers, nhead, drop_path_rate):
        super().__init__()
        self.n_cls = n_cls
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, nhead, d_model*4, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        scale = d_model ** -0.5
        self.proj_mask = nn.Parameter(scale * torch.randn(d_model, d_model))
        self.proj_cls = nn.Parameter(scale * torch.randn(d_model, d_model))
        self.mask_norm = nn.LayerNorm(self.n_cls)

    def forward(self, x, bs, h, w):
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        mask_feat, cls_feat = x[:, :-self.n_cls], x[:, -self.n_cls:]
        mask_feat = mask_feat @ self.proj_mask.to(dtype=mask_feat.dtype, device=mask_feat.device)
        mask_feat = mask_feat / mask_feat.norm(dim=-1, keepdim=True)
        cls_feat = cls_feat @ self.proj_cls.to(dtype=cls_feat.dtype, device=cls_feat.device)
        cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
        masks = mask_feat @ cls_feat.transpose(1, 2)
        masks = self.mask_norm(masks)  # [b,1024,21]
        masks = masks.reshape(bs, h, w, -1).permute(0, 3, 1, 2)

        return masks


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return pos_encoding