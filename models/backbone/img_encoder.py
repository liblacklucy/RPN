from collections import OrderedDict
from curses import A_ALTCHARSET
from tkinter import OUTSIDE
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
from mmseg.models.builder import BACKBONES

from mmseg.models.backbones import ResNet
from mmseg.models.backbones import VisionTransformer as MMVisionTransformer

from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck

from functools import reduce
from operator import mul

import math
from .utils import *


@BACKBONES.register_module()
class CPTCLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512,
                 drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=False,
                 num_tokens=20, prompt_dim=512, total_d_layer=12, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.width = width
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers
        self.patch_size = patch_size

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices
        # self.ln_post = list()  # list of LayerNorm
        # self.proj = list()  # list of Parameter
        # for _ in range(len(self.out_indices)+1):
        #     ln_post = LayerNorm(width)
        #     proj = nn.Parameter(scale * torch.randn(width, output_dim))
        #     self.ln_post.append(ln_post)
        #     self.proj.append(proj)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # embed_dim = width

        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer)

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        assert total_d_layer > 0, 'Input correct total_d_layer, prompt vector must be positive channels!'
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
        self.prompt_dropout = Dropout(0.1)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # b, 1024+1, 768

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)  # b,1025,768

        # prompt forward
        features = []
        outs = []
        features = self.forward_prompt(x, features, H, W)  # list([1035,b,768 or 1025,b,768])

        for i, feature in enumerate(features):
            feature = feature.permute(1, 0, 2)  # b,1035,768
            feature = self.ln_post(feature)
            feature = feature @ self.proj  # b,1035,512

            cls_embedding = feature[:, 0]  # b,512
            vis_embedding = feature[:, -(H * W):]  # b,1024,512
            cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)  # b,512
            vis_embedding = vis_embedding / vis_embedding.norm(dim=-1, keepdim=True)  # b,1024,512
            outs.append((vis_embedding, cls_embedding))

        # return [outs[-1][0]], outs[-1][1] # list of tuple, [(size(b,1024,512), size(b,512)), ...], length=len(out_indices)+1
        return outs  # list of tuple, [(size(b,1024,512), size(b,512)), ...], length=len(out_indices)+1

    def forward_prompt(self, embedding_output, features, H, W):
        # concat prompt0
        B = embedding_output.shape[0]
        embedding_output = embedding_output.permute(1, 0, 2)  # 1025, b, 768

        # forward
        hidden_states = embedding_output
        for i in range(self.num_layers):
            if i <= self.total_d_layer - 1:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],  # 1,b,768
                    self.prompt_dropout(
                        self.prompt_proj(self.deep_prompt_embeddings[i]).expand(B, -1, -1).permute(1, 0, 2)),
                    # 10,b,768
                    hidden_states[-(H * W):, :, :]  # 1024,b,768
                ), dim=0)  # 1035,b,768
            else:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[-(H * W):, :, :]
                ), dim=0)
            hidden_states = self.transformer.resblocks[i](hidden_states)
            # if i in self.out_indices:
            features.append(self.prompt_norm(hidden_states))  # 1035,b,768 or 1025,b,768
        # features.append(self.prompt_norm(hidden_states))  # 1035,b,768 or 1025,b,768 the last layer's output

        return features

    def input_preprocessing(self, x):
        x = self.conv1(x)  # downsample 16x
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # b,hw+1,768

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # hw+1,b,768

        return x, B, H, W

    def output_postprocessing(self, embedding_out, H, W):
        feature = embedding_out
        feature = feature.permute(1, 0, 2)  # b,1+10+hw,768 or b,1+10+2hw,768
        feature = self.ln_post(feature)
        feature = feature @ self.proj  # b,1+10+hw,512 or b,1+10+2hw,512
        cls_embedding = feature[:, 0]  # b,512
        vis_embedding = feature[:, -(H * W):]  # b,hw,512
        cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)  # b,512
        vis_embedding = vis_embedding / vis_embedding.norm(dim=-1, keepdim=True)  # b,hw,512

        return vis_embedding, cls_embedding

    def forward_first_layer_for_mpt(self, x):
        embedding, B, H, W = self.input_preprocessing(x)
        embedding = torch.cat((
            embedding[:1, :, :],  # 1,b,768
            self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[0]).expand(B, -1, -1).permute(1, 0, 2)),  # 10,b,768
            embedding[-(H * W):, :, :]  # hw,b,768
        ), dim=0)  # 1035,b,768
        embedding_out = self.transformer.resblocks[0](embedding)  # 1+10+hw,b,768
        embedding_out_post = self.output_postprocessing(self.prompt_norm(embedding_out), H, W)

        return embedding_out, embedding_out_post, B, H, W

    def forward_later_layer_for_mpt(self, embedding_out, mask_prompt, B, H, W, idx):
        embedding_out = torch.cat((
            embedding_out[:1, :, :],
            self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[idx]).expand(B, -1, -1).permute(1, 0, 2)),  # 10,b,768
            self.prompt_dropout(self.prompt_proj(mask_prompt)),  #  cls,b,768
            # self.prompt_dropout(mask_prompt),  #  cls,b,768
            embedding_out[-(H * W):, :, :]  # hw,b,768
        ),dim = 0)
        embedding_out = self.transformer.resblocks[idx](embedding_out)  # 1+10+cls+hw,b,768
        embedding_out_post = self.output_postprocessing(self.prompt_norm(embedding_out), H, W)

        return embedding_out, embedding_out_post