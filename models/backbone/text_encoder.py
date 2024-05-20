from collections import OrderedDict
import imghdr
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
from mmseg.models.builder import BACKBONES

from mmseg.models.backbones import ResNet
from mmseg.models.backbones import VisionTransformer as MMVisionTransformer

from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck

import math
from .utils import *
from ..segmentor.untils import tokenize



@BACKBONES.register_module()
class CPTCLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 out_indices=[11],
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.transformer_layers = transformer_layers
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.out_indices = out_indices
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]

                if k == 'positional_embedding' or k == 'text_projection' or k.startswith(
                        'token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_prompt(self, embedding_output):
        features = []
        hidden_states = embedding_output
        for idx in range(self.transformer_layers):
            hidden_states = self.transformer.resblocks[idx](hidden_states)  # 77,cls,512
            features.append(hidden_states)

        return features

    def forward(self, text):
        outs = []
        x = self.token_embedding(text)  # cls,77,512
        x = x + self.positional_embedding  # cls,77,512
        x = x.permute(1, 0, 2)  # 77,cls,512

        features = self.forward_prompt(x)

        for _f in features:
            _f = _f.permute(1, 0, 2)  # cls,77,512
            _f = self.ln_final(_f)
            _f = _f[torch.arange(_f.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            _f = _f / _f.norm(dim=-1, keepdim=True)
            outs.append(_f)

        return outs  # list of size(cls,512)

    # def output_postprocessing(self, embedding_out, text):
    #     feature = embedding_out
    #     feature = feature.permute(1, 0, 2)  # cls,77,512
    #     feature = self.ln_final(feature)
    #     feature = feature[torch.arange(feature.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    #     feature = feature / feature.norm(dim=-1, keepdim=True)
    #
    #     return feature
    #
    # def forward_first_layer_for_mpt(self, text):
    #     x = self.token_embedding(text)  # cls,77,512
    #     x = x + self.positional_embedding  # cls,77,512
    #     x = x.permute(1, 0, 2)  # 77,cls,512
    #     embedding_out = self.transformer.resblocks[0](x)  # 77,cls,512
    #     embedding_out_post = self.output_postprocessing(embedding_out, text)  # cls,512
    #
    #     return embedding_out, embedding_out_post
    #
    # def forward_later_layer_for_mpt(self, embedding_out, text, idx):
    #     embedding_out = self.transformer.resblocks[idx](embedding_out)  # 77,cls,512
    #     embedding_out_post = self.output_postprocessing(embedding_out, text)  # cls,512
    #
    #     return embedding_out, embedding_out_post