from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy

from models.decode_heads.utils import positional_encoding

import math

from functools import reduce
from operator import mul


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore
        

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@HEADS.register_module()
class CLIPHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            embed_dims=768,
            num_layers=2,
            num_heads=8,
            use_stages=1,
            use_proj=True,
            crop_train=False,
            drop_path_rate=0.1,
            **kwargs,
    ):
        super(CLIPHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.num_layers = num_layers
        self.num_heads = num_heads

        dim = embed_dims
        input_proj = []
        proj_norm = []
        vis_proj = []
        tex_proj = []
        vis_norm = []
        tex_norm = []

        self.scale = dim ** -0.5

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)

            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)

            # vis_proj, tex_proj
            v_proj = nn.Linear(dim, dim)
            v_norm = nn.LayerNorm(dim)
            t_proj = nn.Linear(dim, dim)
            t_norm = nn.LayerNorm(dim)
            trunc_normal_(v_proj.weight, std=.02)
            trunc_normal_(t_proj.weight, std=.02)
            self.add_module("vis_proj_{}".format(i + 1), v_proj)
            self.add_module("cls_proj_{}".format(i + 1), t_proj)
            self.add_module("vis_norm_{}".format(i + 1), v_norm)
            self.add_module("tex_norm_{}".format(i + 1), t_norm)
            vis_proj.append(v_proj)
            tex_proj.append(t_proj)
            vis_norm.append(v_norm)
            tex_norm.append(t_norm)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.vis_proj = vis_proj
        self.tex_proj = tex_proj
        self.vis_norm = vis_norm
        self.tex_norm = tex_norm
        self.cls_proj = nn.Linear(dim * 2, dim)

        delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def forward_cascade(self, img_feat, text_feat, proj_norm, input_proj, v_proj, t_proj, v_norm, t_norm):
        vis_token, cls_token = img_feat[0], img_feat[1]  # [b,hw,512] [b,512]
        tex_token = text_feat  # [cls,512]
        cls_token = self.get_cls_token(tex_token, cls_token)  # [b,cls,512]
        vis_token = proj_norm(input_proj(vis_token))
        vis_token = v_norm(v_proj(vis_token))
        cls_token = t_norm(t_proj(cls_token))
        masks = (vis_token @ cls_token.transpose(1, 2)) * self.scale  # b,hw,cls
        masks = self.d3_to_d4(masks)

        return masks

    def forward(self, inputs_both, self_training=None):
        """
        inputs_both: list, i.e., [visual_feat, text_feat]
                    visual_feat: list of tuple, [(size(b,1024,512), size(b,512)), ...], length=len(out_indices)+1
                    text_feat: list of out_indices-th layer's ouput (size: (cls,77,512))
        """
        img_feats = inputs_both[0]
        text_feats = inputs_both[1]
        outs = []

        for idx in range(self.use_stages):
            img_feat = img_feats[idx]
            text_feat = text_feats[idx]
            proj_norm = self.proj_norm[idx]
            input_proj = self.input_proj[idx]
            v_proj = self.vis_proj[idx]
            t_proj = self.tex_proj[idx]
            v_norm = self.vis_norm[idx]
            t_norm = self.tex_norm[idx]

            # forward_cascade
            masks = self.forward_cascade(img_feat, text_feat, proj_norm, input_proj, v_proj, t_proj, v_norm, t_norm)
            masks = F.interpolate(masks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            out = {"pred_masks": masks}

            if not self.training:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.1)
            outs.append(out)

        if not self.training:
            return outs[-1]["pred"]

        return outs[-1]

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        # mask_pred = mask_pred.softmax(dim=1)
        mask_pred[:, seen_idx] = mask_pred[:, seen_idx] - weight
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_cls_token(self, tex, cls):
        # C, dim = tex.shape
        bs, _ = cls.shape
        tex = tex.expand(bs, -1, -1)
        tex_tmp = torch.einsum("bd,bcd->bcd", cls, tex)
        tex_tmp = torch.concat((tex_tmp, tex), dim=-1)

        return self.cls_proj(tex_tmp)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index = self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss