import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(num_groups=1, num_channels=channels)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v, use_gumbel_softmax, is_train, hard_mode=False):
        """
        Apply QKV attention.

        :param q: an [B x C x 1] tensor of query for point feature.
        :param k: an [B x C x T] tensor of keys for image features.
        :param v: an [B x C x T] tensor of values for image features.
        :return: an [B x C x 1] tensor after attention.
        """
        bs, width, length = k.shape
        assert k.shape == v.shape
        assert width % (self.n_heads) == 0
        ch = width//self.n_heads
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, -1),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # import pdb; pdb.set_trace()
        if use_gumbel_softmax:
            if is_train:
                weight_onehot = F.gumbel_softmax(torch.log(weight), tau=1, hard=hard_mode, dim=-1)
                weight = weight * weight_onehot
            else:
                weight_onehot = (weight == weight.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float32)
                weight = weight * weight_onehot

            a = torch.einsum("bts,bcs->bct", weight_onehot, v.reshape(bs * self.n_heads, ch, -1))
        else:
            a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))

        weight_sum = torch.sum(weight, dim=2, keepdim=True)
        if self.n_heads > 1:
            weight_sum = weight_sum.reshape(bs, self.n_heads, -1)
            weight_sum = torch.sum(weight_sum, dim=1, keepdim=True)

        return a.reshape(bs, width, -1), weight_sum


class AttentionBlock(nn.Module):
    def __init__(
        self,
        query_channels,
        context_channels,
        inner_channels=32,
        num_heads=1,
    ):
        super().__init__()
        assert inner_channels % num_heads == 0
        self.num_heads = num_heads
        self.inner_channels = inner_channels
        self.norm_query = normalization(query_channels)
        self.norm_context = normalization(context_channels)
        self.q = conv_nd(1, query_channels, inner_channels, 1)
        self.kv = conv_nd(1, context_channels, inner_channels * 2, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, inner_channels, context_channels, 1))

    def forward(self, pt_feat, img_feats, use_gumbel_softmax=False, is_train=False, frame_level_attention=False, num_nearest_frame=0):
        # import pdb; pdb.set_trace()
        b, c, *spatial = pt_feat.shape
        b_, c_, *spatial_ = img_feats.shape
        pt_feat = pt_feat.reshape(b, c, -1)
        img_feats = img_feats.reshape(b_, c_, -1)
        q = self.q(self.norm_query(pt_feat))
        kv = self.kv(self.norm_context(img_feats))
        k, v = kv.chunk(2, dim=1)
        if frame_level_attention:
            assert img_feats.shape[2] % num_nearest_frame == 0
            num_candidates = int(img_feats.shape[2] / num_nearest_frame)
            h_sum = 0
            weight_sum = 0
            for idx in range(num_nearest_frame):
                k_part = k[:, :, idx*num_candidates:(idx+1)*num_candidates]
                v_part = v[:, :, idx*num_candidates:(idx+1)*num_candidates]
                h_part, weight_part = self.attention(q, k_part, v_part, use_gumbel_softmax, is_train)
                h_sum = h_sum + h_part * weight_part
                weight_sum = weight_sum + weight_part
            h = h_sum / (weight_sum + 1e-8)
        else:
            h, _ = self.attention(q, k, v, use_gumbel_softmax, is_train)
        h = self.proj_out(h)

        return h[:, :, 0]

