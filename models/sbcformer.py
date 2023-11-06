from turtle import forward
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import math
import itertools

from timm.models.layers import DropPath, trunc_normal_, create_pool2d, create_conv2d
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
    
        # global FLOPS_COUNTER
        # output_points = ((resolution + 2 * padding - dilation *
        #                   (ks - 1) - 1) // stride + 1)**2
        # FLOPS_COUNTER += a * b * output_points * (ks**2) // groups
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MV2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1,  stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.hidden_features, self.hidden_features, kernel_size=3,  stride=1, padding=1, groups= self.hidden_features)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.in_features, kernel_size=1,  stride=1, padding=0)

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # @line_profile
    def forward(self, x):
        x1 = self.pwconv1_bn(x)
        x1 = self.act(x1)
        x1 = self.dwconv_bn(x1)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        return x + x1


class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=2, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5
        
        self.nh_kd = key_dim * num_heads
        self.qk_dim = 2 * self.nh_kd
        self.v_dim = int(attn_ratio * key_dim) * num_heads
        dim_h = self.v_dim + self.qk_dim

        self.N = resolution ** 2
        self.N2 = self.N
        # self.qkv = nn.Linear(dim, h)
        self.pwconv = nn.Conv2d(dim, dim_h, kernel_size=1,  stride=1, padding=0)
        self.dwconv = Conv2d_BN(self.v_dim, self.v_dim, kernel_size=3,  stride=1, padding=1, groups=self.v_dim)
        # self.pwconv = nn.Conv2d(h, h, kernel_size=1,  stride=1, padding=0)
        self.proj_out = nn.Linear(self.v_dim, dim)
        self.act = nn.GELU()

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        h, w = self.resolution, self.resolution
        x = x.transpose(1, 2).reshape(B, C, h, w)

        x = self.pwconv(x)
        
        qk, v1 = x.split([self.qk_dim, self.v_dim], dim=1)
        qk = qk.reshape(B, 2, self.num_heads, self.key_dim, N).permute(1, 0, 2, 4, 3)
        q, k = qk[0], qk[1]

        v1 = v1 + self.act(self.dwconv(v1))
        v = v1.reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim)
        x = self.proj_out(x)
        return x

class MobileTransformer(nn.Module):   #TimmBlock for twins
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio= 2, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=self.dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, resolution=resolution)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim*mlp_ratio, out_features=self.dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SlenderViTBlock(nn.Module):   #TimmBlock for twins
    def __init__(self, depth_mv2, depth_trans, depth_mixer, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2, drop=0., attn_drop=0.,
                 drop_paths=[], act_layer=nn.GELU, pool_ratio=1, mv2_ratio=1, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.depth_mv2 = depth_mv2
        self.depth_trans = depth_trans
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        self.mv2_blocks = nn.Sequential()
        for k in range(self.depth_mv2):
            self.mv2_blocks.add_module("mv2_{0}".format(k), MV2(in_features=dim, hidden_features=int(dim*mv2_ratio), out_features=dim, kernel_size=3, drop_path=0.))

        self.pool_ratio= pool_ratio
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.convTrans= nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm = nn.Identity()
        
        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer):
            self.mixer.add_module("proj_{0}".format(k), MV2(in_features=dim, hidden_features=dim*2, out_features=dim, kernel_size=3, drop_path=0.))
        
        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_trans):
            self.trans_blocks.add_module("mobile_trans_{0}".format(k), MobileTransformer(dim=dim, key_dim=key_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
             drop=drop, attn_drop=attn_drop, drop_path=drop_paths[k], resolution=resolution))
        
        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1,  stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim*2, self.dim, kernel_size=1,  stride=1, padding=0)
        
        # no local ablation test.
        # self.proj_fuse = Conv2d_BN(self.dim, self.dim, kernel_size=1,  stride=1, padding=0)
        
    def forward(self, x):
        B, C, _, _ = x.shape
        h, w = self.resolution, self.resolution
        x = self.mv2_blocks(x)
        local_fea = x

        if self.pool_ratio > 1.:
            x = self.pool(x)
        
        x = self.mixer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.trans_blocks(x)
        x = x.transpose(1, 2).reshape(B, C, h, w)

        if self.pool_ratio > 1:
            x = self.convTrans(x)
            x = self.norm(x)
        global_act = self.act(self.proj(x))
        x_mul = local_fea * global_act
        x_cat = torch.cat((x, x_mul), dim=1)
        out = self.proj_fuse(x_cat)

        # out = self.proj_fuse(global_act)

        return out


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """
    def __init__(self,  in_features, out_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(out_features)
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, in_chans=3, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim        
        #ImageNet
        self.stem = nn.Sequential(
                            Conv2d_BN(in_features=in_chans, out_features=self.embed_dim//4, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            Conv2d_BN(in_features=self.embed_dim//4, out_features=self.embed_dim//2, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            Conv2d_BN(in_features=self.embed_dim//2, out_features=self.embed_dim, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True))
        
        # self.stem = nn.Sequential(
        #                     Conv2d_BN(in_features=in_chans, out_features=self.embed_dim, kernel_size=8, stride=8),
        #                     nn.ReLU(inplace=True),
        #                     # Conv2d_BN(in_features=self.embed_dim//4, out_features=self.embed_dim//2, kernel_size=3, stride=2, padding=1),
        #                     # nn.ReLU(inplace=True),
        #                     # Conv2d_BN(in_features=self.embed_dim//2, out_features=self.embed_dim, kernel_size=3, stride=2, padding=1),
        #                     # nn.ReLU(inplace=True)
        #                     )

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.stem(x)
        return x

class SlenderViT(nn.Module):
    """SlenderViT
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[128,320,512], key_dim=32, num_heads=[2,4,8], attn_ratio=2, mlp_ratio=4, mv2_ratio=2,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths_mv2=[2,2,1], depths_trans=[1,4,3], depths_mixer=[2,2,2], pool_ratios=[4,2,1]):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.depths_mv2 = depths_mv2
        self.depths_trans = depths_trans
        self.depths_mixer = depths_mixer
        self.num_stages = len(self.embed_dims)

        self.merging_blocks = nn.ModuleList()
        self.slendervit_blocks = nn.ModuleList()

        self.patch_embed = PatchEmbed(img_size=img_size, in_chans=in_chans, embed_dim=self.embed_dims[0])
        self.merging_blocks.append(self.patch_embed)
        for i in range(self.num_stages-1):
            self.merging_blocks.append(PatchMerging(in_features=self.embed_dims[i], out_features=self.embed_dims[i+1]))

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths_trans))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):

            self.slendervit_block = SlenderViTBlock(depth_mv2= self.depths_mv2[i], depth_trans=self.depths_trans[i], depth_mixer=self.depths_mixer[i], dim=self.embed_dims[i], key_dim=key_dim, num_heads=self.num_heads[i],
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_paths=self.dpr[cur: cur + self.depths_trans[i]], pool_ratio=pool_ratios[i], mv2_ratio=mv2_ratio) 
            cur += self.depths_trans[i]
            self.slendervit_blocks.append(self.slendervit_block)
        
        # classification head
        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)  # Final norm layer
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.merging_blocks[i](x)
            x = self.slendervit_blocks[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x

#depths=[2, 4, 4, 2], [32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 512], dim=48
#embed_dims=[96, 192, 288], key_dim=16, num_heads=[3, 6, 9],
#embed_dims=[96, 192, 288], key_dim=16, num_heads=[3, 6, 9], ***
#embed_dims=[128, 192, 288], key_dim=16, num_heads=[3, 6, 9] (), 
#embed_dims=[128, 256, 384], key_dim=16, num_heads=[4, 8, 12],***

#embed_dims=[192, 288, 384], key_dim=32, num_heads=[4, 6, 8], ***
#embed_dims=[192, 320, 384], key_dim=32, num_heads=[4, 7, 8], ***
#embed_dims=[256, 384, 512], key_dim=32, num_heads=[4, 6, 8],

#embed_dims=[128, 192, 256], key_dim=16, num_heads=[3, 6, 8],
#embed_dims=[128, 224, 288], key_dim=16, num_heads=[3, 7, 9],

#embed_dims=[128, 256, 384], key_dim=16, num_heads=[4, 8, 12], 
#embed_dims=[128, 256, 384], key_dim=32, num_heads=[3, 6, 9],

#embed_dims=[96, 192, 320], key_dim=16, num_heads=[3, 6, 9], depths_trans=[2, 3, 3],
# embed_dims=[96, 192, 288], key_dim=16, num_heads=[2, 4, 8], mv2_ratio=2,  attn_ratio=2, 
# embed_dims=[96, 160, 288], key_dim=16, num_heads=[3, 5, 6], mv2_ratio=2,  attn_ratio=1
# embed_dims=[64, 128, 256], key_dim=16, num_heads=[2, 4, 8], mv2_ratio=2,  attn_ratio=2,

@register_model
def SlenderViT_XXS(pretrained=False, **kwargs):  #SBCFormer-XS
    model = SlenderViT(
        img_size=224, embed_dims=[96, 160, 288], key_dim=16, num_heads=[3, 5, 6], mv2_ratio=2, attn_ratio=2, pool_ratios = [4, 2, 1],
        depths_mv2=[2, 2, 1], depths_trans=[2, 3, 2], depths_mixer=[2, 2, 2], **kwargs)

    model.default_cfg = _cfg()
    return model

#embed_dims=[96, 192, 228], key_dim=16, num_heads=[3, 6, 9], mv2_ratio=2, attn_ratio=1, 
@register_model
def SlenderViT_XS(pretrained=False, **kwargs):  #SBCFormer-S
    model = SlenderViT(
        img_size=224, embed_dims=[96, 192, 320], key_dim=16, num_heads=[3, 5, 7], mv2_ratio=2, attn_ratio=2, pool_ratios = [4, 2, 1],
        depths_mv2=[2, 2, 1], depths_trans=[2, 4, 3], depths_mixer=[2, 2, 2], **kwargs)

    model.default_cfg = _cfg()
    return model

#embed_dims=[128, 256, 384], key_dim=24, num_heads=[4, 6, 8],
# embed_dims=[128, 256, 384], key_dim=16, num_heads=[4, 8, 12],
@register_model
def SlenderViT_S(pretrained=False, **kwargs):   #SBCFormer-B
    model = SlenderViT(
        img_size=224,  embed_dims=[128, 256, 384], key_dim=24, num_heads=[4, 6, 8], mv2_ratio=2, attn_ratio=2, pool_ratios = [4, 2, 1],
        depths_mv2=[2, 2, 1], depths_trans=[2, 4, 3], depths_mixer=[2, 2, 2], **kwargs)

    model.default_cfg = _cfg()
    return model


# embed_dims=[192, 288, 384], key_dim=32, num_heads=[3, 5, 6], mv2_ratio=4, attn_ratio=2,
# embed_dims=[192, 288, 384], key_dim=32, num_heads=[4, 6, 8], mv2_ratio=4, attn_ratio=2,
@register_model
def SlenderViT_B(pretrained=False, **kwargs):  #SBCFormer-L
    model = SlenderViT(
        img_size=224, embed_dims=[192, 288, 384], key_dim=32, num_heads=[4, 6, 8], mv2_ratio=4, attn_ratio=2, pool_ratios = [4, 2, 1],
        depths_mv2=[2, 2, 1], depths_trans=[2, 4, 3], depths_mixer=[2, 2, 2], **kwargs)

    model.default_cfg = _cfg()
    return model

#embed_dims=[224, 320, 416], key_dim=32, num_heads=[4, 6, 8], mv2_ratio=2, attn_ratio=2, R
#embed_dims=[224, 320, 448], key_dim=32, num_heads=[4, 6, 8], mv2_ratio=4, attn_ratio=2, R
#embed_dims=[224, 320, 448], key_dim=32, num_heads=[4, 6, 8], mv2_ratio=2, attn_ratio=2, 
# embed_dims=[224, 320, 448], key_dim=32, num_heads=[5, 7, 9], mv2_ratio=2, attn_ratio=2, 
# embed_dims=[256, 384, 512], key_dim=48, num_heads=[4, 6, 8], mv2_ratio=4,
#
@register_model
def SlenderViT_L(pretrained=False, **kwargs):
    model = SlenderViT(
        img_size=224, embed_dims=[256, 384, 512], key_dim=48, num_heads=[5, 7, 9], mv2_ratio=4, attn_ratio=2, pool_ratios = [4, 2, 1],
        depths_mv2=[2, 2, 1], depths_trans=[2, 4, 3], depths_mixer=[2, 2, 2], **kwargs)

    model.default_cfg = _cfg()
    return model


#embed_dims=[256, 384, 512], key_dim=32, num_heads=[6, 9, 12], mv2_ratio=2, attn_ratio=2,
#embed_dims=[256, 384, 512], key_dim=32, num_heads=[6, 9, 12], mv2_ratio=4, attn_ratio=2,
#embed_dims=[256, 384, 512], key_dim=64, num_heads=[4, 6, 8], mv2_ratio=4, 
# embed_dims=[384, 512, 768], key_dim=64, num_heads=[4, 6, 8], mv2_ratio=2, 
#384_512_768
@register_model
def SlenderViT_XL(pretrained=False, **kwargs):
    model = SlenderViT(
        img_size=224, embed_dims=[256, 384, 512], key_dim=48, num_heads=[6, 9, 12], mv2_ratio=4, attn_ratio=2, pool_ratios = [4, 2, 1],
        depths_mv2=[2, 2, 1], depths_trans=[2, 4, 3], depths_mixer=[2, 2, 2], **kwargs)

    model.default_cfg = _cfg()
    return model
