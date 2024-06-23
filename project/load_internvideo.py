# from intern_action import intern_action_b16
from huggingface_hub import hf_hub_download
# from kinetics_class_index import kinetics_classnames
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

# from transforms import (
#     GroupNormalize, GroupScale, GroupCenterCrop, 
#     Stack, ToTorchFormatTensor
# )
import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class MultiGroupRandomCrop(object):
    def __init__(self, size, groups=1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.groups = groups

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        for i in range(self.groups):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            for img in img_group:
                assert(img.size[0] == w and img.size[1] == h)
                if w == tw and h == th:
                    out_images.append(img)
                else:
                    out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop(
                (offset_w,
                 offset_h,
                 offset_w +
                 crop_w,
                 offset_h +
                 crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(
                x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [
            self.input_size[0] if abs(
                x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(
                    img.resize(
                        (self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class ConvertDataFormat(object):
    def __init__(self, model_type):
        self.model_type = model_type

    def __call__(self, images):
        if self.model_type == '2D':
            return images
        tc, h, w = images.size()
        t = tc // 3
        images = images.view(t, 3, h, w)
        images = images.permute(1, 0, 2, 3)
        return images


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2)
                                   for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1]
                                       for x in img_group], axis=2)
            else:
                #print(np.concatenate(img_group, axis=2).shape)
                # print(img_group[0].shape)
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


#!/usr/bin/env python
import os
from collections import OrderedDict

from timm.models.layers import DropPath
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


MODEL_PATH = './'
_MODELS = {
    "ViT-B/16": os.path.join(MODEL_PATH, "vit_b16.pth"),
    "ViT-L/14": os.path.join(MODEL_PATH, "vit_l14.pth"),
    "ViT-L/14_336": os.path.join(MODEL_PATH, "vit_l14_336.pth"),
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Local_MHRA(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__() 

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), stride=(1, 1, 1), padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        # print('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, d_model, n_head, attn_mask=None, drop_path=0.0, 
            dw_reduction=1.5, no_lmhra=False, double_lmhra=True
        ):
        super().__init__() 
        
        self.n_head = n_head
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Drop path rate: {drop_path}')

        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra
        # print(f'No L_MHRA: {no_lmhra}')
        # print(f'Double L_MHRA: {double_lmhra}')
        if not no_lmhra:
            self.lmhra1 = Local_MHRA(d_model, dw_reduction=dw_reduction)
            if double_lmhra:
                self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, T=8, use_checkpoint=False):
        # x: 1+HW, NT, C
        if not self.no_lmhra:
            # Local MHRA
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape
            N = NT // T
            H = W = int(L ** 0.5)
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # MHSA
        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1(x))
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1(x)))
        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        if use_checkpoint:
            mlp_out = checkpoint.checkpoint(self.mlp, self.ln_2(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Extractor(nn.Module):
    def __init__(
            self, d_model, n_head, attn_mask=None,
            mlp_factor=4.0, dropout=0.0, drop_path=0.0,
        ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_mlp, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x, y):
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim ** 0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x, y):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self, width, layers, heads, attn_mask=None, backbone_drop_path_rate=0., 
            use_checkpoint=False, checkpoint_num=[0], t_size=8, dw_reduction=2,
            no_lmhra=False, double_lmhra=True,
            return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            n_layers=12, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
            mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
            cls_dropout=0.5, num_classes=400,
        ):
        super().__init__()
        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [x.item() for x in torch.linspace(0, backbone_drop_path_rate, layers)]
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, attn_mask, 
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])
        # checkpoint
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.n_layers = n_layers
        # print(f'Use checkpoint: {self.use_checkpoint}')
        # print(f'Checkpoint number: {self.checkpoint_num}')

        # global block
        assert n_layers == len(return_list)
        if n_layers > 0:
            self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
            self.dpe = nn.ModuleList([
                nn.Conv3d(n_dim, n_dim, kernel_size=3, stride=1, padding=1, bias=True, groups=n_dim)
                for i in range(n_layers)
            ])
            for m in self.dpe:
                nn.init.constant_(m.bias, 0.)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
            self.dec = nn.ModuleList([
                Extractor(
                    n_dim, n_head, mlp_factor=mlp_factor, 
                    dropout=mlp_dropout[i], drop_path=dpr[i],
                ) for i in range(n_layers)
            ])
            self.balance = nn.Parameter(torch.zeros((n_dim)))
            self.sigmoid = nn.Sigmoid()
        # projection
        self.proj = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(n_dim, num_classes),
        )

    def forward(self, x):
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1) ** 0.5)

        if self.n_layers > 0:
            cls_token = self.temporal_cls_token.repeat(1, N, 1)

        j = -1
        for i, resblock in enumerate(self.resblocks):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = resblock(x, self.T, use_checkpoint=True)
            else:
                x = resblock(x, T_down)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2, 0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats).view(N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x)

        if self.n_layers > 0:
            weight = self.sigmoid(self.balance)
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
            return self.proj((1 - weight) * cls_token[0, :, :] + weight * residual)
        else:
            residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
            return self.proj(residual)


class VisionTransformer(nn.Module):
    def __init__(
        self, 
        # backbone
        input_resolution, patch_size, width, layers, heads, output_dim, backbone_drop_path_rate=0.,
        use_checkpoint=False, checkpoint_num=[0], t_size=8, kernel_size=3, dw_reduction=1.5,
        temporal_downsample=True,
        no_lmhra=-False, double_lmhra=True,
        # global block
        return_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        n_layers=12, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
        cls_dropout=0.5, num_classes=400,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(3, width, (kernel_size, patch_size, patch_size), (2, patch_size, patch_size), (padding, 0, 0), bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(3, width, (1, patch_size, patch_size), (1, patch_size, patch_size), (0, 0, 0), bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(
            width, layers, heads, dw_reduction=dw_reduction, 
            backbone_drop_path_rate=backbone_drop_path_rate, 
            use_checkpoint=use_checkpoint, checkpoint_num=checkpoint_num, t_size=t_size,
            no_lmhra=no_lmhra, double_lmhra=double_lmhra,
            return_list=return_list, n_layers=n_layers, n_dim=n_dim, n_head=n_head, 
            mlp_factor=mlp_factor, drop_path_rate=drop_path_rate, mlp_dropout=mlp_dropout, 
            cls_dropout=cls_dropout, num_classes=num_classes,
        )

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x)
        return out


def inflate_weight(weight_2d, time_dim, center=True):
    # print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                # print(f'Ignore: {k}')
                continue
            # print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim)
    model.load_state_dict(state_dict, strict=False)


def intern_action_b16(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[8, 9, 10, 11], 
    n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
    )

    if pretrained:
        # print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/16"], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


def intern_action_l14(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[20, 21, 22, 23],
    n_layers=4, n_dim=1024, n_head=16, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
    )

    if pretrained:
        # print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14"], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


def intern_action_l14_336(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    no_temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[20, 21, 22, 23],
    n_layers=4, n_dim=1024, n_head=16, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
):
    model = VisionTransformer(
        input_resolution=336,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        no_temporal_downsample=no_temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
    )

    if pretrained:
        # print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14_336"], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()



class Intern_Action(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
    
    def forward(self, x):
        return self.backbone(x)

def get_index(num_frames, num_segments=8):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def transform_action():
    # transform
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    return T.Compose([
        # T.ToPILImage(),
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

def load_intern_action(device):
    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[k] = v

    # model_path = hf_hub_download(repo_id="Andy1621/uniformerv2", filename="k400+k710_uniformerv2_b16_8x224.pyth")
    model_path = "./checkpoints/k400+k710_uniformerv2_b16_8x224.pyth"
    # Pick a pretrained model 
    model = Intern_Action(intern_action_b16(pretrained=False, t_size=8, no_lmhra=True, temporal_downsample=False))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()
    return model

def cut_frame_to_8(data):
    index = np.linspace(0, len(data)-1, 8).astype(int)
    return data[index]

kinetics_classnames = {
    "0": "riding a bike", 
    "1": "marching", 
    "2": "dodgeball", 
    "3": "playing cymbals", 
    "4": "checking tires", 
    "5": "roller skating", 
    "6": "tasting beer", 
    "7": "clapping", 
    "8": "drawing", 
    "9": "juggling fire", 
    "10": "bobsledding", 
    "11": "petting animal (not cat)", 
    "12": "spray painting", 
    "13": "training dog", 
    "14": "eating watermelon", 
    "15": "building cabinet", 
    "16": "applauding", 
    "17": "playing harp", 
    "18": "balloon blowing", 
    "19": "sled dog racing", 
    "20": "wrestling", 
    "21": "pole vault", 
    "22": "hurling (sport)", 
    "23": "riding scooter", 
    "24": "shearing sheep", 
    "25": "sweeping floor", 
    "26": "eating carrots", 
    "27": "skateboarding", 
    "28": "dunking basketball", 
    "29": "disc golfing", 
    "30": "eating spaghetti", 
    "31": "playing flute", 
    "32": "riding mechanical bull", 
    "33": "making sushi", 
    "34": "trapezing", 
    "35": "picking fruit", 
    "36": "stretching leg", 
    "37": "playing ukulele", 
    "38": "tying tie", 
    "39": "skydiving", 
    "40": "playing cello", 
    "41": "jumping into pool", 
    "42": "shooting goal (soccer)", 
    "43": "trimming trees", 
    "44": "bookbinding", 
    "45": "ski jumping", 
    "46": "walking the dog", 
    "47": "riding unicycle", 
    "48": "shaving head", 
    "49": "hopscotch", 
    "50": "playing piano", 
    "51": "parasailing", 
    "52": "bartending", 
    "53": "kicking field goal", 
    "54": "finger snapping", 
    "55": "dining", 
    "56": "yawning", 
    "57": "peeling potatoes", 
    "58": "canoeing or kayaking", 
    "59": "front raises", 
    "60": "laughing", 
    "61": "dancing macarena", 
    "62": "digging", 
    "63": "reading newspaper", 
    "64": "hitting baseball", 
    "65": "clay pottery making", 
    "66": "exercising with an exercise ball", 
    "67": "playing saxophone", 
    "68": "shooting basketball", 
    "69": "washing hair", 
    "70": "lunge", 
    "71": "brushing hair", 
    "72": "curling hair", 
    "73": "kitesurfing", 
    "74": "tapping guitar", 
    "75": "bending back", 
    "76": "skipping rope", 
    "77": "situp", 
    "78": "folding paper", 
    "79": "cracking neck", 
    "80": "assembling computer", 
    "81": "cleaning gutters", 
    "82": "blowing out candles", 
    "83": "shaking hands", 
    "84": "dancing gangnam style", 
    "85": "windsurfing", 
    "86": "tap dancing", 
    "87": "skiing (not slalom or crosscountry)", 
    "88": "bandaging", 
    "89": "push up", 
    "90": "doing nails", 
    "91": "punching person (boxing)", 
    "92": "bouncing on trampoline", 
    "93": "scrambling eggs", 
    "94": "singing", 
    "95": "cleaning floor", 
    "96": "krumping", 
    "97": "drumming fingers", 
    "98": "snowmobiling", 
    "99": "gymnastics tumbling", 
    "100": "headbanging", 
    "101": "catching or throwing frisbee", 
    "102": "riding elephant", 
    "103": "bee keeping", 
    "104": "feeding birds", 
    "105": "snatch weight lifting", 
    "106": "mowing lawn", 
    "107": "fixing hair", 
    "108": "playing trumpet", 
    "109": "flying kite", 
    "110": "crossing river", 
    "111": "swinging legs", 
    "112": "sanding floor", 
    "113": "belly dancing", 
    "114": "sneezing", 
    "115": "clean and jerk", 
    "116": "side kick", 
    "117": "filling eyebrows", 
    "118": "shuffling cards", 
    "119": "recording music", 
    "120": "cartwheeling", 
    "121": "feeding fish", 
    "122": "folding clothes", 
    "123": "water skiing", 
    "124": "tobogganing", 
    "125": "blowing leaves", 
    "126": "smoking", 
    "127": "unboxing", 
    "128": "tai chi", 
    "129": "waxing legs", 
    "130": "riding camel", 
    "131": "slapping", 
    "132": "tossing salad", 
    "133": "capoeira", 
    "134": "playing cards", 
    "135": "playing organ", 
    "136": "playing violin", 
    "137": "playing drums", 
    "138": "tapping pen", 
    "139": "vault", 
    "140": "shoveling snow", 
    "141": "playing tennis", 
    "142": "getting a tattoo", 
    "143": "making a sandwich", 
    "144": "making tea", 
    "145": "grinding meat", 
    "146": "squat", 
    "147": "eating doughnuts", 
    "148": "ice fishing", 
    "149": "snowkiting", 
    "150": "kicking soccer ball", 
    "151": "playing controller", 
    "152": "giving or receiving award", 
    "153": "welding", 
    "154": "throwing discus", 
    "155": "throwing axe", 
    "156": "ripping paper", 
    "157": "swimming butterfly stroke", 
    "158": "air drumming", 
    "159": "blowing nose", 
    "160": "hockey stop", 
    "161": "taking a shower", 
    "162": "bench pressing", 
    "163": "planting trees", 
    "164": "pumping fist", 
    "165": "climbing tree", 
    "166": "tickling", 
    "167": "high kick", 
    "168": "waiting in line", 
    "169": "slacklining", 
    "170": "tango dancing", 
    "171": "hurdling", 
    "172": "carrying baby", 
    "173": "celebrating", 
    "174": "sharpening knives", 
    "175": "passing American football (in game)", 
    "176": "headbutting", 
    "177": "playing recorder", 
    "178": "brush painting", 
    "179": "garbage collecting", 
    "180": "robot dancing", 
    "181": "shredding paper", 
    "182": "pumping gas", 
    "183": "rock climbing", 
    "184": "hula hooping", 
    "185": "braiding hair", 
    "186": "opening present", 
    "187": "texting", 
    "188": "decorating the christmas tree", 
    "189": "answering questions", 
    "190": "playing keyboard", 
    "191": "writing", 
    "192": "bungee jumping", 
    "193": "sniffing", 
    "194": "eating burger", 
    "195": "playing accordion", 
    "196": "making pizza", 
    "197": "playing volleyball", 
    "198": "tasting food", 
    "199": "pushing cart", 
    "200": "spinning poi", 
    "201": "cleaning windows", 
    "202": "arm wrestling", 
    "203": "changing oil", 
    "204": "swimming breast stroke", 
    "205": "tossing coin", 
    "206": "deadlifting", 
    "207": "hoverboarding", 
    "208": "cutting watermelon", 
    "209": "cheerleading", 
    "210": "snorkeling", 
    "211": "washing hands", 
    "212": "eating cake", 
    "213": "pull ups", 
    "214": "surfing water", 
    "215": "eating hotdog", 
    "216": "holding snake", 
    "217": "playing harmonica", 
    "218": "ironing", 
    "219": "cutting nails", 
    "220": "golf chipping", 
    "221": "shot put", 
    "222": "hugging", 
    "223": "playing clarinet", 
    "224": "faceplanting", 
    "225": "trimming or shaving beard", 
    "226": "drinking shots", 
    "227": "riding mountain bike", 
    "228": "tying bow tie", 
    "229": "swinging on something", 
    "230": "skiing crosscountry", 
    "231": "unloading truck", 
    "232": "cleaning pool", 
    "233": "jogging", 
    "234": "ice climbing", 
    "235": "mopping floor", 
    "236": "making bed", 
    "237": "diving cliff", 
    "238": "washing dishes", 
    "239": "grooming dog", 
    "240": "weaving basket", 
    "241": "frying vegetables", 
    "242": "stomping grapes", 
    "243": "moving furniture", 
    "244": "cooking sausages", 
    "245": "doing laundry", 
    "246": "dying hair", 
    "247": "knitting", 
    "248": "reading book", 
    "249": "baby waking up", 
    "250": "punching bag", 
    "251": "surfing crowd", 
    "252": "cooking chicken", 
    "253": "pushing car", 
    "254": "springboard diving", 
    "255": "swing dancing", 
    "256": "massaging legs", 
    "257": "beatboxing", 
    "258": "breading or breadcrumbing", 
    "259": "somersaulting", 
    "260": "brushing teeth", 
    "261": "stretching arm", 
    "262": "juggling balls", 
    "263": "massaging person's head", 
    "264": "eating ice cream", 
    "265": "extinguishing fire", 
    "266": "hammer throw", 
    "267": "whistling", 
    "268": "crawling baby", 
    "269": "using remote controller (not gaming)", 
    "270": "playing cricket", 
    "271": "opening bottle", 
    "272": "playing xylophone", 
    "273": "motorcycling", 
    "274": "driving car", 
    "275": "exercising arm", 
    "276": "passing American football (not in game)", 
    "277": "playing kickball", 
    "278": "sticking tongue out", 
    "279": "flipping pancake", 
    "280": "catching fish", 
    "281": "eating chips", 
    "282": "shaking head", 
    "283": "sword fighting", 
    "284": "playing poker", 
    "285": "cooking on campfire", 
    "286": "doing aerobics", 
    "287": "paragliding", 
    "288": "using segway", 
    "289": "folding napkins", 
    "290": "playing bagpipes", 
    "291": "gargling", 
    "292": "skiing slalom", 
    "293": "strumming guitar", 
    "294": "javelin throw", 
    "295": "waxing back", 
    "296": "riding or walking with horse", 
    "297": "plastering", 
    "298": "long jump", 
    "299": "parkour", 
    "300": "wrapping present", 
    "301": "egg hunting", 
    "302": "archery", 
    "303": "cleaning toilet", 
    "304": "swimming backstroke", 
    "305": "snowboarding", 
    "306": "catching or throwing baseball", 
    "307": "massaging back", 
    "308": "blowing glass", 
    "309": "playing guitar", 
    "310": "playing chess", 
    "311": "golf driving", 
    "312": "presenting weather forecast", 
    "313": "rock scissors paper", 
    "314": "high jump", 
    "315": "baking cookies", 
    "316": "using computer", 
    "317": "washing feet", 
    "318": "arranging flowers", 
    "319": "playing bass guitar", 
    "320": "spraying", 
    "321": "cutting pineapple", 
    "322": "waxing chest", 
    "323": "auctioning", 
    "324": "jetskiing", 
    "325": "drinking", 
    "326": "busking", 
    "327": "playing monopoly", 
    "328": "salsa dancing", 
    "329": "waxing eyebrows", 
    "330": "watering plants", 
    "331": "zumba", 
    "332": "chopping wood", 
    "333": "pushing wheelchair", 
    "334": "carving pumpkin", 
    "335": "building shed", 
    "336": "making jewelry", 
    "337": "catching or throwing softball", 
    "338": "bending metal", 
    "339": "ice skating", 
    "340": "dancing charleston", 
    "341": "abseiling", 
    "342": "climbing a rope", 
    "343": "crying", 
    "344": "cleaning shoes", 
    "345": "dancing ballet", 
    "346": "driving tractor", 
    "347": "triple jump", 
    "348": "throwing ball", 
    "349": "getting a haircut", 
    "350": "running on treadmill", 
    "351": "climbing ladder", 
    "352": "blasting sand", 
    "353": "playing trombone", 
    "354": "drop kicking", 
    "355": "country line dancing", 
    "356": "changing wheel", 
    "357": "feeding goats", 
    "358": "tying knot (not on a tie)", 
    "359": "setting table", 
    "360": "shaving legs", 
    "361": "kissing", 
    "362": "riding mule", 
    "363": "counting money", 
    "364": "laying bricks", 
    "365": "barbequing", 
    "366": "news anchoring", 
    "367": "smoking hookah", 
    "368": "cooking egg", 
    "369": "peeling apples", 
    "370": "yoga", 
    "371": "sharpening pencil", 
    "372": "dribbling basketball", 
    "373": "petting cat", 
    "374": "playing ice hockey", 
    "375": "milking cow", 
    "376": "shining shoes", 
    "377": "juggling soccer ball", 
    "378": "scuba diving", 
    "379": "playing squash or racquetball", 
    "380": "drinking beer", 
    "381": "sign language interpreting", 
    "382": "playing basketball", 
    "383": "breakdancing", 
    "384": "testifying", 
    "385": "making snowman", 
    "386": "golf putting", 
    "387": "playing didgeridoo", 
    "388": "biking through snow", 
    "389": "sailing", 
    "390": "jumpstyle dancing", 
    "391": "water sliding", 
    "392": "grooming horse", 
    "393": "massaging feet", 
    "394": "playing paintball", 
    "395": "making a cake", 
    "396": "bowling", 
    "397": "contact juggling", 
    "398": "applying cream", 
    "399": "playing badminton"
}

