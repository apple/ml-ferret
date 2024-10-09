#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from ferretui.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX,
                                DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IM_END_TOKEN, DEFAULT_REGION_FEA_TOKEN)

from ferretui.mm_utils import get_anyres_image_grid_shape

import os

def rand_sample(x, max_len):
    if x.shape[0] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
    return x[rand_idx, :]


def rand_sample_repeat(x, max_len):
    if x.shape[0] < max_len:
        indices = torch.randint(0, x.shape[0], (max_len-x.shape[0],))
        # pdb.set_trace()
        return torch.cat((x, x[indices]), dim=0)
    elif x.shape[0] == max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
        return x[rand_idx, :]
    

def point_sample(input, point_coords, return_dtype, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float(), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class ConvReLULN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvReLULN1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            self.act
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # (B, C, N) -> (B, C_1, N)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        
        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class GeoRegionSampler(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_init_point,
                 num_sub_point,
                 num_neighbor,
                 pooler_mode='mean'):
        super(GeoRegionSampler, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_init_point = num_init_point
        self.num_sub_point = num_sub_point
        self.num_neighbor = num_neighbor

        self.diff_projector_list = nn.ModuleList()
        self.agg_projector_list = nn.ModuleList()
        self.pooler_list = nn.ModuleList()

        for ii in range(len(num_sub_point)):
            self.diff_projector_list.append(nn.Linear(self.input_dim + 2, self.input_dim + 2))
            self.agg_projector_list.append(ConvReLULN1D(in_channels=2*(self.input_dim + 2),
                                                        out_channels=self.input_dim,
                                                        ))
            if pooler_mode == 'mean':
                self.pooler_list.append(nn.AvgPool1d(kernel_size=num_neighbor[ii]))
            elif pooler_mode =='max':
                self.pooler_list.append(nn.AdaptiveMaxPool1d(output_size=1))
            else:
                raise NotImplementedError(f'{self.pooler_mode} is not supported.')

        self.flatten_projector = nn.Linear(self.input_dim * num_sub_point[-1], self.input_dim)
        self.dim_projector = nn.Linear(self.input_dim, self.output_dim)
        # self.dim_projector = nn.Sequential(*[
        #     nn.Linear(self.input_dim, self.output_dim),
        #     nn.GELU(),
        #     nn.Linear(self.output_dim, self.output_dim)
        # ])

        self.norm_init_weights()

    #  self.dtype = torch.float32
    def norm_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)


    def forward(self, 
                feature_map, 
                region_masks, 
                original_dtype,
                return_dtype):

        assert len(feature_map) == len(region_masks)

        all_points = []
        all_points_fea = []
        all_points_img_ids = []

        # Sample points and their features
        for img_idx, (region_feature_map_i, region_masks_list_i) in enumerate(zip(feature_map, region_masks)):
            if len(region_masks_list_i) != 0:
                # (w, h)
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]], device=region_masks_list_i[0].device)[None,]
                # list of elements of shape [num_sample_point, 2] 
                cur_non_zero_pos = [rand_sample_repeat((m.nonzero()/ori_image_wh), self.num_init_point) for m in region_masks_list_i]
                # list -> [num_mask, num_sample_point, 2]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos)
                # [HxW, C] -> [H, W, C] -> [C, H, W] -> [N, C, H, W]
                if region_feature_map_i.ndim == 2:
                    h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                    c = region_feature_map_i.shape[-1]
                    region_feature_map_i = region_feature_map_i.reshape(h, w, c)
                else:
                    assert region_feature_map_i.ndim == 3
                dup_region_feature_map_i = region_feature_map_i.permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(cur_non_zero_pos.shape[0], 1, 1, 1)
                # [num_mask, C, H, W] x [num_mask, num_sample_point, 2] -> [num_mask, C, num_sample_point] -> [num_mask, num_sample_point, C]
                # F.grid_sample doesn't support BF16. Need to tranform into float32 then transform back.
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                region_feature_i = point_sample(dup_region_feature_map_i_ori_type, 
                                                cur_non_zero_pos.flip(dims=(2,)).type(original_dtype), 
                                                return_dtype,
                                                align_corners=True,
                                                )
                # region_feature_i = region_feature_i.to(dup_region_feature_map_i.dtype)
                region_feature_i = region_feature_i.transpose(-2, -1)

                cur_img_ids = [img_idx] * len(cur_non_zero_pos)
                # save to global list
                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_ids)

        # No region found, return list of None.
        if len(all_points) == 0:
            return [None] * len(region_masks)
        
        all_points = torch.cat(all_points, dim=0).to(return_dtype)  # [B*num_mask, num_sample_point, 2]
        all_points_fea = torch.cat(all_points_fea, dim=0)  # [B*num_mask, num_sample_point, C]
        all_points_img_ids = torch.tensor(all_points_img_ids, device=all_points_fea.device)

        assert all_points_fea.shape[:-1] == all_points_fea.shape[:-1]
        
        # Processing.
        for stage_i in range(len(self.num_sub_point)):
            cur_num_sub_point = self.num_sub_point[stage_i]
            cur_num_neighbor = self.num_neighbor[stage_i]

            all_points = all_points.contiguous()  # xy [btach, points, xy]
            fps_idx = farthest_point_sample(all_points, cur_num_sub_point).long()

            new_points = index_points(all_points, fps_idx)  # [B, npoint, 2]
            new_points_fea = index_points(all_points_fea, fps_idx)  # [B, npoint, d]

            idx = knn_point(cur_num_neighbor, all_points, new_points)
            grouped_points = index_points(all_points, idx)  # [B, npoint, k, 2]
            grouped_points_fea = index_points(all_points_fea, idx)  # [B, npoint, k, d]

            local_points_fea = torch.cat([grouped_points_fea, grouped_points],dim=-1)  # [B, npoint, k, d+2]
            anchor_points_fea = torch.cat([new_points_fea, new_points],dim=-1).unsqueeze(-2)
            diff_points_fea = local_points_fea-anchor_points_fea

            diff_points_fea = self.diff_projector_list[stage_i](diff_points_fea)
            gather_points_fea = torch.cat([diff_points_fea, anchor_points_fea.repeat(1, 1, cur_num_neighbor, 1)], dim=-1)  # [B, npoint, k, 2(d+2)]

            b, n, s, d = gather_points_fea.size() 
            gather_points_fea = gather_points_fea.permute(0, 1, 3, 2)   # [B, npoint, 2(d+2), k]
            gather_points_fea = gather_points_fea.reshape(-1, d, s)   # [B*npoint, 2(d+2), k]
            gather_points_fea = self.agg_projector_list[stage_i](gather_points_fea) # [B*npoint, d, k]

            batch_size, new_dim, _ = gather_points_fea.size()
            gather_points_fea = self.pooler_list[stage_i](gather_points_fea).view(batch_size, new_dim) # [B*npoint, d]

            gather_points_fea = gather_points_fea.reshape(b, n, -1)     # [B, npoint, d]

            all_points = new_points
            all_points_fea = gather_points_fea

        x = all_points_fea.flatten(1, -1)  # [B, npoint x d]
        x = self.flatten_projector(x)
        all_region_fea = self.dim_projector(x)  # [B, d]

        output_region_fea = []
        for img_idx in range(len(region_masks)):
            cur_mask = all_points_img_ids == img_idx

            if not cur_mask.any():
                output_region_fea.append(None)
            else:
                output_region_fea.append(all_region_fea[cur_mask])

        return output_region_fea


class FerretMetaModel:

    def __init__(self, config):
        super(FerretMetaModel, self).__init__(config)
        self.max_sample_point = 512
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
        
        if hasattr(config, "region_fea_adapter"):
            self.region_fea_adapter = nn.Linear(config.mm_hidden_size, config.hidden_size)

        if hasattr(config, "region_geo_sampler"):
            if getattr(config, 'mm_patch_merge_type', 'flat').startswith('spatial'):
                self.region_geo_sampler = GeoRegionSampler(input_dim=config.mm_hidden_size,
                                                        output_dim=config.hidden_size,
                                                        num_init_point=self.max_sample_point,
                                                        num_sub_point=[128, 32],
                                                        num_neighbor=[24, 24],
                                                        pooler_mode=config.sampler_pooler_mode
                                                        )
            else:
                self.region_geo_sampler = GeoRegionSampler(input_dim=config.mm_hidden_size,
                                                        output_dim=config.hidden_size,
                                                        num_init_point=self.max_sample_point,
                                                        num_sub_point=[128, 32],
                                                        num_neighbor=[24, 24],
                                                        pooler_mode=config.sampler_pooler_mode
                                                        )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None,
                                  add_region_feature=False, 
                                  region_geo_sampler=False, 
                                  sampler_pooler_mode='mean',
                                  ):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if add_region_feature:
            if region_geo_sampler:
                self.config.region_geo_sampler = True
                self.config.sampler_pooler_mode = sampler_pooler_mode
                
                if not hasattr(self, 'region_geo_sampler'):
                    if mm_patch_merge_type.startswith('spatial'):
                        # === if feature is concated ===
                        # self.region_geo_sampler = GeoRegionSampler(input_dim=self.config.mm_hidden_size * 2,
                        #                                         output_dim=self.config.hidden_size,
                        #                                         num_init_point=self.max_sample_point,
                        #                                         num_sub_point=[128, 32],
                        #                                         num_neighbor=[24, 24],
                        #                                         pooler_mode=sampler_pooler_mode
                        #                                         )
                        # === if feature is added ===
                        self.region_geo_sampler = GeoRegionSampler(input_dim=self.config.mm_hidden_size,
                                                                output_dim=self.config.hidden_size,
                                                                num_init_point=self.max_sample_point,
                                                                num_sub_point=[128, 32],
                                                                num_neighbor=[24, 24],
                                                                pooler_mode=sampler_pooler_mode
                                                                )
                    else:
                        self.region_geo_sampler = GeoRegionSampler(input_dim=self.config.mm_hidden_size,
                                                                output_dim=self.config.hidden_size,
                                                                num_init_point=self.max_sample_point,
                                                                num_sub_point=[128, 32],
                                                                num_neighbor=[24, 24],
                                                                pooler_mode=sampler_pooler_mode
                                                                )
            else:
                self.config.region_fea_adapter = True
                if not hasattr(self, 'region_fea_adapter'):
                    self.region_fea_adapter = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # print(f"pretrain mm mlp adapter: {type(pretrain_mm_mlp_adapter)}") # String
        if pretrain_mm_mlp_adapter is not None and pretrain_mm_mlp_adapter != "None":
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class FerretMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, region_flag=False, region_geo_sampler=False):
        image_features = self.get_model().get_vision_tower()(images)
        projected_image_features = self.get_model().mm_projector(image_features)
        if region_flag:
            if region_geo_sampler:
                new_region_feature_map = image_features
            else:
                new_region_feature_map = self.get_model().region_fea_adapter(image_features)
        else:
            new_region_feature_map = None

        return image_features, projected_image_features, new_region_feature_map

    def extract_region_feature(self, region_feature_map, region_masks, original_dtype, return_dtype):
        all_region_features = []
        assert len(region_feature_map) == len(region_masks)
        for region_feature_map_i, region_masks_list_i in zip(region_feature_map, region_masks):
            if len(region_masks_list_i) == 0:
                all_region_features.append(None)
            else:
                # (w, h)
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]], device=region_masks_list_i[0].device)[None,]
                # list of elements of shape [num_sample_point, 2]
                non_zero_pos = [rand_sample((m.nonzero()/ori_image_wh), self.get_model().max_sample_point) for m in region_masks_list_i]
                # [num_mask, num_sample_point(padded), 2]
                non_zero_pos = nn.utils.rnn.pad_sequence(non_zero_pos, padding_value=-1, batch_first=True)
                non_zero_pos_mask = ~(non_zero_pos.sum(dim=-1) < 0)
                # [HxW, C] -> [H, W, C] -> [C, H, W] -> [N, C, H, W]
                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]
                dup_region_feature_map_i = region_feature_map_i.reshape(h, w, c).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(non_zero_pos.shape[0], 1, 1, 1)
                # [num_mask, C, H, W] x [num_mask, num_sample_point(padded), 2] -> [num_mask, C, num_sample_point(padded)]
                # F.grid_sample doesn't support BF16. Need to tranform into float32 then transform back.
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                # pdb.set_trace()
                region_feature_i = point_sample(dup_region_feature_map_i_ori_type, 
                                                non_zero_pos.flip(dims=(2,)).type(original_dtype), 
                                                return_dtype, 
                                                align_corners=True
                                                )
                region_feature_i = region_feature_i.to(dup_region_feature_map_i.dtype)
                # [num_mask, C]
                region_feature_i = torch.stack([x[m].mean(dim=0) for x, m in zip(region_feature_i.transpose(1,2), non_zero_pos_mask)]).nan_to_num()
                all_region_features.append(region_feature_i)
        
        return all_region_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, region_masks=None
    ):
        if region_masks is not None:
            region_flag = True
        else:
            region_flag = False
        region_geo_sampler = region_flag and getattr(self.config, 'region_geo_sampler', False)

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            
            concat_images = torch.cat([image for image in images], dim=0)
            raw_image_features, image_features, region_feature_map = self.encode_images(concat_images, region_flag=region_flag, region_geo_sampler=region_geo_sampler)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            if region_flag:
                region_feature_maps = torch.split(region_feature_map, split_sizes, dim=0)  #  (#images, #patches, h*w, c)
                # ======== This is for only taking the global image feature map for referring ======
                # region_feature_map = torch.split(region_feature_map, split_sizes, dim=0)
                # first_region_feature_map = [x[0:1] for x in region_feature_map]
                # region_feature_map = torch.cat(first_region_feature_map, dim=0)

            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square_nocrop')

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                # TODO: here we use the first feature map default for each batch (global feaure map) for referring
                first_region_feature_map = [x[0:1] for x in region_feature_map]
                region_feature_map = torch.cat(first_region_feature_map, dim=0) #  (#images, h, w, c)
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                new_region_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if region_flag:
                            cur_region_feature_map = region_feature_maps[image_idx]  # (#patches, h*w, c)
                            cur_region_feature_map = cur_region_feature_map.view(cur_region_feature_map.shape[0], height, width, cur_region_feature_map.shape[-1])  # (#patches, h, w, c)
                            base_region_feature = cur_region_feature_map[0]
                            region_feature = cur_region_feature_map[1:]
                            # pdb.set_trace()
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            if region_flag:
                                region_feature = region_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError

                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        if region_flag:
                            region_feature = region_feature.permute(0, 2, 1, 3, 4).contiguous()   # (patch_h, patch_w, h, w, c) -> (patch_h, h, patch_w, w, c)
                            region_feature = region_feature.flatten(0, 1).flatten(1, 2)   # (patch_h, h, patch_w, w, c) -> (all_h, all_w, c)
                            # Tranform dtype, if using pytorch2.1+, no need to do this.
                            base_region_feature = base_region_feature.to(dtype=torch.float32)
                            base_region_feature_resized = F.interpolate(base_region_feature.unsqueeze(0).permute(0, 3, 1, 2), (region_feature.shape[0], region_feature.shape[1]))  # (1, c, all_h, all_w)
                            base_region_feature_resized = base_region_feature_resized.to(region_feature.dtype)
                            base_region_feature_resized = base_region_feature_resized.squeeze(0).permute(1, 2, 0)   # (all_h, all_w, c)
                            # === Add: 
                            new_region_feature = base_region_feature_resized + region_feature
                            # === Concat: A bit lower, 1/3 more GPU memory consumption.
                            # new_region_feature = torch.cat((base_region_feature_resized, region_feature), dim=2)  # (all_h, all_w, 2c)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        if region_flag:
                            new_region_feature = region_feature_maps[image_idx][0]   # (h, w, c)
                    new_image_features.append(image_feature)
                    if region_flag:
                        new_region_features.append(new_region_feature)
                        # pdb.set_trace()
                image_features = new_image_features
                if region_flag:
                    # region_feature_map = torch.stack(new_region_features, dim=0)  # (#images, h, w, c or 2c)
                    region_feature_map = new_region_features
                    # pdb.set_trace()
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            raw_image_features, image_features, region_feature_map = self.encode_images(images, region_flag=region_flag, region_geo_sampler=region_geo_sampler)
        
        if region_flag:
            assert len(region_masks) == len(input_ids)
            for img_idx, (cur_input_id, cur_region_mask) in enumerate(zip(input_ids, region_masks)):
                cur_region_token_num = (cur_input_id == self.config.im_region_fea_token).sum()
                if cur_region_token_num != len(cur_region_mask):
                    print('Found regions cropped because of text beyond max_len, removed them.')
                    region_masks[img_idx] = cur_region_mask[:cur_region_token_num]

            # dump_region_mask = torch.zeros(100, 100).to(device='cuda')
            dump_region_mask = torch.zeros(100, 100, device='cuda')
            dump_region_mask[10:20, 10:20] = 1
            dump_region_masks = [[dump_region_mask.clone()]]
            for _ in range(len(region_feature_map)-1):
                dump_region_masks.append([])

            if region_geo_sampler:
                if type(image_features) is list:
                    region_features = self.get_model().region_geo_sampler(region_feature_map, region_masks, 
                                                                        original_dtype=raw_image_features.dtype,
                                                                        return_dtype=image_features[0].dtype)
                    dump_region_features = self.get_model().region_geo_sampler(region_feature_map, dump_region_masks, 
                                                                        original_dtype=raw_image_features.dtype,
                                                                        return_dtype=image_features[0].dtype)
                else:
                    region_features = self.get_model().region_geo_sampler(region_feature_map, region_masks, 
                                                                        original_dtype=raw_image_features.dtype,
                                                                        return_dtype=image_features.dtype)
                    dump_region_features = self.get_model().region_geo_sampler(region_feature_map, dump_region_masks, 
                                                                        original_dtype=raw_image_features.dtype,
                                                                        return_dtype=image_features.dtype)
            else:
                if type(image_features) is list:
                    region_features = self.extract_region_feature(region_feature_map, region_masks, 
                                                                original_dtype=raw_image_features.dtype,
                                                                return_dtype=image_features[0].dtype)
                    dump_region_features = self.extract_region_feature(region_feature_map, dump_region_masks, 
                                                                original_dtype=raw_image_features.dtype,
                                                                return_dtype=image_features[0].dtype)
                else:
                    region_features = self.extract_region_feature(region_feature_map, region_masks, 
                                                              original_dtype=raw_image_features.dtype,
                                                              return_dtype=image_features.dtype)
                    dump_region_features = self.extract_region_feature(region_feature_map, dump_region_masks, 
                                                                original_dtype=raw_image_features.dtype,
                                                                return_dtype=image_features.dtype)
            # assert len(dump_region_features) == 1
            assert len([df for df in dump_region_features if df is not None]) == 1
            assert len(dump_region_features[0]) == 1
            assert len(region_features) == len(input_ids)
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_id_with_im = []
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            assert len(cur_input_ids_noim) == len(cur_input_embeds_no_im)
            for i in range(num_images + 1):
                cur_input_id_with_im.append(cur_input_ids_noim[i])
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_input_id_with_im.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_input_id_with_im = torch.cat(cur_input_id_with_im)

            assert len(cur_input_id_with_im) == len(cur_new_input_embeds)
            # Add region feature into text feature embeddings.
            # Currently only support one image in each input.
            assert batch_idx+1 == cur_image_idx
            if region_flag and region_features[batch_idx] is not None:
                region_embs = torch.zeros_like(cur_new_input_embeds)
                region_replace_mask = (cur_input_id_with_im == self.config.im_region_fea_token)
                # region_embs[region_replace_mask] = region_features[batch_idx].to(cur_new_input_embeds.dtype)
                if len(region_embs[region_replace_mask]) != len(region_features[batch_idx]):
                    # ("Found a region cropped in text")
                    region_embs[region_replace_mask] = region_features[batch_idx][:len(region_embs[region_replace_mask])].to(cur_new_input_embeds.dtype)
                else:
                    region_embs[region_replace_mask] = region_features[batch_idx].to(cur_new_input_embeds.dtype)
                cur_new_input_embeds = cur_new_input_embeds * (~region_replace_mask).to(cur_new_input_embeds.dtype)[:, None] + region_embs 
            else:
                if hasattr(self.config, 'im_region_fea_token'):
                    assert (cur_input_id_with_im == self.config.im_region_fea_token).sum() == 0
            
            # Add dump region feature to input embedding, to make sure the gradient for region sampler always exist when open region_flag.
            if region_flag:
                # cur_new_input_embeds[0] = cur_new_input_embeds[0] + 0 * dump_region_features[0, 0].to(cur_new_input_embeds.dtype)
                cur_new_input_embeds[0] = cur_new_input_embeds[0] + 0.0 * dump_region_features[0][0].to(cur_new_input_embeds.dtype)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer, add_region_feature=False):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        
        if add_region_feature:
            region_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_REGION_FEA_TOKEN])[0]
            # If region_token doesn't exist, add it.
            if region_token_id == tokenizer.unk_token_id:
                num_region_fea_tokens = tokenizer.add_tokens([DEFAULT_REGION_FEA_TOKEN], special_tokens=True)
                self.config.im_region_fea_token = tokenizer.convert_tokens_to_ids([DEFAULT_REGION_FEA_TOKEN])[0]
                self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if add_region_feature:
                num_new_tokens = num_new_tokens + num_region_fea_tokens

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
