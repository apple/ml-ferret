import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
# Added for customized Processor.
import math
import numpy as np
from typing import Dict
from transformers.image_utils import PILImageResampling, ChannelDimension
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import (
    get_resize_output_image_size,
    resize,
)
from typing import List, Optional, Tuple, Union

class CLIPImageProcessor_GIT(CLIPImageProcessor):
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size, default_to_square=True, height_width_order=True)
        # Hack(haoxuan): Bypass the shortest_edge detection. We hope to get a {"height": size[0], "width": size[1]}, where w=h.
        # if "shortest_edge" not in size:
        #     raise ValueError(f"The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}")
        # output_size = get_resize_output_image_size(image, size=size["shortest_edge"], default_to_square=True)
        output_size = get_resize_output_image_size(image, size=(size["height"], size["width"]), default_to_square=True)
        return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)
    

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, vision_tower_path=None):
        self.image_processor = CLIPImageProcessor_GIT.from_pretrained(self.vision_tower_name)
        if vision_tower_path is not None:
            self.vision_tower, loading_info = CLIPVisionModel.from_pretrained(vision_tower_path, output_loading_info=True)
            print('loading_info:', loading_info)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
