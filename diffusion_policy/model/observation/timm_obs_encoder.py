'''
Author: Si-Yuan Huang siyuan.huang@quantgroup.com
Date: 2025-01-09 16:42:47
LastEditors: Si-Yuan Huang siyuan.huang@quantgroup.com
LastEditTime: 2025-01-23 12:34:00
FilePath: /code/policy_learning_pipline/bos_learning/policy/observation/timm_obs_encoder.py
Description: Timm-based observation encoder, with unified interface that handles different observation shape requirements.

'''
import copy
import timm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
# Model utilities
from bos_learning.policy.common.module_attr_mixin import ModuleAttrMixin
from bos_learning.policy.common.pytorch_util import replace_submodules
from bos_learning.policy.observation.vision_modules import AttentionPool2d
logger = logging.getLogger(__name__)


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            global_pool: str,
            transforms: list,
            frozen: bool,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            vision_aggregation: str='attention_pool_2d',
            feature_aggregation: str='all', # "all, token, feature"
            downsample_ratio: int=32,
            position_encording: str='learnable',
            # For transformer-based models, project the feature to n_emb
            n_emb: int=768,
            feature_projection: bool=False,
            **kwargs
        ):
        """
        Assumes rgb input: B,To,C,H,W
        Assumes low_dim input: B,To,D
        
        
        Args:
            shape_meta: dict, shape meta for observation and action
            model_name: str, name of the timm model
            pretrained: bool, whether to use pretrained model
            global_pool: '' means no pooling - we use self-defined vision_aggregation
            transforms: list, custom transforms for images additional to the default crop and resize
            frozen: bool, whether to freeze the vision encoder
            use_group_norm: bool, whether to use group normalization, ignored if pretrained is True
            share_rgb_model: bool, whether to share the rgb model for all rgb inputs
            vision_aggregation: ['attention_pool_2d', 'avg', etc.], vision aggregation method for encoder outputs
            feature_aggregation: ['all', 'token', 'feature'], how to aggregate the final output
            downsample_ratio: Downsample ratio that affects ResNet's final feature map size. 32: 7*7, 16: 14*14.
            position_encording: str, position encoding method only for transformer based aggregation.
            n_emb: int, embedding dimension, works if feature_projection is True.
            feature_projection: bool, whether to project the feature to n_emb.
        """
        super().__init__()
        # ============Parse shape meta============
        rgb_keys = list()
        low_dim_keys = list()
        # ============Model dicts to initialize============
        key_model_map = nn.ModuleDict() # Models for each key
        key_transform_map = nn.ModuleDict() # Preprocessing for each key
        if feature_projection:
            key_projection_map = nn.ModuleDict() # Feature dim Linear Projection for each key
        key_shape_map = dict() # TODO: Use unified function.
        # ============Model Creation============
        assert global_pool == '' # '' means no pooling - we use self-defined vision aggregation
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        self.model_name = model_name
        # ============Detach model if frozen============
        if frozen:
            # freeze all parameters for observation encoder
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        # ============Model Modifications============
        feature_dim = None
        if model_name.startswith('resnet'):
            # ============ResNet Modifications============
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        
        elif model_name.startswith('convnext'):
            # ============ConvNext Modifications============
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        # TODO: This hierarchy is not clear. 
        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) \
                        if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        
        # ============vision backbone's raw featuremap aggregation============
        # This includes the postprocess of the high-dim feature map from resnet 
        # or Select output token from ViT
        self.vision_aggregation = vision_aggregation
        if model_name.startswith('vit'):
            # assert self.vision_aggregation is None # vit uses the CLS token
            if self.vision_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif self.vision_aggregation == 'cls':
                logger.warn(f'vit will use the CLS token. \
                    vision_aggregation ({self.vision_aggregation}) is ignored!')
            else:
                raise NotImplementedError(
                    f'Unsupported vision_aggregation for ViT: {self.vision_aggregation}'
                )
        else:
            feature_map_shape = [x // downsample_ratio for x in image_shape]
            # ResNet-based models
            if self.vision_aggregation == 'attention_pool_2d':
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_shape[0],
                    embed_dim=feature_dim,
                    num_heads=feature_dim // 64,
                    output_dim=feature_dim
                )
            elif self.vision_aggregation == 'soft_attention':
                self.attention = nn.Sequential(
                    nn.Linear(feature_dim, 1, bias=False),
                    nn.Softmax(dim=1)
                )
            elif self.vision_aggregation == 'spatial_embedding':
                self.spatial_embedding = torch.nn.Parameter(
                    torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim)
                )
            elif self.vision_aggregation == 'transformer':
                if position_encording == 'learnable':
                    self.position_embedding = torch.nn.Parameter(
                        torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim)
                    )
                elif position_encording == 'sinusoidal':
                    num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                    self.position_embedding = torch.zeros(num_features, feature_dim)
                    position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, feature_dim, 2).float() \
                            * (-math.log(2 * num_features) / feature_dim)
                    )
                    self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                    self.position_embedding[:, 1::2] = torch.cos(position * div_term)
                self.aggregation_transformer = nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                    num_layers=4)

        # ============Image Transformations============
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            # Must-have random crop and resize, and customizable transforms
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        image_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        # ============Prepare model for all the keys============
        for key, attr in obs_shape_meta.items():
            # TODO(siyuan): Conversion to new feature format.
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # Determine if rgb model is shared
                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                # ============Handle feature projection============
                if feature_projection:
                    with torch.no_grad():
                        example_img = torch.zeros((2,)+tuple(shape)) # (1, C, H, W)
                        example_feature_map = this_model(example_img) # (1, n_tokens, feature_dim) or (1, C', H', W')
                        example_features = self.aggregate_feature(example_feature_map)
                        print("[TimmObsEncoder] shape from model:", example_feature_map.shape)
                        print("[TimmObsEncoder] Feature shape after aggregation:", example_features.shape)
                        feature_shape = example_features.shape
                        feature_size = feature_shape[-1]
                    if feature_size != n_emb:
                        # If feature size is not the same as n_emb, we need to project it to n_emb.
                        proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                    else:
                        proj = nn.Identity()
                    key_projection_map[key] = proj
                this_transform = image_transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                # TODO(siyuan): StateMLP for state features.
                # TODO(siyuan): This should be either using feature projection or specified mlp shape.
                # ============Handle low_dim projection============
                if feature_projection:
                    dim = np.prod(shape) # theoretically, this is simply the input dim
                    proj = nn.Identity()
                    if dim != n_emb:
                        proj = nn.Linear(in_features=dim, out_features=n_emb)
                    key_projection_map[key] = proj
                    print("Projection keys:  ",key)
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        # =============== Vision Aggregation ===============
        # The raw list of features we got from  the model is like
        # - (B, To, n_tokens, feature_dim)
        # We want to aggregate in different ways for different backbones.
        # - all: (B, To, n_tokens, feature_dim) -> (B, To*n_tokens*feature_dim)
        # - token: (B, To, n_tokens, feature_dim) -> (B, To*n_tokens, feature_dim)
        # - feature: (B, To, n_tokens, feature_dim) -> (B, To, n_tokens*feature_dim)   
        if feature_aggregation == 'all':
            self.feature_aggregation_dimensions = [1, -1]
        elif feature_aggregation == 'token':
            self.feature_aggregation_dimensions = [1, 2]
        elif feature_aggregation == 'feature':
            # This is for old transformer diffusion policy
            self.feature_aggregation_dimensions = [2, 3]
        # ============Log keys============
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('[TimmObsEncoder] rgb keys:         ', rgb_keys)
        print('[TimmObsEncoder] low_dim_keys keys:', low_dim_keys)
        # print('Projection keys:  ', key_projection_map.keys())
        # ============Save attributes============
        self.shape_meta = shape_meta
        self.feature_aggregation = feature_aggregation
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.feature_projection = feature_projection
        if feature_projection:
            self.key_projection_map = key_projection_map
            self.n_emb = n_emb
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self._output_shape = None
        self.kwargs = kwargs # TODO: remove this
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        print(key_shape_map.keys())

    def aggregate_feature(self, feature):
        """ 
        Input:  (B*To, n_tokens, feature_dim) from ViT
                (B*To, C, H, W) from ResNet
        
        Output: (B*To, n or 1, feature_dim)
        """
        # ===========ViT token selection============
        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.vision_aggregation == 'cls':
                return feature[:, [0], :] # (B*T,1,n_emb)
            # or use all tokens
            elif self.vision_aggregation == 'all_tokens':
                return feature # (B*T, num_vit_tokens, n_emb)
            else:
                raise NotImplementedError(
                    f'Unsupported vision_aggregation for ViT: {self.vision_aggregation}')
        
        # ===========resnet ============
        # Input: (B*T, Channel, feature_map[0], feature_map[1])
        assert len(feature.shape) == 4
        if self.vision_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature).unsqueeze(1) # (B*T, 1, n_emb)

        # TODO: possible bug here, but not used or tested
        feature = torch.flatten(feature, start_dim=-2) # (B*T, 512, 7*7)
        feature = torch.transpose(feature, 1, 2) # (B*T, 7*7, 512)

        # TODO: Is it necessay to use keep dim?
        if self.vision_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.vision_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.vision_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.vision_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        # ============TODO: untested modules Transformer============
        elif self.vision_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.vision_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        """
        Assume image input shape is (B,To,C,H,W)
        Assume low_dim input shape is (B,To,D)

        """
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, To = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*To, *img.shape[2:]) # (B*To, C, H, W)
            img = self.key_transform_map[key](img) # TODO: preprocessing -- Put this to data loader.
            raw_feature = self.key_model_map[key](img) # Pass thru encoder
            feature = self.aggregate_feature(raw_feature) # (B*To, 1, feature_dim)
            # TODO: remove this sanity check
            assert len(feature.shape) == 3 and feature.shape[0] == B * To # ideally (B*To, n_tokens, feature_dim)
            # ===========Handle feature projection============
            if self.feature_projection:
                # This is for transformer-based models that requires projection to n_emb
                feature = self.key_projection_map[key](feature) # (B*To, n_tokens, n_emb)
                # TODO: comment out this sanity check
                assert feature.shape[-1] == self.n_emb
                # feature = emb.reshape(B, -1, self.n_emb) # (B, To, n_emb)
            # Reshape to (B, To, n_tokens, n_emb)
            feature = feature.reshape(B, To, *feature.shape[1:])
            # Aggregate features according to feature_aggregation
            feature = torch.flatten(
                feature,
                start_dim=self.feature_aggregation_dimensions[0],
                end_dim=self.feature_aggregation_dimensions[1]
            )
            features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key] # (B, To, D)
            B, To = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            # ===========Handle feature projection============
            if self.feature_projection:
                data = self.key_projection_map[key](data) # (B, To, n_emb)
                assert data.shape[-1] == self.n_emb
            if self.feature_aggregation == 'all':
                data = data.reshape(B, -1) # (B, To*n_emb)
            features.append(data)
        # print("Single feature shape:")
        # for feature in features:
        #     print(feature.shape)
        # concatenate all features
        if self.feature_aggregation == 'feature':
            result = torch.cat(features, dim=-1)
        else:
            result = torch.cat(features, dim=1)
        return result # (B, n_tokens, n_emb) for transformer-based models, (B, n_obs * features) for unet-based models

    @property
    def output_shape(self):
        if self._output_shape is None:
            with torch.no_grad():
                example_obs_dict = dict()
                obs_shape_meta = self.shape_meta['obs']
                for key, attr in obs_shape_meta.items():
                    shape = tuple(attr['shape'])
                    this_obs = torch.zeros(
                        (1, attr['horizon']) + shape, 
                        dtype=self.dtype,
                        device=self.device)
                    example_obs_dict[key] = this_obs
                example_output = self.forward(example_obs_dict)
            self._output_shape = example_output.shape
        return self._output_shape


def test():

    shape_meta = {
        "obs": {
            "observation_images_d405": {
                "shape": [3, 224, 224],  # Shape fed into the encoder
                "raw_shape": [480, 848, 3],      # Shape from raw sensor
                "type": "rgb",
                "hz": 30
            },
            "observation_images_d455": {
                "shape": [3, 224, 224],
                "raw_shape": [480, 640, 3],
                "type": "rgb",
                "hz": 30
            },
            "observation_states_ee_pose": {
                "shape": [9],     # 3D position + 6D orientation
                "raw_shape": [6], # 3D position + 3D ZYX
                "type": "low_dim",
                "hz": 200
            },
            "observation_states_joint_angle": {
                "shape": [7],     # 7 joint angles
                "raw_shape": [7], # 7 joint angles
                "type": "low_dim",
                "hz": 200
            }
            # Commented out section from original YAML:
            # "observation.states.gripper_pos": {
            #     "shape": [1],
            #     "raw_shape": [1],
            #     "type": "low_dim",
            #     "hz": 200
            # }
        },
        "action": {
            "shape": [10]  # 3D position + 6D orientation + 1d gripper
        }
    }    
    # Test ViT

    encoder = TimmObsEncoder(
        shape_meta,
        model_name='vit_base_patch16_clip_224.openai',
        pretrained=True,
        global_pool='',
        transforms=None,
        frozen=False,
        use_group_norm=False,
        share_rgb_model=False,
        vision_aggregation='cls',
        feature_aggregation='token',
        downsample_ratio=32,
        position_encording='learnable',
        imagenet_norm=False,
        n_emb=768,
        feature_projection=True
    )
    
    test_obs_dict = {
        "observation_images_d405": torch.randn(5, 2, 3, 224, 224),
        "observation_images_d455": torch.randn(5, 2, 3, 224, 224),
        "observation_states_ee_pose": torch.randn(5, 2, 9),
        "observation_states_joint_angle": torch.randn(5, 2, 7)
    }
    output = encoder(test_obs_dict)
    print("Output feature dim:", output.shape)


if __name__ == '__main__':
    test()
    