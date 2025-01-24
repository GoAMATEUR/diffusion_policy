'''
Author: Si-Yuan Huang siyuan.huang@quantgroup.com
Date: 2025-01-13 11:13:54
LastEditors: Si-Yuan Huang
LastEditTime: 2025-01-24 12:12:36
FilePath: /diffusion_policy/diffusion_policy/policy/diffusion_unet_policy.py
Description: 1. Added 2 layer MLP for low-dim observation.

'''
from typing import Dict, Union, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
# from diffusion_policy.common.robomimic_config_util import get_robomimic_config
# from robomimic.algo import algo_factory
# from robomimic.algo.algo import PolicyAlgo
# import robomimic.utils.obs_utils as ObsUtils
# import robomimic.models.base_nets as rmbn
# import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


# Vision Encoder:
# from bos_learning.policy.observation.diffusion_rgb_encoder import DiffusionRgbEncoder
# from bos_learning.policy.observation.state_mlp_encoder import StateMlpEncoder
from diffusion_policy.model.observation.base_obs_encoder import BaseObsEncoder
# from bos_learning.dataset.common.shape_meta_parser import parse_shape_meta
import hydra


def parse_shape_meta(shape_meta: dict):
    """shape_meta:
    {
        "obs": {
            "observation_states_xxx": {
                "shape": [10],
                "type": "low_dim",
                "hz": 200
            },
            "observation_images_xxx": {
                "shape": [100, 100, 3],
                "type": "rgb",
                "hz": 30
            },
        "action": {
            "shape": [10],
            "type": "float32",
            "hz": 200
        }
    }

    Args:
        shape_meta (_type_): _description_
    """
    state_shape_meta = {}
    image_shape_meta = {}
    obs_shape_meta = shape_meta['obs']
 
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            image_shape_meta[key] = attr['shape']
        elif type == 'low_dim':
            state_shape_meta[key] = attr['shape']
        else:
            raise RuntimeError(f"Unsupported obs type: {type}")
    action_shape = shape_meta['action']['shape']
    return state_shape_meta, image_shape_meta, action_shape


class DiffusionUnetPolicy(BaseImagePolicy):
    def __init__(self,
        shape_meta: dict,
        encoder_meta: dict,
        noise_scheduler,
        # task params
        horizon: int,
        n_action_steps: int, 
        n_obs_steps: int,
        num_inference_steps: Optional[int]=None,
        # Model
        diffusion_step_embed_dim: int=256,
        # lowdim_embed_dims: int=(256, 32),
        down_dims: Tuple[int, int, int]=(256,512,1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs
    ):
        super().__init__()

        # ========parse shape_meta=========
        state_shape_meta, image_shape_meta, action_shape = \
            parse_shape_meta(shape_meta)
        
        # 3. prepare obs encoder and accumulate obs feature dim
        # TODO: Put this into a unified class.
        obs_feature_single_dim = 0 # accumulator for obs feature dim
        obs_encoder_dict = {}
        if len(image_shape_meta) > 0:
            for key in image_shape_meta:
                this_encoder_cls = hydra.utils.get_class(encoder_meta.images.encoder_type)
                this_obs_encoder: BaseObsEncoder = this_encoder_cls(**encoder_meta.images)
                obs_encoder_dict[key] = this_obs_encoder
                obs_feature_single_dim += this_obs_encoder.feature_dim
                print(f"[Policy] obs_feature_single_dim: {key}: {this_obs_encoder.feature_dim}")
        if len(state_shape_meta) > 0:
            for key in state_shape_meta:
                # Create MLP layers dynamically
                this_encoder_cls = hydra.utils.get_class(encoder_meta.states.encoder_type)
                this_obs_encoder: BaseObsEncoder = this_encoder_cls(
                    input_shape=state_shape_meta[key][0], **encoder_meta.states)
                obs_encoder_dict[key] = this_obs_encoder
                obs_feature_single_dim += this_obs_encoder.feature_dim
                print(f"[Policy] obs_feature_single_dim: {key}: {this_obs_encoder.feature_dim}")
        global_cond_dim = obs_feature_single_dim * n_obs_steps
        # self.global_cond_dim = global_cond_dim
        self.obs_feature_single_dim = obs_feature_single_dim
        self.obs_encoder_dict = nn.ModuleDict(obs_encoder_dict)
        print("[Policy] global_cond_dim: ", global_cond_dim)
        print("[Policy] obs_feature_single_dim: ", obs_feature_single_dim)
        # ========Prepare Unet=========
        input_dim = action_shape[0] # unet input
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.obs_as_global_cond = True
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_shape[0],
            obs_dim=0 if self.obs_as_global_cond else obs_feature_single_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        ) # TODO: del
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_single_dim
        self.action_dim = input_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("[Policy] Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("[Policy] obs_encoder params: %e" % sum(p.numel() for p in self.obs_encoder_dict.parameters()))

    def _prepare_global_cond(self, nobs_dict: Dict[str, torch.Tensor]):
        # TODO: handle different ways of passing observation
        global_cond = []
        for key, value in nobs_dict.items():
            global_cond.append(self.obs_encoder_dict[key](value))
        global_cond = torch.cat(global_cond, dim=-1) # (B*T, D)
        assert global_cond.shape[-1] == self.obs_feature_single_dim
        return global_cond

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            # get global feature
            # nobs_features = self.obs_encoder(this_nobs)
            nobs_features = self._prepare_global_cond(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # nobs_features = self.obs_encoder(this_nobs)
            # global_cond = nobs_features.reshape(batch_size, -1)
            nobs_features = self._prepare_global_cond(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
