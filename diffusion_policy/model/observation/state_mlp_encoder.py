'''
Author: Si-Yuan Huang siyuan.huang@quantgroup.com
Date: 2025-01-17 15:33:58
LastEditors: Si-Yuan Huang
LastEditTime: 2025-01-24 12:05:12
FilePath: /diffusion_policy/diffusion_policy/model/observation/state_mlp_encoder.py
Description: 

'''
import torch
import torch.nn as nn
from diffusion_policy.model.observation.base_obs_encoder import BaseObsEncoder


class StateMlpEncoder(BaseObsEncoder):
    def __init__(self, input_shape, lowdim_embed_dims, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        # in_dim = obs_key_shapes[key][0]  # input dimension from shape meta
        in_dim = input_shape
        layers = []
        if lowdim_embed_dims is not None:
            layers.extend([nn.Linear(in_dim, lowdim_embed_dims[0])])
            for i in range(len(lowdim_embed_dims)-1):
                layers.extend([nn.ReLU(),nn.Linear(lowdim_embed_dims[i], lowdim_embed_dims[i+1])])
            self.out_dim = lowdim_embed_dims[-1]
        else:
            layers = [nn.Identity()]
            self.out_dim = in_dim
        self.mlp = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.mlp(x)
    
    @property
    def feature_dim(self):
        return self.out_dim



if __name__ == "__main__":
    params = {
        "lowdim_embed_dims": [10, 32],
        "shape_meta": 111
    }
    encoder = StateMlpEncoder(input_shape=(10,), **params)
    print(encoder.feature_dim)