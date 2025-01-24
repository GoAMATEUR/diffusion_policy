'''
Author: Si-Yuan Huang siyuan.huang@quantgroup.com
Date: 2025-01-17 15:50:36
LastEditors: Si-Yuan Huang siyuan.huang@quantgroup.com
LastEditTime: 2025-01-17 15:50:52
FilePath: /policy_learning_pipline/bos_learning/policy/observation/base_obs_encoder.py
Description: 

'''
import torch
import torch.nn as nn

class BaseObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def feature_dim(self):
        # Output deature dimension.
        raise NotImplementedError

    def forward(self, obs):
        raise NotImplementedError
