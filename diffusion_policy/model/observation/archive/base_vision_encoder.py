'''
Author: Si-Yuan Huang siyuan.huang@quantgroup.com
Date: 2025-01-09 11:45:51
LastEditors: Si-Yuan Huang siyuan.huang@quantgroup.com
LastEditTime: 2025-01-09 11:47:06
FilePath: /policy_learning_pipline/bos_learning/policy/vision/base_vision_encoder.py
Description: 

'''
import torch
from torch import nn
from bos_learning.policy.common.module_attr_mixin import ModuleAttrMixin


class BaseVisionEncoder(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
