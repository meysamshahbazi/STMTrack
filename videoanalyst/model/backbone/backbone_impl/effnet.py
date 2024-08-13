import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.efficientnet import *

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.module_base import ModuleBase


@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class Efficientnet_b0_M(ModuleBase):
    default_hyper_params = dict(
        pretrained=True
    )

    def __init__(self, transform_input=False):
        super(Efficientnet_b0_M, self).__init__()
        effnet = efficientnet_b0(pretrained=self._hyper_params['pretrained'])
        self.l1 = effnet.features[0]
        self.feat_ = effnet.features[1:4]
        self.proj_fg_bg_label_map = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        torch.nn.init.normal_(self.proj_fg_bg_label_map.weight, std=0.01)

    def forward(self, x, fg_bg_label_map=None):
        bias = 255 / 2
        x_ch0 = (torch.unsqueeze(x[:, 2], 1) - bias) / bias
        x_ch1 = (torch.unsqueeze(x[:, 1], 1) - bias) / bias
        x_ch2 = (torch.unsqueeze(x[:, 0], 1) - bias) / bias
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.l1(x) + self.proj_fg_bg_label_map(fg_bg_label_map)
        x = self.feat_(x)
        return x
    
    def update_params(self):
        super().update_params()
        self.pretrained = self._hyper_params['pretrained']



@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class Efficientnet_b0_Q(ModuleBase):
    default_hyper_params = dict(
        pretrained=True
    )

    def __init__(self, transform_input=False):
        super(Efficientnet_b0_Q, self).__init__()
        effnet = efficientnet_b0(pretrained=self._hyper_params['pretrained'])
        self.feat_ = effnet.features[0:4]

    def forward(self, x, fg_bg_label_map=None):
        bias = 255 / 2
        x_ch0 = (torch.unsqueeze(x[:, 2], 1) - bias) / bias
        x_ch1 = (torch.unsqueeze(x[:, 1], 1) - bias) / bias
        x_ch2 = (torch.unsqueeze(x[:, 0], 1) - bias) / bias
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.feat_(x)
        return x
    
    def update_params(self):
        super().update_params()
        self.pretrained = self._hyper_params['pretrained']

