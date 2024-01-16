
from fnnmodule import *
import torch.nn as nn

from fnnconfig.utils.str2cls import dict2cls, new_cls


class CLRNet(nn.modules):

    def __init__(
            self,        
            backbone=None,
            neck=None,
            head=None,
            is_qat=False,
        ):
        super().__init__()

        self.is_qat = is_qat

        if isinstance(backbone, dict):
            backbone = dict2cls(backbone)

        if isinstance(neck, dict):
            neck = dict2cls(neck)

        if isinstance(head, dict):
            head = dict2cls(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)
        
        return out
    