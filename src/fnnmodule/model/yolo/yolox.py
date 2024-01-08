
from fnnmodule import *
import torch.nn as nn

from fnnmodule.head.yolox import YOLOXHead
from fnnmodule.neck.yolox_pafpn import YOLOPAFPN
from fnnconfig.utils.str2cls import dict2cls, new_cls


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

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

        # if backbone is None:
        #     backbone = YOLOPAFPN()
        # if head is None:
        #     head = YOLOXHead(num_classes)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.is_decode = False

        if self.is_qat:
            from pytorch_nndct import nn as nndct_nn
            self.quant_in = nndct_nn.QuantStub()
            

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        if self.is_qat:
            x = self.quant_in(x)
        dark_outs = self.backbone(x)
        fpn_outs = self.neck(dark_outs)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x, is_decode=self.is_decode
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs, is_decode=self.is_decode)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
