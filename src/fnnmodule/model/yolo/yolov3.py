from fnnmodule import *

from fnnmodule.backbone import DarkNet53
from fnnmodule.head import YOLOv3Head


class YOLOv3(nn.Module):
    def __init__(self, num_cls, img_w, img_h, stride=[32,16,8]):
        super().__init__()
        self.num_cls = num_cls
        self.final_channel = 3*(self.num_cls+1+4)
        self.backbone = DarkNet53()

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        self.head = YOLOv3Head(num_cls=self.num_cls)

        self.stride = stride

        self.grid_0 = self._make_grid(nx=img_w//self.stride[0], ny=img_h//self.stride[0])
        self.grid_1 = self._make_grid(nx=img_w//self.stride[1], ny=img_h//self.stride[1])
        self.grid_2 = self._make_grid(nx=img_w//self.stride[2], ny=img_h//self.stride[2])

        self.anchor_grid_0 = torch.tensor([[[[[116.,  90.]]],
                                            [[[156., 198.]]],
                                            [[[373., 326.]]]]])
        self.anchor_grid_1 = torch.tensor([[[[[ 30.,  61.]]],
                                            [[[ 62.,  45.]]],
                                            [[[ 59., 119.]]]]])
        self.anchor_grid_2 = torch.tensor([[[[[10., 13.]]],
                                            [[[16., 30.]]],
                                            [[[33., 23.]]]]])

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)]) # , indexing='ij'
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        x0, x1, x2 = self.backbone(x)  # 256, 512, 1024

        out0, out1, out2 = self.head(x0, x1, x2)

        return out0, out1, out2  # 高中低级别特征检测结果（大中小目标检测结果）
