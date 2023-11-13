from fnnmodule import *

from fnnmodule.backbone import *


class YOLOBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channel, out_channel*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(out_channel, out_channel*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
    
        return x


class YOLOv3Head(nn.Module):
    def __init__(self):
        # 大目标特征处理
        self.yoloblock0 = YOLOBlock(in_channel=1024, out_channel=512)
        self.upsample0 = nn.Upsample(scale_factor=2)
        self.conv00 = get_cbl(512, 256, kernel_size=1, padding=0)  # 为高级特征层向低级特征信息流动而转换通道
        self.conv01 = get_cbl(512, 1024, kernel_size=3, padding=1)
        # self.conv02 = get_cbl(1024, self.final_channel, kernel_size=1, padding=0)
        self.conv02 = nn.Conv2d(1024, self.final_channel, kernel_size=1, padding=0)

        # 中等目标特征处理
        self.yoloblock1 = YOLOBlock(in_channel=768, out_channel=256)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv10 = get_cbl(256, 128, kernel_size=1, padding=0)  # 为高级特征层向低级特征信息流动而转换通道
        self.conv11 = get_cbl(256, 512, kernel_size=3, padding=1)
        # self.conv12 = get_cbl(512, self.final_channel, kernel_size=1, padding=0)
        self.conv12 = nn.Conv2d(512, self.final_channel, kernel_size=1, padding=0)

        # 小目标特征处理
        self.yoloblock2 = YOLOBlock(in_channel=384, out_channel=128)
        self.conv20 = get_cbl(128, 256, kernel_size=3, padding=1)
        # self.conv21 = get_cbl(256, self.final_channel, kernel_size=1, padding=0)
        self.conv21 = nn.Conv2d(256, self.final_channel, kernel_size=1, padding=0)

    def forward(self, x0, x1, x2):
        feat0 = self.yoloblock0(x2)
        feat1 = self.conv00(feat0)
        feat0 = self.conv01(feat0)
        out0 = self.conv02(feat0)


        feat1 = torch.cat([self.upsample0(feat1), x1], dim=1)
        feat1 = self.yoloblock1(feat1)
        feat2 = self.conv10(feat1)
        feat1 = self.conv11(feat1)
        out1 = self.conv12(feat1)


        feat2 = self.upsample1(feat2)
        feat2 = torch.cat([feat2, x0], dim=1)
        feat2 = self.yoloblock2(feat2)
        feat2 = self.conv20(feat2)
        out2 = self.conv21(feat2)

        out0 = out0.view(out0.shape[0], 3, -1, out0.shape[2], out0.shape[3])
        out0 = out0.permute(0, 1, 3, 4, 2)
        out1 = out1.view(out1.shape[0], 3, -1, out1.shape[2], out1.shape[3])
        out1 = out1.permute(0, 1, 3, 4, 2)
        out2 = out2.view(out2.shape[0], 3, -1, out2.shape[2], out2.shape[3])
        out2 = out2.permute(0, 1, 3, 4, 2)

        return out0, out1, out2