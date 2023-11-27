from fnnmodule import *


def get_cbl(in_channel, out_channel, kernel_size, stride=1, padding=0, eps=0.001, momentum=0.03):
    """get Conv BachNorm Activate Layers"""
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel, 
                        #    eps=eps, momentum=momentum, affine=True, track_running_stats=True
                           ),
            # nn.ReLU()
            nn.LeakyReLU(0.1015625)
            # nn.ReLU()
        )


def get_Bottleneck(in_channel):
    """get BottleNeck"""
    neck_channel = in_channel//2
    return nn.Sequential(
        get_cbl(in_channel, neck_channel, 1, 1, 0),
        get_cbl(neck_channel, in_channel, 3, 1, 1),
    )


# def get_StackBottleneck(num_stack: int, in_channel):
#     """get StackBottleneck"""
#     blocks = []
#     for i in range(num_stack):
#         blocks.append(get_Bottleneck(in_channel=in_channel))
#     return nn.Sequential(*blocks)

# def get_ResBlock(num_block: int, in_channel):
#     """get StackBottleneck"""
#     blocks = []
#     for i in range(num_block):
#         blocks.append(get_Bottleneck(in_channel=in_channel))
#     return nn.Sequential(*blocks)


class ResBlock(nn.Module):

    def __init__(self, num_block: int, in_channel):
        super().__init__()

        self.num_block = num_block

        for i in range(self.num_block):
            setattr(self, f'bottleneck{i}', get_Bottleneck(in_channel=in_channel))
    
    def forward(self, x):
        for i in range(self.num_block):
            x = getattr(self, f'bottleneck{i}')(x) + x
        
        return x


class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            get_cbl(in_channel=3, out_channel=32, kernel_size=3, stride=1, padding=1),
            get_cbl(in_channel=32, out_channel=64, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=1, in_channel=64)
        )

        self.stage1 = nn.Sequential(
            get_cbl(in_channel=64, out_channel=128, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=2, in_channel=128)
        )

        self.stage2 = nn.Sequential(
            get_cbl(in_channel=128, out_channel=256, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=8, in_channel=256)
        )

        self.stage3 = nn.Sequential(
            get_cbl(in_channel=256, out_channel=512, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=8, in_channel=512)
        )

        self.stage4 = nn.Sequential(
            get_cbl(in_channel=512, out_channel=1024, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=4, in_channel=1024)
        )

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        out0 = self.stage2(x)
        out1 = self.stage3(out0)
        out2 = self.stage4(out1)

        return out0, out1, out2


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
