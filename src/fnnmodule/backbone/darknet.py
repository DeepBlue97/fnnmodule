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
