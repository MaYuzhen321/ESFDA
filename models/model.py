'''
自定义网络结构，包含三元注意力模块，实际测试下效果并不好
'''

import torch
from torch import nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


# 定义一个基础的卷积模块
class BasicConv(nn.Module):
    def __init__(
            self,
            in_planes,  # 输入通道数
            out_planes,  # 输出通道数
            kernel_size,  # 卷积核大小
            stride=1,  # 步长
            padding=0,  # 填充
            dilation=1,  # 空洞率
            groups=1,  # 分组卷积的组数
            relu=True,  # 是否使用ReLU激活函数
            bn=True,  # 是否使用批标准化
            bias=False,  # 卷积是否添加偏置
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # 定义卷积层
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # 可选的批标准化层
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        # 可选的ReLU激活层
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 定义一个通道池化模块
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


# 定义一个空间门控模块
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


# 定义一个三元注意力模块
class TripletAttention(nn.Module):
    def __init__(
            self,
            gate_channels,  # 门控通道数
            reduction_ratio=16,  # 缩减比率
            pool_types=["avg", "max"],  # 池化类型
            no_spatial=False,  # 是否禁用空间门控
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)

        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()

        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class MyModel(nn.Module):
    def __init__(self, input_shape=(1, 224, 224)):
        super(MyModel, self).__init__()

        # 输入尺寸为(input_channels, height, width)，这里假设输入图像通道数为1，高度和宽度分别为224
        self.input_shape = input_shape

        # 卷积层
        self.conv1 = Convolution(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Convolution(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Convolution(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = Convolution(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=5)
        self.attention = TripletAttention(256, 16, 'max', True)

        # 全连接层
        self.fc1 = nn.Linear(4608, 10)
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.dropout2 = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(1024, 1024)  # 假设最后输出维度为1000

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.attention(x)

        x = x.view(-1, 128 * 6 * 6)


        return x


class MyModel2(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), classes=10):
        super(MyModel2, self).__init__()
        self.layer1 = MyModel()
        self.layer2 = nn.Linear(4608, classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# x = torch.randn(500, 1, 32, 32)
# model = MyModel2()
# out = model(x)
# print(out.shape)
