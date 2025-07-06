from basic.train import *

"""
在原始论文模型的基础之上添加了三元注意力机制
"""


class BasicConv1d(nn.Module):
    def __init__(
            self,
            in_channels,  # 输入通道数
            out_channels,  # 输出通道数
            kernel_size,  # 卷积核大小
            stride=1,  # 步长
            padding=0,  # 填充
            dilation=1,  # 空洞率
            groups=1,  # 分组卷积的组数
            relu=True,  # 是否使用ReLU激活函数
            bn=True,  # 是否使用批标准化
            bias=False,  # 卷积是否添加偏置
    ):
        super(BasicConv1d, self).__init__()
        self.out_channels = out_channels
        # 定义卷积层
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # 可选的批标准化层
        self.bn = (
            nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True)
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


# 通道池化模块
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


# 空间门控模块
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv1d(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_compress = x_compress.squeeze(-1)
        x_out = self.spatial(x_compress)
        x_out = x_out.unsqueeze(-1)
        scale = torch.sigmoid_(x_out)
        scale = scale.squeeze(-1)
        scale = scale.expand_as(x)
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
        x_perm1 = x.permute(0, 2, 1).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out2)
        else:
            x_out = (1 / 2) * (x_out11 + x_out2)
        return x_out


class Net(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super(Net, self).__init__()
        # Feature encoding layers
        self.num_classes = out_classes
        self.flag = 0
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, padding=32, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            TripletAttention(32, 64, 'max', True),

            nn.Conv1d(32, 48, kernel_size=16, padding=8, stride=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            TripletAttention(48, 64, 'max', True),

            nn.Conv1d(48, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(

            nn.Linear(2048*2, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, out_classes)
        )

    def forward(self, x):
        # print('输入尺寸', x.shape)
        x = self.encoder(x)
        # print('特征提取', x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(encoder, self).__init__()
        # Feature encoding layers
        self.flag = 0
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, padding=32, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            TripletAttention(32, 64, 'max', True),

            nn.Conv1d(32, 48, kernel_size=16, padding=8, stride=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            TripletAttention(48, 64, 'max', True),

            nn.Conv1d(48, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # print('输入尺寸', x.shape)
        x = self.encoder(x)
        # print('特征提取', x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = self.classifier(x)
        return x


class classifier(nn.Module):
    def __init__(self, out_classes=2):
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(

            nn.Linear(2048 * 2, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, out_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = Net(1, 10)
    # save_dir = r'G:\数据集\华中科技大学数据集\HUST\model2_weights\400W.pth'
    # path = r'G:\数据集\华中科技大学数据集\HUST\use\400W'
    save_dir = r'G:\数据集\机械故障诊断数据集\CRWU_for_Use\weights\domain0.pth'
    path = r'G:\数据集\机械故障诊断数据集\CRWU_for_Use\use\1797_12K_load0'
    abc = DATA_ONE_DIMENSION(path, save_dir, model)
    abc.train(10, 32)
