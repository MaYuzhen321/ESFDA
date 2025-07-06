# 用于复现无源域故障诊断论文中的实验
# 在hust数据集上测试，预训练过程无过拟合且自适应效果良好
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from basic.train import DATA_ONE_DIMENSION,VibrationDataset


class Net_paper(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super(Net_paper, self).__init__()
        # Feature encoding layers
        self.num_classes = out_classes
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, padding=32, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 48, kernel_size=16, padding=8, stride=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(48, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Classification layers
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


if __name__ == '__main__':
    model = Net_paper(1, 4)
    save_dir = r'G:\数据集\机械故障诊断数据集\JUST_USE\一维卷积测试权重\domain6.pth'
    path = r'G:\数据集\机械故障诊断数据集\JUST_USE\use\domain6'
    abc = DATA_ONE_DIMENSION(path, save_dir, model)
    abc.train(20, 32)
