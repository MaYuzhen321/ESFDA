import torch
import torch.nn as nn
import torch.optim as optim
from core.data import loader_onedimension
from models.model2 import encoder, classifier
from UDAD.base import Validation


class COAD():
    def __init__(self, parth_target, epoch, device, dataloader_source, dataloader_target, criterion, feature_extractor,
                 classifier):
        print('Using CoAD method to adapt...')
        self.device = device
        self.feature_extractor = feature_extractor
        self.dataloader_source = dataloader_source
        self.dataloader_target = dataloader_target
        self.classifier = classifier
        self.optimizer_fe = optim.SGD( self.feature_extractor.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_cl = optim.SGD( self.classifier.parameters(), lr=0.001, momentum=0.9)
        self.num_epochs = epoch
        self.criterion = criterion
        self.target_path = parth_target

    def train(self):
        for epoch in range(self.num_epochs):
            # print(f'Epoch [{epoch + 1}/{self.num_epochs}]')
            for i, ((source_data, source_label), (target_data, _)) in enumerate(
                    zip(self.dataloader_source, self.dataloader_target)):
                # 确保source_data和target_data的大小相同
                batch_size = min(len(source_data), len(target_data))
                source_data = source_data[:batch_size].to(self.device)
                source_label = source_label[:batch_size].to(self.device)
                target_data = target_data[:batch_size].to(self.device)

                if len(source_data) < len(target_data):
                    padding_size = len(target_data) - len(source_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *source_data.shape[1:])).to(self.device)
                    source_data = torch.cat(
                        [source_data, padding_tensor], dim=0)
                    source_label = torch.cat([source_label, torch.zeros(
                        padding_size, dtype=torch.long).to(self.device)], dim=0)
                elif len(target_data) < len(source_data):
                    padding_size = len(source_data) - len(target_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *target_data.shape[1:])).to(self.device)
                    target_data = torch.cat(
                        [target_data, padding_tensor], dim=0)

                self.optimizer_fe.zero_grad()
                self.optimizer_cl.zero_grad()
                # optimizer_dd.zero_grad()
                # 提取特征
                # print('source_data', source_data.shape)
                source_features = self.feature_extractor(source_data.float())
                target_features = self.feature_extractor(target_data.float())
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = self.criterion(source_pred, source_label.long())
                # 计算coral损失
                coral_loss_value = self.coral_loss(
                    source_features, target_features)
                # 计算总损失
                lambda_coral = 0.5
                total_loss = classifier_loss + lambda_coral * coral_loss_value
                total_loss.backward()
                self.optimizer_fe.step()
                self.optimizer_cl.step()
            self.val()
        # print('调整结束')
        # self.val()

    def val(self):

        Validation(self.target_path, self.feature_extractor, self.classifier, self.device)

    # self.t_SNE()

    def coral_loss(self, source, target):
        d = source.size(1)
        # 计算均值
        mean_source = torch.mean(source, 0)
        mean_target = torch.mean(target, 0)
        # 中心化
        centered_source = source - mean_source
        centered_target = target - mean_target
        # 计算协方差矩阵
        cov_source = torch.mm(centered_source.t(), centered_source) / (d - 1)
        cov_target = torch.mm(centered_target.t(), centered_target) / (d - 1)
        # 计算Frobenius范数
        loss = torch.norm(cov_source - cov_target, p='fro')

        return loss


if __name__ == '__main__':
    path_source = r'D:\yuzhen\PADE\OneDimension\use\Domain1'
    path_target = r'D:\yuzhen\PADE\OneDimension\use\Domain4'
    batch_size = 256
    epoch = 10
    class_nums = 9
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    _, dataloader_source = loader_onedimension(path_source, batch_size=batch_size)
    _, dataloader_target = loader_onedimension(path_target, batch_size=batch_size)

    feature_extractor = encoder(1).to(device)
    classifier = classifier(class_nums).to(device)

    criterion = nn.CrossEntropyLoss()

    coad = COAD(path_target, epoch, device, dataloader_source, dataloader_target, criterion, feature_extractor,
                classifier)
    coad.train()
    # coad.val()
    # coad.train()
    # coad.val()
    # coad.train()
    # coad.val()
    # coad.train()
    # coad.val()
    # coad.train()
    # coad.val()
