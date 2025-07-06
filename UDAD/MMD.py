import torch
import torch.nn as nn
import torch.optim as optim
from core.data import loader_onedimension
from models.model2 import encoder, classifier
from UDAD.base import DomainDiscriminator, Validation


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


class MAD():

    def __init__(self, epoch, target_path, device, dataloader_source, dataloader_target, feature_extractor, classifier,
                 domain_discriminator, criterion):
        print('Using MAD method to adapt...')
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.device = device
        self.dataloader_source = dataloader_source
        self.dataloader_target = dataloader_target
        self.domain_discriminator = domain_discriminator
        # 定义优化器
        self.optimizer_fe = optim.SGD(
            feature_extractor.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_dd = optim.SGD(
            domain_discriminator.parameters(), lr=0.001, momentum=0.9)
        self.target_path = target_path
        self.num_epochs = epoch
        self.criterion = criterion

    def mmd_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

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
                self.optimizer_dd.zero_grad()
                # 提取特征
                source_features = self.feature_extractor(source_data.float())
                target_features = self.feature_extractor(target_data.float())
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = self.criterion(source_pred, source_label.long())
                # 计算mmd损失
                mmd_loss_value = self.mmd_loss(
                    source_features, target_features)
                # 域判别器前向传播
                source_domain_pred = self.domain_discriminator(source_features)
                target_domain_pred = self.domain_discriminator(target_features)
                # 计算域判别器损失
                # 指定源域的样本域标签为0
                domain_label_source = torch.zeros(
                    source_domain_pred.size()[0]).to(self.device)
                # 指定目标域域标签为1
                domain_label_target = torch.ones(
                    target_domain_pred.size()[0]).to(self.device)
                domain_loss = self.criterion(source_domain_pred, domain_label_source.long()) + \
                              self.criterion(target_domain_pred, domain_label_target.long())
                # 结合分类器损失和MMD损失 计算全部的损失
                lambda_mmd = 0.5
                lambda_dd = 0.5
                total_loss = classifier_loss + lambda_mmd * \
                             mmd_loss_value + lambda_dd * domain_loss
                # 反向传播
                total_loss.backward()
                # 更新参数
                self.optimizer_fe.step()
                self.optimizer_cl.step()
                self.optimizer_dd.step()
            self.val()

    # print('自适应结束\n')
    # print("验证域自适应效果进行时\n")
    # self.val()
    def val(self):

        Validation(self.target_path, self.feature_extractor, self.classifier, self.device)


if __name__ == '__main__':
    path_source = r'D:\yuzhen\PADE\OneDimension\use\Domain1'
    path_target = r'D:\yuzhen\PADE\OneDimension\use\Domain4'
    batch_size = 256
    epoch = 10
    class_nums = 9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, dataloader_source = loader_onedimension(path_source, batch_size=batch_size)
    _, dataloader_target = loader_onedimension(path_target, batch_size=batch_size)
    feature_extractor = encoder(1).to(device)
    classifier = classifier(class_nums).to(device)
    domain_discriminator = DomainDiscriminator(input_dim=2048 * 2).to(device)
    criterion = nn.CrossEntropyLoss()
    mmd = MAD(epoch, path_target, device, dataloader_source, dataloader_target, feature_extractor, classifier,
              domain_discriminator, criterion)
    mmd.train()
