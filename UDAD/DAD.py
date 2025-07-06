import torch
import torch.nn as nn
import torch.optim as optim
from core.data import loader_onedimension
from models.model2 import encoder, classifier
from UDAD.base import DomainDiscriminator, Validation


class DAD():
    def __init__(self, epoch, path_target, device, dataloader_source, dataloader_target, feature_extractor, classifier,
                 criterion,
                 domain_discriminator):
        print('Using DAD method to adapt...')
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.device = device
        self.dataloader_source = dataloader_source
        self.dataloader_target = dataloader_target
        self.domain_discriminator = domain_discriminator
        self.classifier = classifier
        # 定义优化器
        self.optimizer_fe = optim.SGD(
            feature_extractor.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_dd = optim.SGD(
            domain_discriminator.parameters(), lr=0.001, momentum=0.9)
        self.num_epochs = epoch
        self.target_path = path_target
        self.criterion = criterion
        self.criterion_cl = criterion
        self.criterion_dd = criterion
        self.lambda_adv = 0.1  # 对抗性损失的权重

    def train(self):
        for epoch in range(self.num_epochs):
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

                # 提取特征
                source_features = self.feature_extractor(source_data.float())
                target_features = self.feature_extractor(target_data.float())
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = self.criterion(source_pred, source_label.long())
                # 域判别器前向传播
                source_domain_pred = self.domain_discriminator(source_features)
                target_domain_pred = self.domain_discriminator(target_features)
                # 构建域标签
                source_domain_label = torch.zeros(
                    source_domain_pred.size(0)).long().cuda()
                target_domain_label = torch.ones(
                    target_domain_pred.size(0)).long().cuda()
                # 计算域判别器损失
                domain_loss = self.criterion_dd(source_domain_pred, source_domain_label) + \
                              self.criterion_dd(target_domain_pred, target_domain_label)
                # 特征提取器和分类器的对抗性损失
                adversarial_loss = -domain_loss
                # 总损失
                total_loss = classifier_loss + self.lambda_adv * adversarial_loss
                # 反向传播和优化
                self.optimizer_fe.zero_grad()
                self.optimizer_cl.zero_grad()
                self.optimizer_dd.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer_fe.step()
                self.optimizer_cl.step()
                # 计算domain_loss及其相关张量，避免inplace操作的影响
                source_domain_pred = self.domain_discriminator(
                    source_features.detach())
                target_domain_pred = self.domain_discriminator(
                    target_features.detach())
                source_domain_label = torch.zeros(
                    source_domain_pred.size(0)).long().cuda()
                target_domain_label = torch.ones(
                    target_domain_pred.size(0)).long().cuda()
                domain_loss = self.criterion_dd(source_domain_pred, source_domain_label) + \
                              self.criterion_dd(target_domain_pred, target_domain_label)
                # 单独更新域判别器
                domain_loss.backward()
                self.optimizer_dd.step()
                # # 打印进度
                # if i % 100 == 0:
                #     print(f'Epoch [{epoch + 1}/{self.num_epochs}]')
        # print('自适应结束\n')
        # print("验证域自适应效果进行时\n")
            self.val()

    # self.val()
    def val(self):

        Validation(self.target_path, self.feature_extractor, self.classifier, self.device)


if __name__ == '__main__':
    path_source = r'D:\yuzhen\PADE\OneDimension\use\Domain1'
    parth_target = r'D:\yuzhen\PADE\OneDimension\use\Domain4'
    batch_size = 512
    epoch = 10
    class_nums = 9
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    _, dataloader_source = loader_onedimension(path_source, batch_size=batch_size)
    _, dataloader_target = loader_onedimension(parth_target, batch_size=batch_size)

    feature_extractor = encoder(1).to(device)
    classifier = classifier(class_nums).to(device)
    domain_discriminator = DomainDiscriminator(
        input_dim=2048 * 2).to(device)
    criterion = nn.CrossEntropyLoss()

    dad = DAD(epoch, parth_target, device, dataloader_source, dataloader_target, feature_extractor, classifier,
              criterion, domain_discriminator)
    dad.train()
