import torch
import torch.nn as nn
import torch.optim as optim
from core.data import loader_onedimension
from models.model2 import encoder, classifier
from UDAD.base import DomainDiscriminator, Validation


class CADA():
    def __init__(self, epoch, device, target_path, dataloader_source, dataloader_target, feature_extractor, classifier,
                 domain_discriminator_cada, criterion, class_nums):
        print("使用CADA方案进行无监督域自适应进行时\n")
        self.device = device
        self.target_path = target_path
        self.dataloader_source = dataloader_source
        self.dataloader_target = dataloader_target
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator_cada = domain_discriminator_cada
        # 定义优化器
        self.optimizer_fe = optim.SGD(
            self.feature_extractor.parameters(), lr=0.01, momentum=0.9)
        self.optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.01, momentum=0.9)
        self.optimizer_dd = optim.SGD(
            self.domain_discriminator_cada.parameters(), lr=0.01, momentum=0.9)
        self.num_epochs = epoch
        self.criterion = criterion
        self.criterion_cl = self.criterion
        self.criterion_dd = self.criterion
        self.lambda_adv = 0.1  # 对抗性损失的权重
        # 对抗学习引入权重
        self.class_nums = class_nums
        self.lambda_adv = 0.1  # 对抗性损失的权重

    def train(self):
        for epoch in range(self.num_epochs):

            for i, ((source_data, source_label), (target_data, _)) in enumerate(
                    zip(dataloader_source, dataloader_target)):
                # 确保source_data和target_data的大小相同
                batch_size = min(len(source_data), len(target_data))
                source_data = source_data[:batch_size].to(device)
                source_label = source_label[:batch_size].to(device).long()
                target_data = target_data[:batch_size].to(device)

                if len(source_data) < len(target_data):
                    padding_size = len(target_data) - len(source_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *source_data.shape[1:])).to(device)
                    source_data = torch.cat([source_data, padding_tensor], dim=0)
                    source_label = torch.cat([source_label, torch.zeros(
                        padding_size, dtype=torch.long).to(device)], dim=0)
                elif len(target_data) < len(source_data):
                    padding_size = len(source_data) - len(target_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *target_data.shape[1:])).to(device)
                    target_data = torch.cat([target_data, padding_tensor], dim=0)
                # 提取特征
                source_features = self.feature_extractor(source_data.float())
                target_features = self.feature_extractor(target_data.float())
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = self.criterion(source_pred, source_label.long())
                # 构建one-hot编码的条件标签
                source_label_one_hot = torch.eye(self.class_nums)[source_label].cuda()
                target_label_one_hot = torch.eye(self.class_nums)[torch.randint(
                    0, self.class_nums, (target_features.size(0),))].cuda()
                # 域判别器前向传播
                source_domain_pred = self.domain_discriminator_cada(
                    source_features.detach(), source_label_one_hot)
                target_domain_pred = self.domain_discriminator_cada(
                    target_features.detach(), target_label_one_hot)

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

                # 单独更新域判别器
                domain_loss.backward()
                self.optimizer_dd.step()

                # # 打印进度
                # if i % 100 == 0:
                #     print(
                #         f'Epoch [{epoch + 1}/{self.num_epochs}]')
        # print('自适应结束\n')
        # print("验证域自适应效果进行时\n")
            self.val()

    def val(self):
        Validation(self.target_path, self.feature_extractor, self.classifier, self.device)


class DomainDiscriminator_CADA(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DomainDiscriminator_CADA, self).__init__()
        self.domain_discriminator = nn.Sequential(
            nn.Linear(input_dim + num_classes, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x, y):
        # 将特征x和条件y拼接在一起
        x_y = torch.cat([x, y], dim=1)
        return self.domain_discriminator(x_y)


if __name__ == '__main__':
    path_source = r'D:\yuzhen\PADE\OneDimension\use\Domain1'
    path_target = r'D:\yuzhen\PADE\OneDimension\use\Domain4'
    batch_size = 32
    epoch = 10
    class_nums = 9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, dataloader_source = loader_onedimension(path_source, batch_size=batch_size)
    _, dataloader_target = loader_onedimension(path_target, batch_size=batch_size)
    feature_extractor = encoder(1).to(device)
    classifier = classifier(class_nums).to(device)
    domain_discriminator = DomainDiscriminator(input_dim=2048 * 2).to(device)
    criterion = nn.CrossEntropyLoss()
    domain_discriminator_cada = DomainDiscriminator_CADA(
        input_dim=2048*2, num_classes=class_nums).to(device)
    # for i in range(10):
    #     # print(f'实验组别：{i+1}')
    #     print("Experiment_num: {}".format(i + 1))
    cada = CADA(epoch, device, path_target, dataloader_source, dataloader_target, feature_extractor, classifier,
                domain_discriminator_cada, criterion, class_nums)
    cada.train()
