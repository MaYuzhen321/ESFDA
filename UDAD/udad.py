import os
import torch
import copy
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models.model2 import Net, encoder, classifier
from core.data import load_onedimension, loader_onedimension


class Comeparsion():
    def __init__(self, source_path, target_path, batch=64):
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225])
            ]),
        }
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # 源域目录
        self.source_path = source_path
        # 目标域目录
        self.target_path = target_path
        #
        data_source_domain = os.path.join(self.source_path, 'train')
        data_target_domain = os.path.join(self.target_path, 'train')
        self.batch_size = batch
        self.image_dataset1, self.dataloader_source = self.load_data(
            data_source_domain)
        self.image_dataset2, self.dataloader_target = self.load_data(
            data_target_domain)
        self.files = os.listdir(data_source_domain)
        # class_names = self.image_dataset1.classes
        self.class_nums = len(self.files)
        # self.feature_extractor = FeatureExtractor().to(self.device)

        self.feature_extractor = encoder(1).to(self.device)
        self.classifier = classifier(self.class_nums).to(self.device)
        # self.feature_extractor = transformer.to(self.device)
        # self.classifier = Classifier(
        #     input_dim=1000, num_classes=self.class_nums).to(self.device)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=2048 * 2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.domain_discriminator_cada = DomainDiscriminator_CADA(
            input_dim=2048, num_classes=self.class_nums).to(self.device)

    def load_data(self, path, phase='train', shuffle=True, num_workers=0):
        print(path)
        dataset, dataloader = loader_onedimension(path, batch_size=self.batch_size)
        return dataset, dataloader

    def mmd_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = gaussian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

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

    def MAD(self, epoch):
        print("使用MAD方案进行无监督域自适应进行时\n")
        device = self.device
        dataloader_source = self.dataloader_source
        dataloader_target = self.dataloader_target
        domain_discriminator = self.domain_discriminator
        # 定义优化器
        optimizer_fe = optim.SGD(
            self.feature_extractor.parameters(), lr=0.01, momentum=0.9)
        optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.01, momentum=0.9)
        optimizer_dd = optim.SGD(
            domain_discriminator.parameters(), lr=0.01, momentum=0.9)
        num_epochs = epoch
        '''
        明天把训练过程封装一下
        -  epoch 训练轮数
        -  dataloader  两个域上的
        - 特征提取器
        - 分类器
        - 损失计算方式
        - 优化器
        '''
        criterion = self.criterion
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            for i, ((source_data, source_label), (target_data, _)) in enumerate(
                    zip(dataloader_source, dataloader_target)):
                # 确保source_data和target_data的大小相同
                batch_size = min(len(source_data), len(target_data))
                source_data = source_data[:batch_size].to(device)
                source_label = source_label[:batch_size].to(device)
                target_data = target_data[:batch_size].to(device)

                if len(source_data) < len(target_data):
                    padding_size = len(target_data) - len(source_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *source_data.shape[1:])).to(device)
                    source_data = torch.cat(
                        [source_data, padding_tensor], dim=0)
                    source_label = torch.cat([source_label, torch.zeros(
                        padding_size, dtype=torch.long).to(device)], dim=0)
                elif len(target_data) < len(source_data):
                    padding_size = len(source_data) - len(target_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *target_data.shape[1:])).to(device)
                    target_data = torch.cat(
                        [target_data, padding_tensor], dim=0)

                optimizer_fe.zero_grad()
                optimizer_cl.zero_grad()
                optimizer_dd.zero_grad()
                # 提取特征
                source_features = self.feature_extractor(source_data)
                target_features = self.feature_extractor(target_data)
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = criterion(source_pred, source_label)
                # 计算mmd损失
                mmd_loss_value = self.mmd_loss(
                    source_features, target_features)
                # 域判别器前向传播
                source_domain_pred = domain_discriminator(source_features)
                target_domain_pred = domain_discriminator(target_features)
                # 计算域判别器损失
                # 指定源域的样本域标签为0
                domain_label_source = torch.zeros(
                    source_domain_pred.size()[0]).to(device)
                # 指定目标域域标签为1
                domain_label_target = torch.ones(
                    target_domain_pred.size()[0]).to(device)
                domain_loss = criterion(source_domain_pred, domain_label_source.long()) + \
                              criterion(target_domain_pred, domain_label_target.long())
                # 结合分类器损失和MMD损失 计算全部的损失
                lambda_mmd = 0.5
                lambda_dd = 0.5
                total_loss = classifier_loss + lambda_mmd * \
                             mmd_loss_value + lambda_dd * domain_loss
                # 反向传播
                total_loss.backward()
                # 更新参数
                optimizer_fe.step()
                optimizer_cl.step()
                optimizer_dd.step()
        print('自适应结束\n')
        print("验证域自适应效果进行时\n")
        self.val()

    def COAD(self, epoch):

        print("使用COAD方案进行无监督域自适应进行时\n")
        device = self.device
        dataloader_source = self.dataloader_source
        dataloader_target = self.dataloader_target
        optimizer_fe = optim.SGD(
            self.feature_extractor.parameters(), lr=0.01, momentum=0.9)
        optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.01, momentum=0.9)
        num_epochs = epoch
        criterion = self.criterion
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            for i, ((source_data, source_label), (target_data, _)) in enumerate(
                    zip(dataloader_source, dataloader_target)):
                # 确保source_data和target_data的大小相同
                batch_size = min(len(source_data), len(target_data))
                source_data = source_data[:batch_size].to(device)
                source_label = source_label[:batch_size].to(device)
                target_data = target_data[:batch_size].to(device)

                if len(source_data) < len(target_data):
                    padding_size = len(target_data) - len(source_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *source_data.shape[1:])).to(device)
                    source_data = torch.cat(
                        [source_data, padding_tensor], dim=0)
                    source_label = torch.cat([source_label, torch.zeros(
                        padding_size, dtype=torch.long).to(device)], dim=0)
                elif len(target_data) < len(source_data):
                    padding_size = len(source_data) - len(target_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *target_data.shape[1:])).to(device)
                    target_data = torch.cat(
                        [target_data, padding_tensor], dim=0)

                optimizer_fe.zero_grad()
                optimizer_cl.zero_grad()
                # optimizer_dd.zero_grad()
                # 提取特征
                # print('source_data', source_data.shape)
                source_features = self.feature_extractor(source_data.float())
                target_features = self.feature_extractor(target_data.float())
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = criterion(source_pred, source_label.long())
                # 计算coral损失
                coral_loss_value = self.coral_loss(
                    source_features, target_features)
                # 计算总损失
                lambda_coral = 0.5
                total_loss = classifier_loss + lambda_coral * coral_loss_value
                total_loss.backward()
                optimizer_fe.step()
                optimizer_cl.step()
        print('调整结束')
        self.val()
        # self.t_SNE()

    def DAD(self, epoch):
        print("使用DAD方案进行无监督域自适应进行时\n")
        device = self.device
        dataloader_source = self.dataloader_source
        dataloader_target = self.dataloader_target
        # 定义优化器
        optimizer_fe = optim.SGD(
            self.feature_extractor.parameters(), lr=0.01, momentum=0.9)
        optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.01, momentum=0.9)
        optimizer_dd = optim.SGD(
            self.domain_discriminator.parameters(), lr=0.01, momentum=0.9)
        num_epochs = epoch
        criterion_cl = self.criterion
        criterion_dd = self.criterion
        lambda_adv = 0.1  # 对抗性损失的权重
        for epoch in range(num_epochs):
            for i, ((source_data, source_label), (target_data, _)) in enumerate(
                    zip(dataloader_source, dataloader_target)):
                # 确保source_data和target_data的大小相同
                batch_size = min(len(source_data), len(target_data))
                source_data = source_data[:batch_size].to(device)
                source_label = source_label[:batch_size].to(device)
                target_data = target_data[:batch_size].to(device)

                if len(source_data) < len(target_data):
                    padding_size = len(target_data) - len(source_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *source_data.shape[1:])).to(device)
                    source_data = torch.cat(
                        [source_data, padding_tensor], dim=0)
                    source_label = torch.cat([source_label, torch.zeros(
                        padding_size, dtype=torch.long).to(device)], dim=0)
                elif len(target_data) < len(source_data):
                    padding_size = len(source_data) - len(target_data)
                    padding_tensor = torch.zeros(
                        (padding_size, *target_data.shape[1:])).to(device)
                    target_data = torch.cat(
                        [target_data, padding_tensor], dim=0)

                # 提取特征
                source_features = self.feature_extractor(source_data)
                target_features = self.feature_extractor(target_data)
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = self.criterion(source_pred, source_label)
                # 域判别器前向传播
                source_domain_pred = self.domain_discriminator(source_features)
                target_domain_pred = self.domain_discriminator(target_features)
                # 构建域标签
                source_domain_label = torch.zeros(
                    source_domain_pred.size(0)).long().cuda()
                target_domain_label = torch.ones(
                    target_domain_pred.size(0)).long().cuda()
                # 计算域判别器损失
                domain_loss = criterion_dd(source_domain_pred, source_domain_label) + \
                              criterion_dd(target_domain_pred, target_domain_label)
                # 特征提取器和分类器的对抗性损失
                adversarial_loss = -domain_loss

                # 总损失
                total_loss = classifier_loss + lambda_adv * adversarial_loss

                # 反向传播和优化
                optimizer_fe.zero_grad()
                optimizer_cl.zero_grad()
                optimizer_dd.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer_fe.step()
                optimizer_cl.step()

                # 计算domain_loss及其相关张量，避免inplace操作的影响
                source_domain_pred = self.domain_discriminator(
                    source_features.detach())
                target_domain_pred = self.domain_discriminator(
                    target_features.detach())
                source_domain_label = torch.zeros(
                    source_domain_pred.size(0)).long().cuda()
                target_domain_label = torch.ones(
                    target_domain_pred.size(0)).long().cuda()
                domain_loss = criterion_dd(source_domain_pred, source_domain_label) + \
                              criterion_dd(target_domain_pred, target_domain_label)

                # 单独更新域判别器
                domain_loss.backward()
                optimizer_dd.step()

                # 打印进度
                if i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print('自适应结束\n')
        print("验证域自适应效果进行时\n")
        self.val()

    def CADA(self, epoch):
        print("使用CADA方案进行无监督域自适应进行时\n")
        device = self.device
        dataloader_source = self.dataloader_source
        dataloader_target = self.dataloader_target
        # 定义优化器
        optimizer_fe = optim.SGD(
            self.feature_extractor.parameters(), lr=0.01, momentum=0.9)
        optimizer_cl = optim.SGD(
            self.classifier.parameters(), lr=0.01, momentum=0.9)
        optimizer_dd = optim.SGD(
            self.domain_discriminator_cada.parameters(), lr=0.01, momentum=0.9)
        num_epochs = epoch
        criterion_cl = self.criterion
        criterion_dd = self.criterion
        lambda_adv = 0.1  # 对抗性损失的权重
        # 对抗学习引入权重
        lambda_adv = 0.1  # 对抗性损失的权重
        for epoch in range(num_epochs):

            for i, ((source_data, source_label), (target_data, _)) in enumerate(
                    zip(dataloader_source, dataloader_target)):
                # 确保source_data和target_data的大小相同
                batch_size = min(len(source_data), len(target_data))
                source_data = source_data[:batch_size].to(device)
                source_label = source_label[:batch_size].to(device)
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
                source_features = self.feature_extractor(source_data)
                target_features = self.feature_extractor(target_data)
                # 分类器前向传播
                source_pred = self.classifier(source_features)
                # 计算分类损失
                classifier_loss = self.criterion(source_pred, source_label)
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
                domain_loss = criterion_dd(source_domain_pred, source_domain_label) + \
                              criterion_dd(target_domain_pred, target_domain_label)

                # 特征提取器和分类器的对抗性损失
                adversarial_loss = -domain_loss

                # 总损失
                total_loss = classifier_loss + lambda_adv * adversarial_loss

                # 反向传播和优化
                optimizer_fe.zero_grad()
                optimizer_cl.zero_grad()
                optimizer_dd.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer_fe.step()
                optimizer_cl.step()

                # 单独更新域判别器
                domain_loss.backward()
                optimizer_dd.step()

                # 打印进度
                if i % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}]')
        print('自适应结束\n')
        print("验证域自适应效果进行时\n")
        self.val()

    def FULL(self):
        print("源域模型直接用于测试")
        self.val()

    def val(self):
        val_path = os.path.join(self.target_path, 'train')
        dataset, dataloader = self.load_data(val_path, "val")
        self.feature_extractor.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(self.device), label.to(self.device)
                features = self.feature_extractor(data.float())
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        accuracy = 100 * correct / total
        print("Accuracy of the model on the target domain validation set {}%\n".format(accuracy))
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Confusion Matrix:\n{cm}')

        self.feature_extractor.train()
        self.classifier.train()

    def t_SNE(self):
        val_path = os.path.join(self.target_path, 'train')
        print(val_path)
        dataset, dataloader = self.load_data(val_path, "val")
        self.feature_extractor.eval()
        self.classifier.eval()
        self.feature_extractor.to(self.device)
        self.classifier.to(self.device)
        # 提取特征和标签
        features = []
        labels = []
        with torch.no_grad():
            for images, targets in dataloader:
                # print('targets',targets)
                images = images.to(self.device)
                feature = self.feature_extractor(images.float())
                output = self.classifier(feature)
                features.append(output.cpu().numpy())
                labels.append(targets.numpy())
        # 将特征和标签拼接为一个完整的数组
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        # 应用t-SNE
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(features)

        # 可视化t-SNE结果
        plt.figure(figsize=(16, 10))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
        plt.legend(handles=scatter.legend_elements()[0], labels=self.files)
        plt.title('t-SNE visualization of features')
        plt.show()


# 分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # 设置最后一层的输出维度
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# 域判别器
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        return self.fc(x)


# 定义域判别器
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


# 高斯核MMD损失函数
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


if __name__ == '__main__':
    # 源域
    source_domain_path = r'G:\数据集\华中科技大学数据集\SIZE\use\domain1'
    # 目标域
    target_domain_path = r'G:\数据集\华中科技大学数据集\SIZE\use\domain2'
    cp = Comeparsion(source_domain_path, target_domain_path)
    # cp.FULL()
    cp.COAD(20)  # 0.37
    cp.CADA(50)
    # cp.MAD(50)
    # cp.DAD(50)
