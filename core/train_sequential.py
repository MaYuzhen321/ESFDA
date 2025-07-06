'''
target:
1. 读取数据与标签 给定一个域
2. 搭建网络
'''
# from mine import Model
from rich.progress import track
from rich import print
from rich.console import Console
from torchvision import transforms, models
import h5py

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import os
import math
import numpy as np
from typing import List, Tuple
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm


class VibrationDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx].reshape(1, 32, 32)
        label = self.labels[idx]

        if self.transform:
            signal = self.transform(signal)

        return signal, label


# CRWU的lr用的是0.001
# PADE的lr用的是   0.6889  0.6489


class DATA_SEQUENTIAL:
    def __init__(self, data_dir: str, weight_dir: str, model_type: str = "resnet50",
                 epochs: int = 200, criterion: str = "CrossEntropyLoss",
                 batch_size: int = 256, device: torch.device = None):
        self.data_dir = data_dir
        self.model_type = model_type
        self.epochs = epochs
        self.criterion_name = criterion
        self.batch_size = batch_size
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._set_model(classes=self._get_class_count())
        self.criterion = self._set_criterion()
        # lr=0.01, weight_decay=0.00009
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.save_dir = weight_dir

    # 获取分类类别
    def _get_class_count(self) -> int:
        return len(os.listdir(os.path.join(self.data_dir, 'train')))

    # 加载数据并调整数据尺寸
    def _load_and_adjust_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        data, labels = self.load_data(path)
        # print(labels.shape)
        return self.adjust_to_square(data), labels

    # 加载数据

    def load_data(self, data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
        data = []
        labels_str = []
        unique_labels = {}
        label_count = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.h5'):
                    label = os.path.basename(root)
                    if label not in unique_labels:
                        unique_labels[label] = label_count
                        label_count += 1

                    with h5py.File(os.path.join(root, file), 'r') as hf:
                        signals = hf['signals']
                        signal_names = list(signals.keys())
                        for signal_name in signal_names:
                            signal_data = signals[signal_name][:]
                            data.append(signal_data)
                            labels_str.append(label)
        labels = [unique_labels[label] for label in labels_str]
        return np.array(data), np.array(labels)

    # 调整称为矩形

    def adjust_to_square(self, data: np.ndarray, target_dim: int = 32) -> np.ndarray:
        if target_dim is None:
            seq_length = data.shape[1]
            target_dim = int(math.sqrt(seq_length))
            target_dim = min(target_dim, seq_length // 2)
            while (target_dim * target_dim) < seq_length:
                target_dim += 1
        pad_size = target_dim ** 2 - data.shape[2]
        if pad_size > 0:
            padding = np.zeros((data.shape[0], pad_size))
            data_padded = np.hstack((data, padding))
        elif pad_size < 0:
            data_padded = data[:, :target_dim ** 2]
        else:
            data_padded = data
        data_squared = data_padded.reshape(
            data.shape[0], data.shape[1], target_dim, target_dim)
        return data_squared

    def _set_model(self, classes: int) -> nn.Module:
        if self.model_type == "resnet50":
            model = models.resnet50(pretrained=True)
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, classes)
        elif self.model_type == "resnet18":
            model = models.resnet18(pretrained=True)
            model.conv1 = nn.Conv2d(
                6, 64, kernel_size=5, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, classes)
        elif self.model_type == "vgg16":
            model = models.vgg16(pretrained=False)
            model.features[0] = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.classifier[6] = nn.Linear(4096, classes)
        elif self.model_type == "own":
            model = Model()
        return model.to(self.device)

    def _set_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss() if self.criterion_name == "CrossEntropyLoss" else None

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.float(), labels.long()
                outputs = self.model(inputs.to(self.device))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        self.model.train()
        return accuracy_score(all_labels, all_preds)

    def train(self):
        print(self.data_dir)
        train_data, train_labels = self._load_and_adjust_data(
            os.path.join(self.data_dir, 'train'))
        test_data, test_labels = self._load_and_adjust_data(
            os.path.join(self.data_dir, 'test'))

        train_dataset = VibrationDataset(train_data, train_labels)
        test_dataset = VibrationDataset(test_data, test_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)
        # 初始化最佳模型以及模型权重
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in track(range(self.epochs)):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())

                # print(outputs.shape)

                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            train_acc = self.evaluate(train_loader)
            test_acc = self.evaluate(test_loader)

            print(
                f'Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.save_dir)
        self.model.load_state_dict(best_model_wts)
        print(f'Best test Acc: {best_acc:.4f}')


class CRWU_p:
    def __init__(self, data_dir: str, model_type: str = "resnet50",
                 epochs: int = 100, criterion: str = "CrossEntropyLoss",
                 batch_size: int = 32, device: torch.device = None):
        self.data_dir = data_dir
        self.model_type = model_type
        self.epochs = epochs
        self.criterion_name = criterion
        self.batch_size = batch_size
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.model = self._set_model(classes=self._get_class_count())
        # self.criterion = self._set_criterion()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # self.save_dir = weight_dir

    def load_data(self, data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
        data = []
        labels_str = []
        unique_labels = {}
        label_count = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.h5'):
                    label = os.path.basename(root)
                    if label not in unique_labels:
                        unique_labels[label] = label_count
                        label_count += 1

                    with h5py.File(os.path.join(root, file), 'r') as hf:
                        signals = hf['signals']
                        signal_names = list(signals.keys())
                        for signal_name in signal_names:
                            signal_data = signals[signal_name][:]
                            data.append(signal_data)
                            labels_str.append(label)
        labels = [unique_labels[label] for label in labels_str]
        return np.array(data), np.array(labels)

    def adjust_to_square(self, data: np.ndarray, target_dim: int = 32) -> np.ndarray:
        if target_dim is None:
            seq_length = data.shape[1]
            target_dim = int(math.sqrt(seq_length))
            target_dim = min(target_dim, seq_length // 2)
            while (target_dim * target_dim) < seq_length:
                target_dim += 1
        pad_size = target_dim ** 2 - data.shape[1]
        if pad_size > 0:
            padding = np.zeros((data.shape[0], pad_size))
            data_padded = np.hstack((data, padding))
        elif pad_size < 0:
            data_padded = data[:, :target_dim ** 2]
        else:
            data_padded = data
        data_squared = data_padded.reshape(
            data.shape[0], target_dim, target_dim)
        return data_squared

    def _load_and_adjust_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Combines load_data and adjust_to_square logic
        data, labels = self.load_data(path)
        data = data.reshape(data.shape[0], -1)
        return self.adjust_to_square(data), labels


if __name__ == '__main__':
    console = Console()

    # domain_path = r'G:\数据集\机械故障诊断数据集\JUST_for_use\use\domain1'
    # domain_path = r'G:\数据集\机械故障诊断数据集\JUST_for_use\use\domain2'
    # domain_path = r'G:\数据集\机械故障诊断数据集\JUST_for_use\use\domain3'
    # domain_path = r'G:\数据集\机械故障诊断数据集\JUST_for_use\use\domain4'  # 从这个域开始出问题
    # domain_path = r'G:\数据集\机械故障诊断数据集\JUST_for_use\use\domain5'
    # domain_path = r'G:\数据集\机械故障诊断数据集\JUST_for_use\use\domain6'

    # weight_dir = r'G:\数据集\机械故障诊断数据集\JUST_for_use\weights\domain1.pth'
    # weight_dir = r'G:\数据集\机械故障诊断数据集\JUST_for_use\weights\domain2.pth'
    # weight_dir = r'G:\数据集\机械故障诊断数据集\JUST_for_use\weights\domain3.pth'
    # weight_dir = r'G:\数据集\机械故障诊断数据集\JUST_for_use\weights\domain4.pth'
    # weight_dir = r'G:\数据集\机械故障诊断数据集\JUST_for_use\weights\domain5.pth'
    # weight_dir = r'G:\数据集\机械故障诊断数据集\JUST_for_use\weights\domain6.pth'

    # just = DATA_SEQUENTIAL(domain_path, weight_dir, "resnet18", 200)
    # just.train()

    # domain_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\use\1797_12K_load0'
    # domain_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\use\1772_12K_load1'
    # domain_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\use\1750_12K_load2'
    # domain_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\use\1730_12K_load3'

    # weight_dir = r'C:\Users\86178\Desktop\weights_temp\1797_12K_load0_temp.pth'
    # weight_dir = r'C:\Users\86178\Desktop\weights_temp\1772_12K_load1.pth'
    # weight_dir = r'C:\Users\86178\Desktop\weights_temp\1750_12K_load2.pth'
    # weight_dir = r'C:\Users\86178\Desktop\weights_temp\1730_12K_load3.pth'

    # weight_dir = r'G:\数据集\机械故障诊断数据集\CRWU_D\weight\1797_12K_load0.pth'

    crwu = DATA_SEQUENTIAL(domain_path, weight_dir, "vgg16")
    crwu.train()
