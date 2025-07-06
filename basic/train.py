from cProfile import label

import torch.nn as nn
import torch
from overrides.signature import ensure_all_positional_args_defined_in_sub
from sympy.physics.paulialgebra import epsilon
from sympy.polys.distributedmodules import sdm_monomial_mul
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import h5py
from torchvision import transforms, models
from models.model2 import Net
import os
from sklearn.metrics import accuracy_score
import numpy as np
from rich.progress import track
import copy


class VibrationDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx].reshape(1, -1)
        label = self.labels[idx]

        if self.transform:
            signal = self.transform(signal)

        return signal, label


class DATA_ONE_DIMENSION():
    def __init__(self, data_dir, weight_dir, model=Net(), criterion='cross_entropy'):
        self.data_dir = data_dir
        self.weight_dir = weight_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.num_classes = self.model.num_classes
        if criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == "smoothing":
            self.criterion = cross_entropy_with_label_smothing(self.num_classes, epsilon=0.1)
        else:
            raise "No Such Criterion!!!!!!!!!!"
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.path = data_dir

    def load_data(self, data_dir=None):
        if data_dir is not None:
            data_dir = self.data_dir
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

    def train(self, epochs, batch_size):

        print('训练集路径', os.path.join(self.path, 'train'))
        print('测试集路径', os.path.join(self.path, 'test'))
        data_train, labels_train = self.load_data(os.path.join(self.path, 'train'))
        data_test, labels_test = self.load_data(os.path.join(self.path, 'test'))
        train_dataset = VibrationDataset(data_train, labels_train)
        test_dataset = VibrationDataset(data_test, labels_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in track(range(epochs)):
            # print(epoch)
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())
                loss = self.criterion.forward(outputs, labels.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            train_acc = self.evaluate(self.model, train_loader, self.device)
            test_acc = self.evaluate(self.model, test_loader, self.device)
            print(
                f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.weight_dir)
        self.model.load_state_dict(best_model_wts)
        print(f'Best test Acc: {best_acc:.4f}')

    def evaluate(self, model, dataloader, device):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.float(), labels.long()
                outputs = model(inputs.to(device))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        model.train()
        return accuracy_score(all_labels, all_preds)


def label_smoothing(true_labels, num_classes, epsilon=0.1):
    # print('true_labels', true_labels.shape[0])

    one_hot_labels = F.one_hot(true_labels, num_classes=num_classes).float()
    smoothed_labels = (1 - epsilon) * one_hot_labels + epsilon / num_classes
    # print('smoothed_labels.shape', smoothed_labels.shape)
    return smoothed_labels


class cross_entropy_with_label_smothing(nn.Module):
    def __init__(self, classes, epsilon=0.1):
        self.epsilon = epsilon
        self.classes = classes

    def forward(self, logits, true_labels):
        smoothed_labels = label_smoothing(true_labels, self.classes, self.epsilon)
        log_probs = F.log_softmax(logits, dim=1)
        # print('smoothed_labels', smoothed_labels.shape)
        # print('log_probs.shape', log_probs.shape)
        loss = -torch.sum(smoothed_labels * log_probs, dim=1).mean()
        # loss = torch.mean(torch.sum(-log_probs * smoothed_labels, dim=1))
        return loss
