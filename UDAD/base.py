import torch
import os
import torch.nn as nn
from core.data import loader_onedimension
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

np.seterr(divide='ignore',invalid='ignore')

# 定义域判别器
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


def Validation(target_path, feature_extractor, classifier, device, batch_size=256):
    val_path = os.path.join(target_path, 'test')
    dataset, dataloader = loader_onedimension(val_path, batch_size)
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_x = []
    all_y = []
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            features = feature_extractor(data.float())
            outputs = classifier(features)
            all_x.append(outputs)
            all_y.append(label)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    accuracy = correct / total
    # print("Accuracy of the model on the target domain validation set {}%\n".format(accuracy))
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Confusion Matrix')
    plt.savefig('./new.png')
    print(f'Test_Accuracy: {accuracy*100:.4f}')
    plt.cla()
    plt.close("all")
    # print(f'Confusion Matrix:\n{cm}')

    all_x = torch.cat(all_x, dim=0).cpu().numpy()
    all_y = torch.cat(all_y, dim=0).cpu().numpy()
    all_x_flat = all_x.reshape(all_x.shape[0], -1)
    scaler = StandardScaler()
    features_std = scaler.fit_transform(all_x_flat)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features_std)
    hex_colors = ['#631f66', '#e2d8c4', '#258277', '#e6af30', '#b80101', '#d68784', '#393955', '#b1c44e', '#e6c737', '#008a9d']
    cmap = ListedColormap(hex_colors)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=all_y, cmap=cmap, s=10, alpha=0.8)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(r'./HUST出图/t.png')
    plt.cla()
    plt.close("all")
    ## print(f'acc: {100* acc.item() / x.shape[0]:.4f}' )

    feature_extractor.train()
    classifier.train()
