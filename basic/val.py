'''
用于在目标域测试域偏移的存在
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_root = r'G:\数据集\机械故障诊断数据集\CRWU_4domain\separated_domain\\'


def val(data_dir, best_model_wts):
    image_datasets = ImageFolder(data_dir, transforms['val'])

    dataloaders = DataLoader(image_datasets, batch_size=32, shuffle=True, num_workers=0)
    class_names = image_datasets.classes
    # print(class_names)
    class_num = len(class_names)
    if "vgg" in best_model_wts:
        model = models.vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, class_num)  # 修改输出层的维度
        model.load_state_dict(torch.load(best_model_wts))
    elif "resnet" in best_model_wts:
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, class_num)
        model.load_state_dict(torch.load(best_model_wts))
    else:
        raise ValueError("无网络")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度以节省内存和加速验证
        for images, labels in dataloaders:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算并打印准确率
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the {total} validation images: {accuracy}%')
