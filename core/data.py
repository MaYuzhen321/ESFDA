from torchvision.datasets import ImageFolder
from core.train_sequential import *
from models.model_paper import *


def set_transform(dataset):
    if dataset == 'CRWU':
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    else:
        raise
    return transform_train, transform_test


def load_CRWU_tiny(n_examples, data_dir, data_transforms):
    image_datasets = ImageFolder(data_dir, data_transforms)
    batch_size = 32
    dataloader = DataLoader(image_datasets, batch_size, shuffle=True, num_workers=0)
    x_test, y_test = [], []
    for i, (x, y) in enumerate(dataloader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    x_test_tensor = x_test_tensor[:n_examples]
    y_test_tensor = y_test_tensor[:n_examples]
    return x_test_tensor, y_test_tensor


# 定义函数用于加载序列数据 需要返回数据以及对应的标签
def load_sequential(n_examples, data_dir):
    batch_size = 32
    # print('data_dir', data_dir)
    crwu = CRWU_p(data_dir, "resnet50")
    train_data, train_labels = crwu._load_and_adjust_data(data_dir)
    train_dataset = VibrationDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x_test, y_test = [], []
    for i, (x, y) in enumerate(train_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    x_test_tensor = x_test_tensor[:n_examples]
    y_test_tensor = y_test_tensor[:n_examples]
    return x_test_tensor, y_test_tensor


def load_onedimension(n_examples, data_dir, batch_size):
    data_one_dimension = DATA_ONE_DIMENSION(data_dir, None)
    train_data, train_labels = data_one_dimension.load_data(data_dir)
    train_dataset = VibrationDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x_test, y_test = [], []
    for i, (x, y) in enumerate(train_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    x_test_tensor = x_test_tensor[:n_examples]
    y_test_tensor = y_test_tensor[:n_examples]
    return x_test_tensor, y_test_tensor

def loader_onedimension(data_dir, batch_size):
    data_one_dimension = DATA_ONE_DIMENSION(data_dir, None)
    train_data, train_labels = data_one_dimension.load_data(data_dir)
    train_dataset = VibrationDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, train_loader

def load_data(data, n_examples=None, severity=None, data_dir=None, shuffle=False, corruptions=None):
    if data == 'CRWU':
        _, transform = set_transform(data)
        x_test, y_test = load_CRWU_tiny(n_examples, data_dir, transform)
    elif data == 'sequential':
        x_test, y_test = load_sequential(n_examples, data_dir)
        x_test = x_test.float()
        y_test = y_test.long()
    else:
        raise
    # print(x_test.shape, n_examples)
    return x_test, y_test


def load_data_onedimension(data, n_examples=None, severity=None, data_dir=None, shuffle=False, corruptions=None):
    if data == 'CRWU':
        _, transform = set_transform(data)
        x_test, y_test = load_CRWU_tiny(n_examples, data_dir, transform)
    elif data == 'sequential':
        x_test, y_test = load_sequential(n_examples, data_dir)
        x_test = x_test.float()
        y_test = y_test.long()
    elif data == 'one_dimension':
        x_test, y_test = load_onedimension(n_examples, data_dir)
        x_test = x_test.float()
        y_test = y_test.long()
    else:
        raise
    # print(x_test.shape, n_examples)
    return x_test, y_test
