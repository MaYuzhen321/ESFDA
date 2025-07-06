'''
用于实现数据集的域划分：
1. 读取mat文件并读取振动信号, 保存至h5文件之中;
2. 按照比例划分训练集与测试集；
'''
from PIL import Image
import scipy.io as scio
from pylab import *

from tqdm import tqdm
import os
import scipy.io as scio
import h5py
import numpy as np
import pandas as pd
from torchvision import transforms, models
import random
import shutil

# 分段长度
SEGMENT_SIZE = 1024
# 采样重叠率
OVERLAP_RATE = 0.5
# 段数
MAX_SEGMENTS = 250
# 训练集测试集比例
SPLIT_RATIO = 0.5


def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)


'''
file_path: mat文件路径
target_path: h5文件目标路径
target_path
'''


def vibration_extract_pade(file_path):
    file_content = scio.loadmat(file_path)
    key = file_path.split('\\')[-1].split('.')[0]
    # print(key)
    temp = file_content[key]
    Y = temp['Y']
    vibration = Y[0][0][0][6]['Data']
    print('vibration.shape', vibration.shape)  # 2 56823  16008
    return vibration[0]


def vibration_extract_hust(file_path):
    file_content = scio.loadmat(file_path)
    vibration = file_content['data']
    return vibration


def signal_file_process_hust(file_path, target_path):
    destination_path = os.path.join(target_path, file_path.split('\\')[-1].split('.')[0] + '.h5')
    overlap_rate = OVERLAP_RATE
    vibration = vibration_extract_hust(file_path)
    # print(type(vibration))
    # print(vibration.shape)  # (256823,)
    overlap_samples = int(SEGMENT_SIZE * overlap_rate)
    effective_segment_size = SEGMENT_SIZE - overlap_samples
    data = vibration.reshape(
        -1)[:effective_segment_size * MAX_SEGMENTS]
    start_idx = 0
    end_idx = SEGMENT_SIZE
    with h5py.File(destination_path, 'w') as f:
        signal_group = f.create_group('signals')
        for i in range(MAX_SEGMENTS):
            if end_idx > len(data):  # 防越界
                break
            # print(start_idx)
            signal_group.create_dataset(
                f'signal_{i}', data=data[start_idx:end_idx])
            start_idx += effective_segment_size  # 更新start_idx时减去重叠部分
            end_idx = start_idx + MAX_SEGMENTS
            i += 1

            if end_idx > len(data):
                # print(i)
                break








def dir_process(data_set, path, after_process):
    domain_list = os.listdir(path)
    for domain in domain_list:
        # ensure_directory_exists(os.path.join(save_path, domain))
        domain_path = os.path.join(path, domain)
        # print(domain_path)
        file_list = os.listdir(domain_path)
        for file in file_list:
            file_path = os.path.join(domain_path, file)
            # print('file_path:',file_path)
            # print(file_path.split('\\')[-3:-1])
            des = os.path.join(after_process,
                               file_path.split('\\')[-2])
            # print(des)
            ensure_directory_exists(des)
            if data_set == "crwu":
                signal_file_process_crwu(file_path, des)
            elif data_set == "pade":
                print(os.listdir(file_path)[0])
                file_path = file_path + '\\' + os.listdir(file_path)[0]
                signal_file_process_pade(file_path, des)
            elif data_set == "hust":
                signal_file_process_hust(file_path, des)
            else:
                raise ValueError


def split_dataset(source_h5_path, train_h5_path, test_h5_path, split_ratio=0.7):
    with h5py.File(source_h5_path, 'r') as source_f:
        signal_group = source_f['signals']
        signal_names = list(signal_group.keys())
        # 打乱
        np.random.shuffle(signal_names)
        total_signals = len(signal_names)
        train_count = int(total_signals * split_ratio)
        test_count = total_signals - train_count
        # 创建训练集和测试集的HDF5文件
        with h5py.File(train_h5_path, 'w') as train_f, h5py.File(test_h5_path, 'w') as test_f:
            train_signals_group = train_f.create_group('signals')
            test_signals_group = test_f.create_group('signals')

            # 保存训练集信号
            for i in range(train_count):
                signal_name = signal_names[i]
                signal_data = signal_group[signal_name][:]
                train_signals_group.create_dataset(
                    signal_name, data=signal_data)

            # 保存测试集信号
            for i in range(train_count, total_signals):
                signal_name = signal_names[i]
                signal_data = signal_group[signal_name][:]
                test_signals_group.create_dataset(
                    signal_name, data=signal_data)

            # print(f"训练集信号个数: {train_count}, 测试集信号个数: {test_count}")


def prepare_used_data(separate_path, use_path, ratio=SPLIT_RATIO):
    domain_list = os.listdir(separate_path)
    for domain in domain_list:
        domain_path = os.path.join(separate_path, domain)
        a = os.path.join(use_path, domain)
        b = os.path.join(a, 'train')

        c = os.path.join(a, 'test')
        ensure_directory_exists(a)
        ensure_directory_exists(b)
        ensure_directory_exists(c)
        file_list = os.listdir(domain_path)
        for i in tqdm(range(len(file_list)), desc='单域数据处理'):
            file = file_list[i]
            # for file in file_list:
            # print(file)
            file_path = os.path.join(domain_path, file)
            d = os.path.join(b, file.split('.')[0])
            e = os.path.join(c, file.split('.')[0])
            ensure_directory_exists(e)
            ensure_directory_exists(d)
            split_dataset(file_path, os.path.join(d, file), os.path.join(e, file), ratio)


def all1(dataset, domain_path, save_path):
    split_path = os.path.join(save_path, 'split')
    use_path = os.path.join(save_path, 'use')
    # ensure_directory_exists(split_path)
    dir_process(dataset, domain_path, split_path)
    prepare_used_data(split_path, use_path)


def grey_img(domain_path, save_path):
    filenames = os.listdir(domain_path)
    for item in filenames:
        file_path = os.path.join(domain_path, item)
        file = scio.loadmat(file_path)
        # print(file_path)
        save_path1 = save_path + '\\' + \
                     domain_path.split('\\')[-1] + '\\' + \
                     file_path.split('\\')[-1].split('.')[0] + '\\'
        # print(save_path1)
        ensure_directory_exists(save_path1)
        for key in file.keys():
            if 'DE' in key:
                X = file[key]
                for i in range(MAX_SEGMENTS):
                    length = 4096
                    all_lenght = len(X)
                    random_start = np.random.randint(
                        low=0, high=(all_lenght - 2 * length))
                    sample = X[random_start:random_start + length]
                    sample = (sample - np.min(sample)) / \
                             (np.max(sample) - np.min(sample))
                    sample = np.round(sample * 255.0)
                    sample = sample.reshape(64, 64)
                    im = Image.fromarray(sample)
                    im.convert('L').save(save_path1 + str(key) +
                                         str(i) + '.jpg', format='jpeg')


def multi_domain_process(mdomain_path, separate_path):
    domain_list = os.listdir(mdomain_path)
    for domain in domain_list:
        domain_path = os.path.join(mdomain_path, domain)
        grey_img(domain_path, separate_path)


def split_dataset1(domain1, domain2, ratio):
    train_dir = os.path.join(domain2, 'train')
    test_dir = os.path.join(domain2, 'test')
    ensure_directory_exists(train_dir)
    ensure_directory_exists(test_dir)
    # for root, dirs, files in os.walk(domain1):
    #     for file in files:
    #         if file.endswith(('.jpg', '.png', '.jpeg')):
    #             source_path = os.path.join(root, file)
    #             target_dir = os.path.join(train_dir, dirs) if random.random() < ratio else os.path.join(test_dir, dirs)
    #             shutil.copy(source_path, os.path.join(target_dir, file))
    modes = os.listdir(domain1)
    # print(modes)
    for mode in modes:
        root = os.path.join(domain1, mode)
        files = os.listdir(root)
        print(files)
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                source_path = os.path.join(root, file)
                target_dir = os.path.join(train_dir, mode) if random.random() < ratio else os.path.join(test_dir, mode)
                ensure_directory_exists(target_dir)
                shutil.copy(source_path, os.path.join(target_dir, file))


def split_train_test(separate_path, use_path):
    domain_list = os.listdir(separate_path)
    for domain in domain_list:
        target_domain_path = os.path.join(use_path, domain)
        ensure_directory_exists(target_domain_path)
        split_dataset1(os.path.join(separate_path, domain),
                       os.path.join(use_path, domain), ratio=SPLIT_RATIO)


def all(domain_path, split_path, use_path):
    multi_domain_process(domain_path, split_path)
    split_train_test(split_path, use_path)


if __name__ == '__main__':
    # path1 = r'G:\数据集\华中科技大学数据集\HUST bearing dataset\0W\B500.mat'
    path1 = r'G:\数据集\华中科技大学数据集\HUST bearing dataset'
    path2 = 'G:\数据集\华中科技大学数据集\HUST'
    all1('hust', path1, path2)
