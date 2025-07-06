'''
用于实现两个数据集的域划分：

1. 读取mat文件并读取振动信号, 保存至h5文件之中;
2. 按照比例划分训练集与测试集；

'''

from tqdm import tqdm
import os
import scipy.io as scio
import h5py
import numpy as np
import pandas as pd
from torchvision import transforms, models

# 分段长度
SEGMENT_SIZE = 1024
# 采样重叠率
OVERLAP_RATE = 0.9
# 段数
MAX_SEGMENTS = 250
# 训练集测试集比例
SPLIT_RATIO = 0.5


def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)


def vibration_extract(file_path):
    file_content = scio.loadmat(file_path)
    key = file_path.split('\\')[-1].split('.')[0]
    # print(key)
    temp = file_content[key]
    Y = temp['Y']
    vibration = Y[0][0][0][6]['Data']
    print('vibration.shape', vibration.shape)  # 2 56823  16008
    return vibration[0]


def signal_file_process_pade(file_path, target_path):
    destination_path = os.path.join(
        target_path, file_path.split('\\')[-1].split('.')[0] + '.h5')
    # print(destination_path)  #G:\TEA_Fault_Diagnosis\data_process\PADE_one_dimension\B021_0.h5
    # file_content = scio.loadmat(file_path)
    overlap_rate = OVERLAP_RATE
    vibration = vibration_extract(file_path)
    print(type(vibration))
    # print(vibration.shape)  # (256823,)
    overlap_samples = int(SEGMENT_SIZE * overlap_rate)
    effective_segment_size = SEGMENT_SIZE - overlap_samples
    data = vibration.reshape(
        -1)[:effective_segment_size * MAX_SEGMENTS]
    start_idx = 0
    end_idx = effective_segment_size
    with h5py.File(destination_path, 'w') as f:
        signal_group = f.create_group('signals')
        for i in range(MAX_SEGMENTS):
            if end_idx > len(data):  # 防越界
                break
            # print(start_idx)
            signal_group.create_dataset(
                f'signal_{i}', data=data[start_idx:end_idx])
            start_idx += effective_segment_size  # 更新start_idx时减去重叠部分
            end_idx = start_idx + effective_segment_size
            i += 1

            if end_idx > len(data):
                # print(i)
                break


def signal_file_process_crwu(file_path, target_path):
    destination_path = os.path.join(
        target_path, file_path.split('\\')[-1].split('.')[0] + '.h5')
    file_content = scio.loadmat(file_path)
    overlap_rate = OVERLAP_RATE
    for key in file_content.keys():
        if 'DE' in key:
            overlap_samples = int(SEGMENT_SIZE * overlap_rate)
            # 调整每段的实际长度以考虑重叠
            effective_segment_size = SEGMENT_SIZE - overlap_samples
            # print(file_content[key].reshape(-1).shape) # (122571,)
            # 确保数据足够
            data = file_content[key].reshape(
                -1)[:effective_segment_size * MAX_SEGMENTS]
            start_idx = 0
            end_idx = effective_segment_size
            with h5py.File(destination_path, 'w') as f:
                signal_group = f.create_group('signals')  # 用于存放多个信号段

                for i in range(MAX_SEGMENTS):
                    if end_idx > len(data):  # 防越界
                        break
                    # print(start_idx)
                    signal_group.create_dataset(
                        f'signal_{i}', data=data[start_idx:end_idx])
                    start_idx += effective_segment_size  # 更新start_idx时减去重叠部分
                    end_idx = start_idx + effective_segment_size
                    i += 1
                    if end_idx > len(data):
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
                # print(os.listdir(file_path)[0])
                file_path = file_path + '\\' + os.listdir(file_path)[0]
                signal_file_process_pade(file_path, des)
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
                # print(signal_data.shape)
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
            split_dataset(file_path, os.path.join(d, file),
                          os.path.join(e, file), ratio)


def all(dataset, domain_path, save_path):
    split_path = os.path.join(save_path, 'split')
    use_path = os.path.join(save_path, 'use')
    # ensure_directory_exists(split_path)
    dir_process(dataset, domain_path, split_path)
    prepare_used_data(split_path, use_path)


if __name__ == '__main__':
    # 单个文件处理
    # path = r'G:\数据集\机械故障诊断数据集\CRWU_4domain\origin\1797_12K_load0\B021_0.mat'
    # path2 = r'G:\TEA_Fault_Diagnosis\data_process\PADE_one_dimension'
    # signal_file_process(path,path2)

    # 域处理
    # path = r'G:\数据集\机械故障诊断数据集\CRWU_4domain\origin'
    # save_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\separate'
    # dir_process(path,save_path)

    # ## 数据集划分
    # separate_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\separate'
    # use_path = r'G:\数据集\机械故障诊断数据集\CRWU_D\use'
    # prepare_used_data(separate_path,use_path)

    # 处理PADE数据集
    # 单个文件处理
    # path = r'G:\PADE\Domain\Domain1\K001\N09_M07_F10_K001_1.mat'
    # path2 = r'G:\PADE\weights_pade'
    # signal_file_process_pade(path, path2)

    # 域处理
    # path = r'G:\PADE\Domain'
    # save_path = r'G:\PADE\序列直输\split'
    # dir_process('pade',path,save_path)

    # 数据集划分
    # separate_path = r'G:\PADE\序列直输\split'
    # use_path = r'G:\PADE\序列直输\use'
    # prepare_used_data(separate_path,use_path)
    all('pade', r'G:\PADE\Domain', r'G:\PADE\b')
