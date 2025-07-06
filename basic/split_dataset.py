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

# 分段长度
SEGMENT_SIZE = 1024
# 采样重叠率
OVERLAP_RATE = 0.9
# 段数
MAX_SEGMENTS = 1000
# 训练集测试集比例
SPLIT_RATIO = 0.5


def ensure_directory_exists(directory):
    """
    确保指定路径存在，如不存在则创建
    """
    os.makedirs(directory, exist_ok=True)


def vibration_extract(file_path):
    """
    从给定路径提取振动信号
    """
    file_content = scio.loadmat(file_path)
    key = file_path.split('\\')[-1].split('.')[0]
    temp = file_content[key]
    Y = temp['Y']
    vibration = Y[0][0][0][6]['Data']
    return vibration[0]


def vibration_extract_hust(file_path):
    file_content = scio.loadmat(file_path)
    vibration = file_content['data']
    return vibration


def vibration_extract_just(file_path):
    df = pd.read_csv(file_path)
    # print(df.head()) # 读取前五行  1, 4, 5, 6是垂直信号，2，3是水平信号
    vibration1 = df['AI 1 (m/s2)']
    # print(vibration1.shape)
    vibration = np.array([vibration1])
    return vibration


def signal_file_process_pade(file_path, target_path):
    """
    用于PADE数据文件的处理：
    - 将振动序列根据预设划分为小段
    - 将小段数据保存在h5文件之中
    """
    destination_path = os.path.join(
        target_path, file_path.split('\\')[-1].split('.')[0] + '.h5')
    overlap_rate = OVERLAP_RATE
    vibration = vibration_extract(file_path)
    # print(type(vibration))
    overlap_samples = int(SEGMENT_SIZE * overlap_rate)
    effective_segment_size = SEGMENT_SIZE - overlap_samples
    data = vibration.reshape(-1)[:effective_segment_size * MAX_SEGMENTS]
    start_idx = 0
    end_idx = SEGMENT_SIZE
    with h5py.File(destination_path, 'w') as f:
        signal_group = f.create_group('signals')
        for i in range(MAX_SEGMENTS):
            # print('i',i)
            if end_idx > len(data):  # 防越界
                break
            signal_group.create_dataset(f'signal_{i}', data=data[start_idx:end_idx])
            # print(end_idx - start_idx)
            start_idx += effective_segment_size  # 更新start_idx时减去重叠部分

            end_idx = start_idx + SEGMENT_SIZE
            i += 1


def signal_file_process_cwru(file_path, target_path):
    """
    用于CWRU数据文件的处理：
    - 将振动序列根据预设划分为小段
    - 将小段数据保存在h5文件之中
    """
    destination_path = os.path.join(
        target_path, file_path.split('\\')[-1].split('.')[0] + '.h5')
    file_content = scio.loadmat(file_path)
    overlap_rate = OVERLAP_RATE
    for key in file_content.keys():
        if 'DE' in key:
            overlap_samples = int(SEGMENT_SIZE * overlap_rate)
            effective_segment_size = SEGMENT_SIZE - overlap_samples
            data = file_content[key].reshape(
                -1)[:effective_segment_size * MAX_SEGMENTS]
            start_idx = 0
            end_idx = SEGMENT_SIZE
            with h5py.File(destination_path, 'w') as f:
                signal_group = f.create_group('signals')
                for i in range(MAX_SEGMENTS):
                    if end_idx > len(data):
                        break
                    signal_group.create_dataset(
                        f'signal_{i}', data=data[start_idx:end_idx])
                    start_idx += effective_segment_size
                    end_idx = start_idx + SEGMENT_SIZE
                    i += 1
                    if end_idx > len(data):
                        break


def signal_file_process_just(file_path, target_path):
    destination_path = os.path.join(
        target_path, file_path.split('\\')[-1].split('.')[0] + '.h5')
    overlap_rate = OVERLAP_RATE
    vibration = vibration_extract_just(file_path)
    overlap_samples = int(SEGMENT_SIZE * overlap_rate)
    effective_segment_size = SEGMENT_SIZE - overlap_samples
    data = vibration[:, :effective_segment_size * MAX_SEGMENTS]
    start_idx = 0
    end_idx = SEGMENT_SIZE
    with h5py.File(destination_path, 'w') as f:
        signal_group = f.create_group('signals')
        for i in range(MAX_SEGMENTS):
            if end_idx > len(data[0]):  # 防越界
                break
            signal_group.create_dataset(
                f'signal_{i}', data=data[:, start_idx:end_idx])
            start_idx += effective_segment_size  # 更新start_idx时减去重叠部分
            end_idx = start_idx + SEGMENT_SIZE
            i += 1
            if end_idx > len(data[0]):
                break


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
            end_idx = start_idx + SEGMENT_SIZE
            i += 1

            if end_idx > len(data):
                # print(i)
                break


def dir_process(data_set, path, after_process):
    """
    处理完整的文件夹
    data_set: crwu or pade
    path: 文件夹路径
    after_process: 处理后的文件夹路径
    """
    domain_list = os.listdir(path)
    for domain in domain_list:
        domain_path = os.path.join(path, domain)
        file_list = os.listdir(domain_path)
        for file in file_list:
            file_path = os.path.join(domain_path, file)
            des = os.path.join(after_process,
                               file_path.split('\\')[-2])
            ensure_directory_exists(des)
            if data_set == "cwru":
                signal_file_process_cwru(file_path, des)
            elif data_set == "pade":
                file_path = file_path + '\\' + os.listdir(file_path)[0]
                signal_file_process_pade(file_path, des)
            elif data_set == "just":
                signal_file_process_just(file_path, des)
            elif data_set == "hust":
                signal_file_process_hust(file_path, des)
            else:
                raise ValueError


def split_dataset(source_h5_path, train_h5_path, test_h5_path, split_ratio=0.7):
    """
    按照比例将h5中信息划分为训练集和测试集
    source_h5_path: h5文件路径
    train_h5_path: 训练h5路径
    test_h5_path: 测试h5路径
    split_ratio: 划分比例
    """
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
    """
    对整个数据集进行训练集和测试集的分割
    separate_path: 单域数据集路径
    use_path: 训练集和测试集保存路径
    ratio: 划分比例
    """
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
            file_path = os.path.join(domain_path, file)
            d = os.path.join(b, file.split('.')[0])
            e = os.path.join(c, file.split('.')[0])
            ensure_directory_exists(e)
            ensure_directory_exists(d)
            split_dataset(file_path, os.path.join(d, file),
                          os.path.join(e, file), ratio)


def all(dataset, domain_path, save_path):
    """
    完整的单个数据集处理函数
    - data_path: 'pade' or 'crwu'
    - domain_path: 数据集路径
    - save_path: 保存路径
    """
    split_path = os.path.join(save_path, 'split')
    use_path = os.path.join(save_path, 'use')
    # ensure_directory_exists(split_path)
    dir_process(dataset, domain_path, split_path)
    prepare_used_data(split_path, use_path)


if __name__ == '__main__':
    all('pade', r'D:\yuzhen\PADE\Domain', r'D:\yuzhen\PADE\512')
