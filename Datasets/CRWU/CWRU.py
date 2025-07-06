"""用于实现两个数据集的域划分：

1. 读取mat文件并读取振动信号, 保存至h5文件之中;
2. 按照比例划分训练集与测试集；"""

from basic.split_dataset import all

if __name__ == '__main__':
    dataset = 'crwu'
    domain_path = r'G:\数据集\机械故障诊断数据集\CRWU_4domain\origin'
    save_path = r'G:\数据集\机械故障诊断数据集\CRWU_for_Use'
    all(dataset=dataset, domain_path=domain_path, save_path=save_path)
