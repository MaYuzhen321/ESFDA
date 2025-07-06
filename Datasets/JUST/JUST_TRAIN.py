from models.model2 import Net, DATA_ONE_DIMENSION

'''
用于对PADE数据集进行训练，数据集中采样长度为512，训练集包含样本495个，测试集包含样本496个
'''
if __name__ == '__main__':
    model = Net(1, 4)
    path = r'G:\数据集\机械故障诊断数据集\JUST_USE\use\domain2'
    save_dir = r'G:\数据集\机械故障诊断数据集\JUST_USE\一维卷积测试权重\domain2.pth'
    just = DATA_ONE_DIMENSION(path, save_dir, model)
    just.train(10, 32)
