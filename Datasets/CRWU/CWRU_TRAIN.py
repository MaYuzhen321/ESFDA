from basic.train import DATA_ONE_DIMENSION
from models.model2 import Net
from models.model_paper import Net_paper

'''
用于对PADE数据集进行训练，数据集中采样长度为512，训练集包含样本495个，测试集包含样本496个
'''
if __name__ == '__main__':
    model = Net_paper(1, 10)
    path = r'D:\yuzhen\CRWU_1024\use\domain4'
    save_dir = r'D:\yuzhen\CRWU_1024\no_assistance\domain4.pth'
    cwru = DATA_ONE_DIMENSION(path, save_dir, model, criterion='smoothing')  # criterion = cross_entropy, smoothing
    cwru.train(10, 32)