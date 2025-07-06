from models.model2 import Net
from basic.train import *
from models.model_paper import Net_paper

if __name__ == '__main__':
    model = Net_paper(1, 7)
    path = r'D:\yuzhen\HUST\LOAD\use\domain3'
    save_dir = r'D:\yuzhen\HUST\LOAD\weights_paper\domain3.pth'
    just = DATA_ONE_DIMENSION(path, save_dir, model)
    just.train(10, 32)
