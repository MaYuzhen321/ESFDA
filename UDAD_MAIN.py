from torch.cuda import device
import time
from UDAD import *
import torch
import torch.nn as nn
from core.data import loader_onedimension
from models.model2 import encoder, classifier


def udad(method, path_source, parth_target, batch_size, epoch, class_nums, device, classifier):
    _, dataloader_source = loader_onedimension(path_source, batch_size=batch_size)
    _, dataloader_target = loader_onedimension(parth_target, batch_size=batch_size)
    feature_extractor = encoder(1).to(device)
    classifier = classifier(class_nums).to(device)

    criterion = nn.CrossEntropyLoss()
    domain_discriminator = DomainDiscriminator(input_dim=2048 * 2).to(device)

    if method == "DAD":
        return DAD(epoch, parth_target, device, dataloader_source, dataloader_target, feature_extractor, classifier,
                   criterion, domain_discriminator)
    elif method == "CoAD":
        return COAD(parth_target, epoch, device, dataloader_source, dataloader_target, criterion, feature_extractor,
                    classifier)
    elif method == "MMD":

        return MAD(epoch, parth_target, device, dataloader_source, dataloader_target, feature_extractor, classifier,
                   domain_discriminator, criterion)
    else:
        raise "No Such UDAD Method!!!"


if __name__ == '__main__':
    methods = ['MMD', 'CoAD', 'DAD']
    path_source = r'D:\yuzhen\CRWU_1024\use\domain1'
    parth_target = r'D:\yuzhen\CRWU_1024\use\domain2'
    batch_size = 64
    epoch = 10
    class_nums = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for i in range(10):
    #     # print(f'实验组别：{i+1}')
    #     print("Experiment_num: {}".format(i+1))

    # model = udad(method=methods[0], path_source=path_source, parth_target=parth_target, batch_size=batch_size,
    #          epoch=epoch,
    #          class_nums=class_nums, device=device, classifier=classifier)
    # model.train()


    # for i in range(10):
    #     # print(f'实验组别：{i+1}')
    #     print("Experiment_num: {}".format(i + 1))


    model = udad(method=methods[1], path_source=path_source, parth_target=parth_target, batch_size=batch_size,
                 epoch=epoch,
                 class_nums=class_nums, device=device, classifier=classifier)
    model.train()


    # for i in range(10):
    #     # print(f'实验组别：{i+1}')
    #     print("Experiment_num: {}".format(i + 1))
    #     model = udad(method=methods[2], path_source=path_source, parth_target=parth_target, batch_size=batch_size,
    #                  epoch=epoch,
    #                  class_nums=class_nums, device=device, classifier=classifier)
    #     model.train()
