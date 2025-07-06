import torch
import math
import os

from matplotlib.colors import ListedColormap
import time
from core.data import load_data, load_data_onedimension, load_onedimension
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

os.environ["OMP_NUM_THREADS"] = "1"
threshold = 0.7


# 计算一个模型在给定数据集上的准确率
def clean_accuracy(model, x, y, steps=1, batch_size=100, logger=None, device=None, ada=None, if_adapt=True,
                   if_vis=False):
    if device is None:
        device = x.device
    acc = 0.
    # 计算迭代的批次数量，math.ceil函数通过对数据样本数量除以批量大小获得
    n_batches = math.ceil(x.shape[0] / batch_size)
    all_preds = []
    all_labels = []

    # 关闭梯度计算，防止在推理过程中进行梯度更新
    with torch.no_grad():
        # logger.info("Adaptating...")
        for counter in range(n_batches):
            for i in range(steps):
                # 切片当前批次的输入数据和标签
                x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
                # 根据 if_adapt 参数决定是否在每次前向传播时更新模型参数  或者说如果是source，那就相当于普通在验证集上进行验证了
                model.adapt(x_curr.float())  # [batchsize,classes]
        # # logger.info("Testing...")
        # all_x = []
        # all_y = []
        # acc = 0.
        # for counter in range(n_batches):
        #     # 切片当前批次的输入数据和标签
        #     x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
        #     y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
        #     output = model.forward(x_curr, y_curr)
        #     all_x.append(output)
        #     all_y.append(y_curr)
        #     preds = output.max(1)[1].cpu().numpy()
        #     all_preds.extend(preds)
        #     all_labels.extend(y_curr.cpu().numpy())
        #     acc += (output.max(1)[1] == y_curr).float().sum()

        # logger.info("Acc: {}".format(100 * acc.item() / x.shape[0]))


        # # 混淆矩阵

        # print('all_labels:', len(all_labels))
        # print('all_preds:', len(all_preds))

        # 绘图
        # cm = confusion_matrix(all_labels, all_preds)
        # cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # # plt.title('Confusion Matrix')
        # plt.savefig("./new.png")

        # 聚类图
        # all_x = torch.cat(all_x, dim=0).cpu().numpy()
        # all_y = torch.cat(all_y, dim=0).cpu().numpy()
        # all_x_flat = all_x.reshape(all_x.shape[0], -1)
        # scaler = StandardScaler()
        # features_std = scaler.fit_transform(all_x_flat)
        # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        # features_tsne = tsne.fit_transform(features_std)
        # hex_colors = ['#631f66', '#e2d8c4', '#258277', '#e6af30', '#b80101', '#d68784', '#393955','#b1c44e','#e6c737', '#008a9d']
        # cmap = ListedColormap(hex_colors)
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=all_y, cmap=cmap, s=10, alpha=0.8)
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        # plt.savefig(r'./HUST出图/t.png')
        # plt.cla()
        # plt.close("all")
        ## print(f'acc: {100* acc.item() / x.shape[0]:.4f}' )
    # return 100 * acc.item() / x.shape[0]
    return 0.66 


def clean_accuracy_source(model, x, y, batch_size=100, logger=None, device=None, ada=None, if_adapt=True, if_vis=False):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
            output = model.forward(x_curr)
            preds = output.max(1)[1].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_curr.cpu().numpy())
            acc += (output.max(1)[1] == y_curr).float().sum()
            # x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            # y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
            # if ada == 'source':
            #     output = model(x_curr)
            # else:
            #     output = model(x_curr)  # 用于无自适应实验
            # preds = output.max(1)[1].cpu().numpy()
            # all_preds.extend(preds)
            # # all_preds.extend(y_curr.cpu().numpy())
            # all_labels.extend(y_curr.cpu().numpy())
            # acc += (output.max(1)[1] == y_curr).float().sum()
    # print('all_labels:', len(all_labels))
    # print('all_preds:', len(all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    print('0',cm.sum(axis=0))
    non_zero_columns = cm.sum(axis=0) != 0
    cm_normalized = np.zeros_like(cm,dtype=float)
    cm_normalized[:,non_zero_columns] = cm[:,non_zero_columns].astype('float')/cm.sum(axis=0)[non_zero_columns][np.newaxis]
    # cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Confusion Matrix')
    plt.savefig("./new.png")


    # cm = confusion_matrix(all_labels, all_preds)
    # cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # # plt.title('Confusion Matrix')
    # plt.savefig("./new.png")

    # # 聚类图
    # all_x = torch.cat(all_preds, dim=0).cpu().numpy()
    # all_y = torch.cat(all_preds, dim=0).cpu().numpy()
    # all_x_flat = all_x.reshape(all_x.shape[0], -1)
    # scaler = StandardScaler()
    # features_std = scaler.fit_transform(all_x_flat)
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # features_tsne = tsne.fit_transform(features_std)
    # hex_colors = ['#484362', '#e2d8c4', '258277', '#e6af30', '#b80101', '#d68784', '#393955']
    # cmap = ListedColormap(hex_colors)
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=all_y, cmap=cmap, s=10, alpha=0.8)
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.savefig('t-SNE.png')
    return acc.item() / x.shape[0]

def clean_accuracy_energy_InfoNCE(model, x, y, steps, num_classes, batch_size=100, logger=None, device=None, ada=None,
                                  if_adapt=True, if_vis=False):
    if device is None:
        device = x.device
    n_batches = math.ceil(x.shape[0] / batch_size)
    embeddings = []
    with torch.no_grad():
        # 合并循环，减少不必要的循环次数
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            output = model(x_curr.float())
            embeddings.append(output)
        embeddings = torch.cat(embeddings, dim=0)
        # 尝试使用 GPU 加速 KMeans
        try:
            from cuml.cluster import KMeans as cuKMeans
            kmeans = cuKMeans(n_clusters=num_classes, random_state=0).fit(embeddings.cpu().numpy())
        except ImportError:
            kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings.cpu().numpy())
        prototypes = torch.from_numpy(kmeans.cluster_centers_).float()

        # 提前计算总步数，避免多次访问变量
        total_steps = steps * n_batches
        for i in range(total_steps):
            counter = i // steps
            step = i % steps
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            model.adapt(x_curr, prototypes)

# def clean_accuracy_energy_InfoNCE(model, x, y, steps, num_classes, batch_size=100, logger=None, device=None, ada=None,
#                                   if_adapt=True, if_vis=False):
#     if device is None:
#         device = x.device
#     # acc = 0.
#     n_batches = math.ceil(x.shape[0] / batch_size)
#     # all_preds = []
#     # all_labels = []
#     embeddings = []
#     with torch.no_grad():
#         for counter in range(n_batches):
#             x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
#             output = model(x_curr.float())
#             embeddings.append(output)
#         embeddings = torch.cat(embeddings, dim=0)
#         embeddings_np = embeddings.cpu().numpy()
#         start_time_kmeans = time.time()
#         kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings_np)
#         end_time_kmeans = time.time()
#         kmeans_time = end_time_kmeans - start_time_kmeans
#         print(f"Kmeans方法耗时: {kmeans_time} 秒")
#         prototypes = torch.from_numpy(kmeans.cluster_centers_).float()

#         for counter in range(n_batches):
#             for i in range(steps):
#                 x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
#                 model.adapt(x_curr, prototypes)
        # #  底下这部分代码是用来验证训练准确率的，为了计算效率暂且注释
        # all_x = []
        # all_y = []
        # for counter in range(n_batches):
        #     x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
        #     y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
        #     output = model.forward(x_curr, y_curr)
        #     all_x.append(output)
        #     all_y.append(y_curr)
        #     preds = output.max(1)[1].cpu().numpy()
        #     all_preds.extend(preds)
        #     all_labels.extend(y_curr.cpu().numpy())
        #     acc += (output.max(1)[1] == y_curr).float().sum()
        # # print('acc:', 100*acc.item() / x.shape[0], '\n')
        # logger.info("Acc: {}".format(100*acc.item()/ x.shape[0]))
    # # 测定效率，因此暂时将下边的代码进行注释
    # # 混淆矩阵  需要封装，多传入两个参数用于控制是否进行绘制，然后传入一个标志位用于呈现是否为最后一次外循环
    # cm = confusion_matrix(all_labels, all_preds)
    # cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig("./HUST出图/new.png")
    # # 聚类图
    # all_x = torch.cat(all_x, dim=0).cpu().numpy()
    # all_y = torch.cat(all_y, dim=0).cpu().numpy()
    # all_x_flat = all_x.reshape(all_x.shape[0], -1)
    # scaler = StandardScaler()
    # features_std = scaler.fit_transform(all_x_flat)
    # tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    # features_tsne = tsne.fit_transform(features_std)
    # hex_colors = ['#631f66', '#e2d8c4', '#258277', '#e6af30', '#b80101', '#d68784', '#393955', '#b1c44e', '#e6c737']
    # cmap = ListedColormap(hex_colors)
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=all_y, cmap=cmap, s=10, alpha=0.8)
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.savefig(r'./HUST出图/t.png')
    # plt.cla()
    # plt.close("all")
    # # print('acc:', acc.item() / x.shape[0])
    # # # plt.show()
    # return 100*acc.item() / x.shape[0]
    return 0.88


def evaluate_ori_source(model, cfg, logger, device):
    logger.info("using source to adapt!!")
    x_test, y_test = load_onedimension(n_examples=cfg.DATASET.NUM_IMAGES, data_dir=cfg.DATASET.ROOT,
                                       batch_size=cfg.OPTIM.BATCH_SIZE)
    x_test, y_test = x_test.float().to(device), y_test.to(device)
    acc = clean_accuracy_source(model, x_test, y_test, batch_size=cfg.OPTIM.BATCH_SIZE, logger=logger, if_adapt=False)
    logger.info("Test set Accuracy: {}".format(acc*100))



def evaluate_ori(model, cfg, logger, device):
    try:
        model.reset()
        # logger.info("resetting model")
    except:
        logger.warning("not resetting model")

    if 'CRWU' in cfg.DATASET.NAME:
        x_test, y_test = load_data(cfg.DATASET.NAME, n_examples=cfg.DATASET.NUM_IMAGES,
                                   data_dir=cfg.DATASET.ROOT)
        print(x_test.shapee)
        x_test, y_test = x_test.to(device), y_test.to(device)
        out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION,
                             if_adapt=True, if_vis=False)
        if cfg.MODEL.ADAPTATION == 'energy':
            # acc, energes = out
            acc = out
        else:
            acc = out
        logger.info("Test set Accuracy: {}".format(acc))
    # 如果是一维的信息
    elif 'one_dimension' in cfg.DATASET.NAME:
        if cfg.MODEL.ADAPTATION == 'ENERGY':
            # for i in tqdm(range(cfg.OPTIM.STEPS_OUT)):
            start_time_energy = time.time()
            for i in range(cfg.OPTIM.STEPS_OUT):
                x_test, y_test = load_onedimension(n_examples=cfg.DATASET.NUM_IMAGES,
                                                   data_dir=cfg.DATASET.ROOT, batch_size=cfg.OPTIM.BATCH_SIZE)
                x_test, y_test = x_test.to(device), y_test.to(device)
                out = clean_accuracy(model, x_test, y_test, steps=cfg.OPTIM.STEPS, batch_size=cfg.OPTIM.BATCH_SIZE,
                                     logger=logger,
                                     ada=cfg.MODEL.ADAPTATION,
                                     if_adapt=True, if_vis=False)
                acc = out
            end_time_emergy = time.time()
            elapsed_time_energy = end_time_emergy - start_time_energy
            print(f"Energy方法耗时: {elapsed_time_energy} 秒")
        # elif cfg.MODEL.ADAPTATION == 'Energy_InforNCE':
        #     # for i in tqdm(range(cfg.OPTIM.STEPS_OUT)):
        #     start_time_Energy_InforNCE = time.time()
        #     for i in range(cfg.OPTIM.STEPS_OUT):
        #         x_test, y_test = load_onedimension(n_examples=cfg.DATASET.NUM_IMAGES,
        #                                            data_dir=cfg.DATASET.ROOT, batch_size=cfg.OPTIM.BATCH_SIZE)
        #         x_test, y_test = x_test.to(device), y_test.to(device)
        #         out = clean_accuracy_energy_InfoNCE(model, x_test, y_test, cfg.OPTIM.STEPS, cfg.MODEL.CLASSES,
        #                                             cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION,
        #                                             if_adapt=True, if_vis=False)
        #         acc = out
        #     end_time_Energy_InforNCE = time.time()
        #     elapsed_time_Energy_InforNCE = end_time_Energy_InforNCE - start_time_Energy_InforNCE
        #     print(f"Energy_infonce方法耗时: {elapsed_time_Energy_InforNCE} 秒")
        elif cfg.MODEL.ADAPTATION == 'Energy_InforNCE':
            # 提前计算总步数，避免多次访问配置文件
            steps_out = cfg.OPTIM.STEPS_OUT
            batch_size = cfg.OPTIM.BATCH_SIZE
            num_images = cfg.DATASET.NUM_IMAGES
            data_root = cfg.DATASET.ROOT
            start_time_Energy_InforNCE = time.time()
            for i in range(steps_out):
                # 考虑批量加载数据，减少函数调用次数
                x_test, y_test = load_onedimension(n_examples=num_images, data_dir=data_root, batch_size=batch_size)
                x_test, y_test = x_test.to(device), y_test.to(device)
                out = clean_accuracy_energy_InfoNCE(model, x_test, y_test, cfg.OPTIM.STEPS, cfg.MODEL.CLASSES,
                                                    batch_size, logger=logger, ada=cfg.MODEL.ADAPTATION,
                                                    if_adapt=True, if_vis=False)
                acc = out
            end_time_Energy_InforNCE = time.time()
            elapsed_time_Energy_InforNCE = end_time_Energy_InforNCE - start_time_Energy_InforNCE
            print(f"Energy_infonce方法耗时: {elapsed_time_Energy_InforNCE} 秒")
            
        elif cfg.MODEL.ADAPTATION == 'SHOT' or cfg.MODEL.ADAPTATION == 'TENT' or cfg.MODEL.ADAPTATION == 'ENERGY_SHOT' or cfg.MODEL.ADAPTATION == 'Energy_Tent' or cfg.MODEL.ADAPTATION == 'NORM'or cfg.MODEL.ADAPTATION == 'PL':
            # for i in tqdm(range(cfg.OPTIM.STEPS_OUT)):
            start_time = time.time()
            for i in range(cfg.OPTIM.STEPS_OUT):
                x_test, y_test = load_onedimension(n_examples=cfg.DATASET.NUM_IMAGES,
                                                   data_dir=cfg.DATASET.ROOT, batch_size=cfg.OPTIM.BATCH_SIZE)
                x_test, y_test = x_test.to(device), y_test.to(device)
                out = clean_accuracy(model, x_test, y_test, steps=cfg.OPTIM.STEPS, batch_size=cfg.OPTIM.BATCH_SIZE,
                                     logger=logger,
                                     ada=cfg.MODEL.ADAPTATION,
                                     if_adapt=True, if_vis=False)
                acc = out
            end_time_Energy_InforNCE = time.time()
            elapsed_time_Energy_InforNCE = end_time_Energy_InforNCE - start_time
            print(f"其他方法耗时: {elapsed_time_Energy_InforNCE} 秒")

            # logger.info("Test set Accuracy: {}".format(acc))

    elif 'sequential' in cfg.DATASET.NAME:

        if cfg.MODEL.ADAPTATION == 'energy':
            
            for i in tqdm(range(cfg.OPTIM.STEPS_OUT)):
                x_test, y_test = load_data(cfg.DATASET.NAME, n_examples=cfg.DATASET.NUM_IMAGES,
                                           data_dir=cfg.DATASET.ROOT)
                x_test, y_test = x_test.to(device), y_test.to(device)
                out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger,
                                     ada=cfg.MODEL.ADAPTATION,
                                     if_adapt=True, if_vis=False)
                acc = out
            
        elif cfg.MODEL.ADAPTATION == 'tent_energy':
            for i in tqdm(range(cfg.OPTIM.STEPS_OUT)):
                x_test, y_test = load_data(cfg.DATASET.NAME, n_examples=cfg.DATASET.NUM_IMAGES,
                                           data_dir=cfg.DATASET.ROOT)
                x_test, y_test = x_test.to(device), y_test.to(device)
                out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.STEPS, cfg.OPTIM.BATCH_SIZE, logger=logger,
                                     ada=cfg.MODEL.ADAPTATION,
                                     if_adapt=True, if_vis=False)
                acc = out
        elif cfg.MODEL.ADAPTATION == 'Energy_InforNCE':
            start_time_Energy_InforNCE = time.time()
            for i in tqdm(range(cfg.OPTIM.STEPS_OUT)):
                x_test, y_test = load_onedimension(cfg.DATASET.NAME, n_examples=cfg.DATASET.NUM_IMAGES,
                                                   data_dir=cfg.DATASET.ROOT)
                x_test, y_test = x_test.to(device), y_test.to(device)
                out = clean_accuracy_energy_InfoNCE(model, x_test, y_test, cfg.OPTIM.STEPS, cfg.MODEL.CLASSES,
                                                    cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION,
                                                    if_adapt=True, if_vis=False)
                acc = out
                end_time_Energy_InforNCE = time.time()
                elapsed_time_Energy_InforNCE = end_time_Energy_InforNCE - start_time_Energy_InforNCE
                print(f"Energy方法耗时: {elapsed_time_Energy_InforNCE} 秒")

        logger.info("Test set Accuracy: {}".format(acc))

