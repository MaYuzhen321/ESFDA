import torch
import torch.nn as nn
from copy import deepcopy
# from tea_relevent.core.param import load_model_and_optimizer, copy_model_and_optimizer
from torchvision.utils import save_image
import os
from torch.nn import functional as F
import logging


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def init_random(bs, im_sz=32, n_ch=3):
    return torch.FloatTensor(bs, n_ch, im_sz).uniform_(-1, 1)


class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):  # x的通道数是3?
        penult_z = self.f(x.float())
        return penult_z

    def forward(self, x, y=None):
        logits = self.classify(x)  # [batch,classes]
        if y is None:
            probabilities = F.softmax(logits, dim=-1)
            epsilon = 1e-12
            # [batchsize,]
            entropy = torch.sum(-probabilities * torch.log(probabilities + epsilon), dim=-1)
            result_tensor = (entropy < 1.4).to(torch.int)
            k = logits.logsumexp(1)
            # p = k * (10 * result_tensor + 1)
            # return p, logits  # 返回计算得到的能量以及logits
            # return "wrong"
            # print('能量：', k)
            return k, logits  # 返回计算得到的能量以及logits

        else:
            return torch.gather(logits, 1, y[:, None]), logits


def sample_p_0(reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs, im_sz=im_sz, n_ch=n_ch), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)
    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None]
    samples = choose_random * random_samples + \
              (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    """
    this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    bs = batch_size if y is None else y.size(0)
    init_sample, buffer_inds = sample_p_0(reinit_freq=reinit_freq, replay_buffer=replay_buffer, bs=bs, im_sz=im_sz,
                                          n_ch=n_ch, device=device, y=y)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)

    for k in range(n_steps):
        f_prime = torch.autograd.grad(
            f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples, init_samples.detach()


def sample_odl(f, replay_buffer, n_steps, odl_lr, odl_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    # Set the model to evaluation mode if necessary
    # model.eval()
    f.eval()
    # Determine batch size
    batch_size = len(y) if y is not None else batch_size

    # Sample initial points from replay buffer or initialize them randomly
    if replay_buffer is not None and len(replay_buffer) > 0:
        idxs = torch.randint(len(replay_buffer), (batch_size,))
        x = replay_buffer[idxs].to(device)
        if torch.rand(1) < reinit_freq:
            x = torch.randn(batch_size, n_ch, im_sz, im_sz).to(device)
    else:
        x = torch.randn(batch_size, n_ch, im_sz, im_sz).to(device)

    # ODL updates
    for _ in range(n_steps):
        x.requires_grad_()
        energy = f(x)
        energy_sum = energy[0] + energy[1]
        grad = torch.autograd.grad(energy, x, create_graph=True)[0]
        x.data.add_(-odl_lr * grad + odl_std * torch.randn_like(x))
    # Set the model back to training mode if necessary
    # model.train()
    # Update replay buffer
    if replay_buffer is not None:
        replay_buffer[idxs] = x.cpu().detach()
    return x


class Energy_InfoNCE(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 buffer_size=10000, sgld_steps=30, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05, if_cond=False,
                 n_classes=7, im_sz=224, n_ch=3, path=None, logger=None):
        super().__init__()
        self.energy_model = EnergyModel(model)
        self.replay_buffer = init_random(buffer_size, im_sz=im_sz, n_ch=n_ch)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond
        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch
        self.path = path
        self.logger = logger
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.energy_model, self.optimizer)

    def adapt(self, x, prototypes):
        if self.episodic:
            # 根据episodic来判定是否重置模型参数
            self.reset()
            # for i in range(self.steps):
            acc = 0

        outputs, loss = forward_and_adapt(x, prototypes, self.energy_model, self.optimizer,
                                          self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std,
                                          self.reinit_freq,
                                          if_cond=self.if_cond, n_classes=self.n_classes)
        # if i % 1 == 0 and if_vis:
        #     visualize_images(path=self.path, replay_buffer_old=self.replay_buffer_old,
        #                      replay_buffer=self.replay_buffer, energy_model=self.energy_model,
        #                      sgld_steps=self.sgld_steps, sgld_lr=self.sgld_lr, sgld_std=self.sgld_std,
        #                      reinit_freq=self.reinit_freq,
        #                      batch_size=100, n_classes=self.n_classes, im_sz=self.im_sz, n_ch=self.n_ch,
        #                      device=x.device, counter=counter, step=i)

        # logging.info(f'Loss Energy: {loss.item()}')

    def forward(self, x, counter=None):
        self.energy_model.eval()
        with torch.no_grad():
            outputs = self.energy_model.classify(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.enable_grad()
def visualize_images(path, replay_buffer_old, replay_buffer, energy_model,
                     sgld_steps, sgld_lr, sgld_std, reinit_freq,
                     batch_size, n_classes, im_sz, n_ch, device=None, counter=None, step=None):
    num_cols = 10
    repeat_times = batch_size // n_classes
    y = torch.arange(n_classes).repeat(repeat_times).to(device)
    x_fake, _ = sample_q(energy_model, replay_buffer, n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std,
                         reinit_freq=reinit_freq, batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)
    images = x_fake.detach().cpu()
    save_image(images, os.path.join(path, 'sample.png'),
               padding=2, nrow=num_cols)
    num_cols = 40
    images_init = replay_buffer_old.cpu()
    images = replay_buffer.cpu()
    images_diff = replay_buffer.cpu() - replay_buffer_old.cpu()
    if step == 0:
        save_image(images_init, os.path.join(
            path, 'buffer_init.png'), padding=2, nrow=num_cols)
    save_image(images, os.path.join(path, 'buffer-' + str(counter) + "-" + str(step) + '.png'), padding=2,
               nrow=num_cols)  #
    save_image(images_diff, os.path.join(
        path, 'buffer_diff.png'), padding=2, nrow=num_cols)


def generate_pseudolabels(model_out, prototypes):
    # model.eval()
    pseudolabels = []
    distances = torch.cdist(model_out, prototypes)
    closest_indices = torch.argmin(distances, dim=1)
    pseudolabels.extend(closest_indices)
    return torch.tensor(pseudolabels).long()


def calculate_change_rate(loss_history):
    if len(loss_history) < 2:
        return 0
    current_loss = loss_history[-1]
    prev_loss = loss_history[-2]
    change_rate = (current_loss - prev_loss) / prev_loss if prev_loss != 0 else 0
    return change_rate

class HSSCLoss(nn.Module):
    def __init__(self, num_classes, tau=1.0, epsilon=0.10):
        super(HSSCLoss, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.epsilon = epsilon
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.threshold1 = 0.9

    def forward(self, features, labels):  # 归一化特征，伪标签
        # 将类别标签转换为独热编码
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        # 计算每个类别的特征中心
        class_centers = []
        for c in range(self.num_classes):
            class_features = features[labels == c]
            if len(class_features) > 0:
                class_center = class_features.mean(dim=0)
            else:
                class_center = torch.zeros(features.size(1), device=features.device)
            class_centers.append(class_center)
        class_centers = torch.stack(class_centers)

        losses = []
        j = 0
        for i in range(features.size(0)):
            gi = features[i]
            wi = one_hot_labels[i]
            true_label = labels[i].item()  # 获取真实类别标签

            # 利用置信度筛选带有可靠伪标签的样本进行损失计算
            if torch.max(gi) > self.threshold1:
                j += 1
                # 计算gi与对应类别中心的余弦相似度
                class_center = class_centers[true_label]
                cos_sim = self.cos_sim(gi, class_center)
                if cos_sim > self.epsilon:
                    exp_giwj_over_tau = torch.exp(cos_sim / self.tau)
                    exp_giwk_over_tau = torch.sum(torch.exp(self.cos_sim(gi.unsqueeze(0), features) / self.tau))
                    loss = -torch.log(exp_giwj_over_tau / exp_giwk_over_tau)
                    losses.append(loss)
                else:
                    losses.append(torch.tensor(0.0, device=features.device, requires_grad=True))
            else:
                losses.append(torch.tensor(0.0, device=features.device, requires_grad=True))

        # print('高于置信度阈值样本量', j)
        return torch.mean(torch.stack(losses))

# class HSSCLoss(nn.Module):
#     def __init__(self, num_classes, tau=1.0, epsilon=0.80):
#         super(HSSCLoss, self).__init__()
#         self.num_classes = num_classes
#         self.tau = tau
#         self.epsilon = epsilon
#         self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
#         self.threshold1 = 0.99
#
#     def forward(self, features, labels):
#         # 将类别标签转换为独热编码
#         one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
#         losses = []
#         j = 0
#         for i in range(features.size(0)):  # [batchsize, classes]
#             gi = features[i]  # 经过softmax之后的probs
#             wi = one_hot_labels[i]
#             # 利用置信度筛选带有可靠伪标签的样本进行损失计算
#             if torch.max(gi) > self.threshold1:
#                 j += 1
#                 # 计算gi与其他同类样本的平均余弦相似度
#                 same_class_features = features[(one_hot_labels == wi).any(dim=1)]
#                 if len(same_class_features) > 0:
#                     cos_sim = self.cos_sim(gi, same_class_features.mean(dim=0))
#                     if cos_sim > self.epsilon:
#                         exp_giwj_over_tau = torch.exp(self.cos_sim(gi, wi) / self.tau)
#                         exp_giwk_over_tau = torch.sum(torch.exp(self.cos_sim(gi.unsqueeze(0), features) / self.tau))
#                         loss = -torch.log(exp_giwj_over_tau / exp_giwk_over_tau)
#                         losses.append(loss)
#                     else:
#                         losses.append(torch.tensor(0.0, device=features.device, requires_grad=True))
#                 else:
#                     losses.append(torch.tensor(0.0, device=features.device, requires_grad=True))
#         # print('高于置信度阈值样本量', j)
#         return torch.mean(torch.stack(losses))


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, prototypes, energy_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq,
                      if_cond=False, n_classes=10):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    batch_size = x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device

    if if_cond == 'uncond':
        x_fake, _ = sample_q(energy_model, replay_buffer,
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq,
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=None)
    elif if_cond == 'cond':
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(energy_model, replay_buffer,
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq,
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)

    # forward
    infoNCE_loss_history = []
    energy_loss_history = []

    # 能量方案损失
    out_real = energy_model(x)
    energy_real = out_real[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()
    # print('energy_real:', energy_real.item())
    loss_energy = (-(energy_real - energy_fake))  # 损失函数
    # 计算熵损失
    # shot
    # out_model = energy_model.classify(x)
    # ent_loss = softmax_entropy(out_model).mean(0)
    # softmax_out = F.softmax(out_model, dim=-1)
    # msoftmax = softmax_out.mean(dim=0)
    # ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
    # py, y_prime = F.softmax(out_model, dim=-1).max(1)
    # flag = py > 0.99
    # clf_loss = F.cross_entropy(out_model[flag], y_prime[flag])
    # loss_shot = ent_loss + 0.1 * clf_loss

    # InfoNCE损失
    out_model = energy_model.classify(x)
    probs = F.softmax(out_model, dim=1)
    # counts = torch.zeros(n_classes)
    hssc_loss = HSSCLoss(num_classes=n_classes, tau=1.0, epsilon=0.1)
    # updated_prototypes = torch.zeros_like(prototypes.to(device))
    pseudolabels = generate_pseudolabels(out_model, prototypes.to(device))
    _, predicted = torch.max(out_model.data, 1)

    # for j in range(len(predicted)):
    #     class_idx = predicted[j].item()
    #     updated_prototypes[class_idx] += out_model[j]
    #     counts[class_idx] += 1
    loss_infoNCE = hssc_loss(probs.to(device), pseudolabels.to(device))
    infoNCE_loss_history.append(loss_infoNCE.item())
    energy_loss_history.append(loss_energy.item())

    energy_change_rate = calculate_change_rate(energy_loss_history)
    infoNCE_change_rate = calculate_change_rate(infoNCE_loss_history)

    # weight_energy = 1 / (1 + abs(energy_change_rate))
    # weight_infoNCE = 1 / (1 + abs(infoNCE_change_rate))
    # adapt
    # loss =  weight_infoNCE * loss_infoNCE + weight_energy * loss_energy

    # 消融实验用于比较动态权重机制作用
    loss = loss_infoNCE + loss_energy

    # print('loss_infonce', loss_infoNCE)
    # print('loss_energy', loss_energy)
    # loss = loss_infoNCE + loss_energy
    # loss = loss_shot
    # loss = loss_infoNCE
    # loss = loss_energy
    # print('loss:', loss)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = energy_model.classify(x)  # 调用能量模型实现分类
    # self.energy_model.eval()
    # outputs = self.energy_model.classify(x)
    # self.energy_model.train()

    return outputs, loss

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """ Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)