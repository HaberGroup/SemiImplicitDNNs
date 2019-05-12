import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

import datetime
import numpy as np
import argparse
import os
from networks.IMEXnet import IMEXnet
from networks.UNet import UNet
from networks.network_utils import conv1x1
import torchnet.meter as tnt
from utils import bcolors, plottable, network_geometry, dataset_stats, dataset_normalization_stats
import matplotlib.pyplot as plt
import matplotlib
import time
from torchvision import transforms
from datasets.SynthSegDataset import SynthSegDataset, ToLabel

def getAccuracy(preds, labels):
    assert preds.shape == labels.shape, "Preds and Labels must be same shape"
    N = preds.numel()
    acc = (preds==labels).sum().item()/N

    return acc

def gen_color_map(normalized=True, base_map_name='tab10'):
    base_map = matplotlib.cm.get_cmap(base_map_name, 10).colors
    cmap = np.zeros_like(base_map)
    cmap[0,-1] = 1
    cmap[1:-1] = base_map[:8]
    cmap[-1] = [1, 1, 1, 1]

    return matplotlib.colors.ListedColormap(cmap)

def plot_probs_ind(image, label, pred, probs, path):
    fig_name = 'image'
    plt.imshow(plottable(image), vmin=-5, vmax=15, cmap="gray")
    # plt.title('Image')
    plt.axis('off')
    plt.savefig(os.path.join(path, fig_name))

    fig_name = 'label'
    plt.imshow(plottable(label, mode='label'), cmap=cmap, vmin=0, vmax=10)
    plt.axis('off')
    # plt.title('Label')
    plt.savefig(os.path.join(path, fig_name))

    fig_name = 'pred'
    plt.imshow(plottable(pred, mode='label'), cmap=cmap, vmin=0, vmax=10)
    plt.axis('off')
    # plt.title('Prediction')
    plt.savefig(os.path.join(path, fig_name))

    ## PROBS
    fig_name = 'prob_bg'
    plt.imshow(plottable(probs[0], mode='label'), vmin=0, vmax=1)
    plt.axis('off')
    # plt.title('Background')
    plt.savefig(os.path.join(path, fig_name))

    fig_name = 'prob_blue'
    plt.imshow(plottable(probs[1], mode='label'), vmin=0, vmax=1)
    plt.axis('off')
    # plt.title('WW (Blue)')
    plt.savefig(os.path.join(path, fig_name))

    fig_name = 'prob_orange'
    plt.imshow(plottable(probs[2], mode='label'), vmin=0, vmax=1)
    plt.axis('off')
    # plt.title('BW (Orange)')
    plt.savefig(os.path.join(path, fig_name))

    fig_name = 'prob_green'
    plt.imshow(plottable(probs[3], mode='label'), vmin=0, vmax=1)
    plt.axis('off')
    # plt.title('BB (Green)')
    plt.savefig(os.path.join(path, fig_name))

    # plt.show()

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    # raise Exception(iou, thresholded)
    return iou.mean()#, thresholded  # Or thresholded.mean() if you are interested in average across the batch

parser = argparse.ArgumentParser(description='Segmentation of Synthetic Qtip dataset')
parser.add_argument('--net_type', '-n', default='imex', type=str, help='either resnet or imex, default is imex')
args = parser.parse_args()


fig_path = os.path.join('figs/paper/', args.net_type)
ckpt_dir = os.path.join('figs', args.net_type)

net_ckpt = os.path.join(ckpt_dir, 'net.ckpt')
state_ckpt = os.path.join(ckpt_dir, 'state.ckpt')
# loss_hist = np.load(os.path.join(ckpt_dir, 'loss_hist.npz'))
# train_hist = loss_hist['train_hist']
# train_time = loss_hist['train_time']
# val_hist = loss_hist['val_hist']
# val_time = loss_hist['val_time']

# plt.plot(train_time, train_hist)

# plt.figure()
# plt.plot(val_time, val_hist)
# plt.show()
## Load model ckpt
cmap = gen_color_map()
batch_size = 8
use_gpu=True


state_dict = torch.load(state_ckpt)
net = torch.load(net_ckpt)

K = state_dict['K']
L = state_dict['L']
W = state_dict['W']

# Move to gpu
W = W.cuda()
K = [Ki.cuda() for Ki in K]
L = [Li.cuda() for Li in L]

# Val set loader
data_transforms = transforms.Compose([
    transforms.ToTensor()
])
target_transforms = transforms.Compose([
    transforms.ToTensor(),
    ToLabel()
])

train_dataset = SynthSegDataset(
    '/scratch/klensink/data/synthseg/train/',
    transform=data_transforms,
    target_transform=target_transforms
)
val_dataset = SynthSegDataset(
    '/scratch/klensink/data/synthseg/val/',
    transform=data_transforms,
    target_transform=target_transforms
)

# Calc class weights
print('\n Calculating class weights...')
N = len(train_dataset)
_, _, weights = dataset_stats(train_dataset, n_classes=4, ex_per_update=8)
mean, std = dataset_normalization_stats(train_dataset, ex_per_update=8)

print('Class Weights', weights, '\n')

data_transforms = transforms.Compose([
    #transforms.CenterCrop((16, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
target_transforms = transforms.Compose([
    #transforms.CenterCrop((16, 64)),
    transforms.ToTensor(),
    ToLabel()
])

train_dataset = SynthSegDataset(
    '/scratch/klensink/data/synthseg/train/',
    transform=data_transforms,
    target_transform=target_transforms
)
val_dataset = SynthSegDataset(
    '/scratch/klensink/data/synthseg/val/',
    transform=data_transforms,
    target_transform=target_transforms
)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)
nbatches_train = len(train_dataset)/batch_size
nbatches_val = len(val_dataset)/batch_size

## Save plots to dir
misfit = nn.CrossEntropyLoss(weight=(1 - weights).cuda())
softmax = nn.Softmax2d()

################
### VALDIATE ###
################
running_loss = tnt.AverageValueMeter()
running_acc = tnt.AverageValueMeter()
running_iou = tnt.AverageValueMeter()

count=0
for batch_idx, (images, labels) in enumerate(val_loader):

    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()

    # Forward Pass
    with torch.no_grad():
        X = net(images, K, L)
        outputs = conv1x1(X, W)
        probs = softmax(outputs)
        loss = misfit(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = getAccuracy(preds, labels)

        iou = iou_pytorch(preds, labels)

    running_loss.add(loss.item())
    running_acc.add(acc)
    running_iou.add(iou)

    #Save every val image
    for i in range(images.shape[0]):
        if count in [18, 33, 3, 11]:
            example_dir = os.path.join(fig_path, '%04d'%count)
            print('Saving to %s' % example_dir)
            if not os.path.exists(example_dir):
                os.makedirs(example_dir)
            plot_probs_ind(images[i], labels[i], preds[i], probs[i], example_dir)
        count += 1

print('\n    Validation Loss: %6.4f, Acc: %6.4f, IOU: %6.5f' % (running_loss.mean, running_acc.mean*100, running_iou.mean))

