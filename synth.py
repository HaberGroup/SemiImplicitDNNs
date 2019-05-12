import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchnet.meter as tnt
from torchvision import transforms
from tensorboardX import SummaryWriter

import datetime
import numpy as np
import argparse
import os
import time
import matplotlib
import matplotlib.pyplot as plt

from networks.IMEXnet import IMEXnet
from networks.UNet import UNet
from networks.network_utils import conv1x1
from utils import bcolors, plottable, network_geometry, dataset_stats, dataset_normalization_stats
from datasets.mytransforms import WhiteNoise, GradientNoise
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
cmap = gen_color_map()

def receptive_field(NG, kernel_size=3):
    nlayers = NG.shape[1]
    rf = nlayers*(kernel_size + kernel_size//2)
    return rf

def plot_preds(image, label, pred, path):
    plt.subplot(1,3,1)
    plt.imshow(plottable(image), vmin=-5, vmax=15, cmap="gray")
    plt.title('Image')

    plt.subplot(1,3,2)
    plt.imshow(plottable(label, mode='label'), cmap=cmap, vmin=0, vmax=10)
    plt.title('label')

    plt.subplot(1,3,3)
    plt.imshow(plottable(pred, mode='label'), cmap=cmap, vmin=0, vmax=10)
    plt.title('preds')
    plt.show()

    plt.savefig(path)

def plot_probs(image, label, pred, probs, path):
    plt.subplot(2,3,1)
    plt.imshow(plottable(image), vmin=0, vmax=1, cmap="gray")
    # plt.imshow(plottable(image), cmap="gray")
    plt.title('Image')

    plt.subplot(2,3,2)
    plt.imshow(plottable(label, mode='label'), cmap=cmap, vmin=0, vmax=10)
    plt.title('label')

    plt.subplot(2,3,3)
    plt.imshow(plottable(pred, mode='label'), cmap=cmap, vmin=0, vmax=10)
    plt.title('preds')

    ## PROBS
    plt.subplot(2,3,4)
    plt.imshow(plottable(probs[1], mode='label'), cmap='gray', vmin=0, vmax=1)
    plt.title('WW (Blue)')

    plt.subplot(2,3,5)
    plt.imshow(plottable(probs[2], mode='label'), cmap='gray', vmin=0, vmax=1)
    plt.title('BW (Orange)')

    plt.subplot(2,3,6)
    plt.imshow(plottable(probs[3], mode='label'), cmap='gray', vmin=0, vmax=1)
    plt.title('BB (Green)')

    # plt.show()
    plt.savefig(path)


def validate(net, K, L, W, misfit, val_loader, use_gpu, epoch, fig_path, save_fig, nbatches=1, is_unet=False):

    # For now just test on one image from the training set, later loop over val set
    running_loss = tnt.AverageValueMeter()
    running_acc = tnt.AverageValueMeter()

    count=0
    for batch_idx, (images, labels) in enumerate(val_loader):

        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        # Forward Pass
        with torch.no_grad():

            if is_unet:
                outputs = net(images)
            else:
                X = net(images, K, L)
                outputs = conv1x1(X, W)
            probs = softmax(outputs)
            loss = misfit(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = getAccuracy(preds, labels)

        running_loss.add(loss.item())
        running_acc.add(acc)
        summary_writer.add_scalar('Val Loss', running_loss.mean, epoch + (batch_idx/nbatches))
        summary_writer.add_scalar('Val Acc', running_acc.mean, epoch + (batch_idx/nbatches))

        # Save every val image
        if save_fig and (epoch+1)%24==0:
            for i in range(images.shape[0]):
                plot_probs(images[i], labels[i], preds[i], probs[i], os.path.join(fig_path, 'final_preds/%06d_%04d.png' % (epoch, count)))
                count += 1
    
    # Save a single val image
    if save_fig and epoch%1==0:
        plot_probs(images[0], labels[0], preds[0], probs[0], os.path.join(fig_path, 'validating/%06d.png' % epoch))

    print('\n    Validation Loss: %6.4f, Acc: %6.4f' % (running_loss.mean, running_acc.mean*100))

    return running_loss.mean

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation of Synthetic Qtip dataset')
    parser.add_argument('--net_type', '-n', default='imex', type=str, help='either resnet or imex, default is imex')
    args = parser.parse_args()
    # summary_writer = SummaryWriter('/home/klensink/GIT/SemanticSegmentation/log/%s_%s' % (args.net_type, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))) 
    summary_writer = SummaryWriter('log/%s_%s' % (args.net_type, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))) 

    is_unet = True if (args.net_type=='unet') else False
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    # PARAMS, add these to parser eventually
    if is_unet:
        lr = 1e-1
    else:
        lr = 1e-3
    batch_size = 8
    nClasses = 4
    h = 1
    iter_per_update = 4
    num_epochs = 100

    data_dir = 'data/synthseg/train/'
    save_fig=True
    fig_path = os.path.join('figs/', args.net_type) #HARDCODE
    for sub_dir in ['final_preds', 'final_train', 'training', 'validating']:
        dir = os.path.join(fig_path, sub_dir)
        if not os.path.isdir(dir):
            print('Creating output dir %s'%dir)
            os.makedirs(dir)

    #Create net and init weights
    NG = network_geometry([
        (4, 64),
        (4, 128),
        (4, 256),
    ], features_in = 1)

    W = torch.rand(nClasses, NG[-1, -1], 1, 1)*1e-3
    if args.net_type == 'imex':
        net = IMEXnet(h, NG, use_gpu)
        K, L = net.init_weights()
    elif args.net_type == 'resnet':
        net = IMEXnet(h, NG, use_gpu)
        K, L = net.init_weights(L_mode='zero')
    elif args.net_type == 'unet':
        W = None
        K = None
        L = None
        net = UNet(1, nClasses)
    else:
        raise NotImplementedError()

    if True:
        if is_unet:
            n_params = 0
            for p in net.parameters():
                n_params += p.numel()
            print('UNet Params   : %d' % n_params)

        else:
            explicit_params, implicit_params = net.num_params()
            print('IMEX Params   : %d' % (explicit_params + implicit_params))
            print('ResNet Params : %d' % (explicit_params))

    # Print model stats
    rfield = receptive_field(NG)
    # nparams = net.num_params()

    print(bcolors.BOLD + "Explicit Receptive Field: %d" % rfield + bcolors.ENDC)
    # print(bcolors.BOLD + "Model Paramaters: %d" % nparams + bcolors.ENDC)
    print(bcolors.BOLD + "Net Type: %s" % args.net_type + bcolors.ENDC)

    if use_gpu:
        net = net.cuda()

        if (args.net_type == 'resnet') or (args.net_type == 'imex') or (args.net_type == 'imex_concat'):
            W = W.cuda()
            K = [Ki.cuda() for Ki in K]
            L = [Li.cuda() for Li in L]
    
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transforms = transforms.Compose([
        transforms.ToTensor(),
        ToLabel()
    ])

    train_dataset = SynthSegDataset(
        data_dir,
        transform=data_transforms,
        target_transform=target_transforms
    )
    val_dataset = SynthSegDataset(
        data_dir,
        transform=data_transforms,
        target_transform=target_transforms
    )
    
    # Calc class weights
    print('\n Calculating class weights...')
    N = len(train_dataset)
    _, _, weights = dataset_stats(train_dataset, n_classes=4, ex_per_update=10000)
    weights = (1 - weights)
    if use_gpu:
        weights = weights.cuda()
    mean, std = dataset_normalization_stats(train_dataset, ex_per_update=10000)

    print('Class Weights', weights, '\n')

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    target_transforms = transforms.Compose([
        transforms.ToTensor(),
        ToLabel()
    ])

    train_dataset = SynthSegDataset(
        data_dir,
        transform=data_transforms,
        target_transform=target_transforms
    )
    val_dataset = SynthSegDataset(
        data_dir,
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

    # Begin computational graph
    if (args.net_type == 'resnet') or (args.net_type == 'imex') or (args.net_type == 'imex'):
        W = nn.Parameter(W)
        K = [nn.Parameter(Ki) for Ki in K]
        L = [nn.Parameter(Li) for Li in L]

    if (args.net_type=='imex') or (args.net_type=='imex_concat'):
        optimizer = optim.SGD([{'params':K},{'params':L},{'params': W}], lr=lr, momentum=0.0)
    elif args.net_type=='resnet':
        optimizer = optim.SGD([{'params':K},{'params': W}], lr=lr, momentum=0.0)
    elif args.net_type=='unet':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.0)

    misfit = nn.CrossEntropyLoss(weight=weights)
    softmax = nn.Softmax2d()

    print(bcolors.BOLD + 'Batches=%d' %(N//batch_size) + bcolors.ENDC)
    best_val_loss = np.Inf
    hist_val_loss = []
    hist_train_loss = []
    train_time = []
    val_time = []
    for epoch in range(num_epochs):

        print(bcolors.BOLD + '\n=> Training Epoch #%d' %(epoch+1) + bcolors.ENDC)
        running_loss = tnt.AverageValueMeter()
        running_acc = tnt.AverageValueMeter()
        start_time = time.time()
        needs_header = True

        # Training Loop
        count=0
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            # Forward Pass
            optimizer.zero_grad()
            if is_unet:
                outputs = net(images)
            else:
                X = net(images, K, L)
                outputs = conv1x1(X,W) # Classifier

            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)
            
            # Calc loss and backprop
            loss = misfit(outputs, labels)
            acc = getAccuracy(preds, labels)

            loss.backward()
            optimizer.step()

            hist_train_loss.append(loss.item())
            train_time.append(epoch + batch_idx/nbatches_train)
            running_loss.add(loss.item())
            running_acc.add(acc)
            summary_writer.add_scalar('Train Loss', running_loss.mean, epoch + (batch_idx/nbatches_train))
            summary_writer.add_scalar('Train Acc', running_acc.mean, epoch + (batch_idx/nbatches_train))

            if (batch_idx%iter_per_update==0 and not batch_idx==0):
            # if True:
                end_time = time.time()
                if needs_header:
                    update_hdr = ' ' + bcolors.UNDERLINE + '   Iter       Loss       Acc        EPS   ' + bcolors.ENDC
                    print(update_hdr)
                    needs_header = False

                total_time = end_time - start_time
                eps = batch_size*iter_per_update/total_time
                update_str = '   %5d     %6.4f     %6.2f    %5.1f' % (
                    batch_idx, 
                    running_loss.mean, 
                    running_acc.mean*100, 
                    eps)

                running_loss = tnt.AverageValueMeter()
                running_acc = tnt.AverageValueMeter()
                print(update_str)
                start_time = time.time()

            # Save every train image
            #if save_fig and (epoch+1)%100==0:
                #for i in range(images.shape[0]):
                    #plot_probs(images[i], labels[i], preds[i], probs[i], os.path.join(fig_path, 'final_train/%06d_%04d.png' % (epoch, count)))
                    #count += 1

        # # Save a training fig
        # if save_fig and (epoch%1==0):
        #     plt.savefig(os.path.join(fig_path, 'training/%06d.png' % epoch))
        #     plot_probs(images[0], labels[0], preds[0], probs[0], os.path.join(fig_path, 'training/%06d.png' % epoch))

        # Validate
        val_loss = validate(net, K, L, W, misfit, val_loader, use_gpu, epoch, fig_path, save_fig, nbatches=nbatches_val, is_unet=is_unet)
        hist_val_loss.append(val_loss)
        val_time.append(epoch)

        # Save best model
        if True and (val_loss < best_val_loss):
            print(bcolors.OKGREEN + '    Saving best model %6.4f' % (val_loss) + bcolors.ENDC)
            best_val_loss = val_loss

            if (args.net_type == 'imex') or (args.net_type=='resnet'):
                state_dict = {
                    'K':K,
                    'L':L,
                    'W':W
                }
                torch.save(state_dict, os.path.join(fig_path, 'state.ckpt'))
            torch.save(net, os.path.join(fig_path, 'net.ckpt'))

    np.savez(os.path.join(fig_path, 'loss_hist.npz'),
        val_hist=np.array(hist_val_loss), 
        train_hist=np.array(hist_train_loss), 
        train_time=np.array(train_time), 
        val_time=np.array(val_time)
    )
