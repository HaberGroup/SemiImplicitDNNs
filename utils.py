import numpy as np
import torch
import torch.utils.data as data

# import config

from torchvision import transforms
from datasets.mytransforms import ToLabel, Relabel, OneHot, Squeeze
# from datasets.voc import VOC

import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plottable(x, mode='image'):
    if mode=='image':
        out = np.moveaxis(x.cpu().detach().numpy(), 0, -1).squeeze()
    elif mode =='label':
        out = x.cpu().detach().numpy().squeeze()

    return out

def iou(I1,I2,nc):
    epsilon = 1e-8
    I1 = I1.view(-1)
    I2 = I2.view(-1)
    r = np.zeros(nc)
    for i in range(nc):
        i1 = np.where(I1==i)[0]
        i2 = np.where(I2==i)[0]
        in12 = np.intersect1d(i1, i2)
        un12 = np.union1d(i1,i2)
        r[i] = len(in12)/(len(un12)+epsilon)
    return r

def getAccuracy(S,labels,nc):

    S = S.t().contiguous().view(1,nc,labels.shape[0],labels.shape[1])
    _, predicted = torch.max(S.data, 1)
    total   = labels.shape[0]*labels.shape[1]
    correct = (predicted == labels).sum().item()
    r       = iou(predicted,labels,nc)
    return np.mean(r), r, predicted

def fast_iou(outputs: torch.Tensor, labels: torch.Tensor):
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0

    iou = (intersection + 1e-1) / (union + 1e-1)  # smooth devision to avoid 0/0
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean()  

def network_geometry(blocks, features_in=3):

    # Add opening layer
    blocks = [(1, features_in)] + blocks

    ng = []
    nchannels_old = None
    for block in blocks:
        nlayers, nchannels = block

        # Opening layer
        if nchannels_old is not None:
            ng.append([nchannels_old, nchannels])

        # Layers, remove the '-1' to not include connecting layers
        [ng.append([nchannels, nchannels]) for _ in range(nlayers-1)]
        nchannels_old = nchannels
    NG = np.array(ng).T
    
    return NG

def dataset_stats(dataset, n_classes=22, ex_per_update=100):
    """
    Calculate the mean and standard deviation for each color channel across all input images in a dataset,
    as well as the normalized class occurance for all labels. The mean and standard deviation are often used
    for data normalization before a network, and the class occurances are often used to weight the loss
    function for semantic segmentation applications.

    Input:
        dataset: torch.utils.data.Dataset class that returns and image and its corresponding label.
        n_classes(optional): Number of target classes.
        ex_per_update(optional): Number of examples to process before updating status.

    Returns:
        mean: Per channel mean pixel value.
        std: Per channel pixel value standard deviation.
        weights: Normalized per class label occurance.
    """

    batch_size = 1
    n = len(dataset)
    loader = data.DataLoader(dataset, batch_size=batch_size)
    cc = dataset[0][0].shape[1]

    mean = torch.zeros(cc)
    std = torch.zeros(cc)
    weights = torch.zeros(n_classes)

    one_hot = OneHot(n_classes)

    for i, (image, label) in enumerate(loader):

        # Per image cc mean and std, then take average of batch
        mean += image.view(image.shape[0], cc, -1).mean(-1).mean(0)
        std += image.view(image.shape[0], cc, -1).std(-1).mean(0)

        # One hot encode the label to expand
        label = one_hot(label).view(n_classes, -1)
        n_pixels = label.shape[-1]

        # Sum each channel to get class occurance then scale to image size
        weights += label.sum(-1).float()/n_pixels

        # Print progress
        if i == 0:
            print("Progress:")
        elif i % ex_per_update == 0:
            print('\t%4.1f %%' % (100*i/n))

    weights /= n
    mean /= n
    std /= n

    return mean, std, weights

def dataset_normalization_stats(dataset, ex_per_update=100):
    """
    Calculate the mean and standard deviation for each color channel across all input images in a dataset.
    The mean and standard deviation are often used for data normalization.

    Input:
        dataset: torch.utils.data.Dataset class that returns and image and its corresponding label.
        ex_per_update(optional): Number of examples to process before updating status.

    Returns:
        mean: Per channel mean pixel value.
        std: Per channel pixel value standard deviation.
    """

    batch_size = 1
    n = len(dataset)
    cc = dataset[0][0].shape[0]
    loader = data.DataLoader(dataset, batch_size=batch_size)

    mean = torch.zeros(cc)
    std = torch.zeros(cc)

    for i, (image, _) in enumerate(loader):

        # Per image cc mean and std, then take average of batch
        mean += image.view(image.shape[0], cc, -1).mean(-1).mean(0)
        std += image.view(image.shape[0], cc, -1).std(-1).mean(0)

        # Print progress
        if i == 0:
            print("Progress:")
        elif i % ex_per_update == 0:
            print('\t%4.1f %%' % (100*i/n))

    mean /= n
    std /= n

    return mean, std

def target_normalization_stats(dataset, ex_per_update=100):
    """
    Calculate the mean and standard deviation for each color channel across all target images in a dataset.
    The mean and standard deviation are often used for data normalization in a regression problem.

    Input:
        dataset: torch.utils.data.Dataset class that returns and image and its corresponding label.
        ex_per_update(optional): Number of examples to process before updating status.

    Returns:
        mean: Per channel mean pixel value.
        std: Per channel pixel value standard deviation.
    """

    batch_size = 1
    n = len(dataset)
    cc = dataset[0][1].shape[0]
    loader = data.DataLoader(dataset, batch_size=batch_size)

    mean = torch.zeros(cc)
    std = torch.zeros(cc)

    for i, (_, label) in enumerate(loader):

        
        # Per image cc mean and std, then take average of batch
        mean += label.view(label.shape[0], cc, -1).mean(-1).mean(0)
        std += label.view(label.shape[0], cc, -1).std(-1).mean(0)

        # Print progress
        if i == 0:
            print("Progress:")
        elif i % ex_per_update == 0:
            print('\t%4.1f %%' % (100*i/n))

    mean /= n
    std /= n

    return mean, std

def weighted_sample(loss, label, weights, batch_size, reduction='mean'):
    """
    Returns the loss from a subset of pixels. The subset is determined by created a 
    weighted pixel mask where the probability of a pixel being included is its class's
    weight. 
    
    For example a background pixel, with a class weight of 0.1, has a 10%
    chance of being kept in the subset. Finally, `batch_size` pixels are randomly
    selected from the subset after the weighted pixel mask is applied to `loss`.  

    Input:
        loss (1xHxW): Per-pixel loss tensor
        label (1xHxW): Ground truth label tensor
        weights: Vector of class weights
        batch_size: number of pixels that will contribute to the loss.
        reduction (optional): 
            'mean' Return the mean loss calculated from the subset
            'none': Return the subset 
    Output:
        sub_loss: The subset loss after `reduction` has been applied to the tensor.
    """

    assert label.shape[0]==1, "label must have a batch size of 1"
    assert loss.shape[0]==1, "Loss must have a batch size of 1"
    assert label.shape == loss.shape, "Loss and label must be the same shape"

        
    # Expand label
    n_classes = len(weights)
    label = OneHot(n_classes)(label.cpu())
    label = label.squeeze()
    loss = loss.squeeze()

    # Vectorize
    _, h, w = label.shape
    label = label.view(n_classes, h*w)
    loss = loss.view(h*w)


    # Keep a weighted sample from each class
    sub_inds = torch.tensor([], dtype=torch.long)
    for class_idx, class_mask in enumerate(label):
        
        with torch.no_grad():
            weighted_mask = (torch.rand(class_mask.shape) < weights[class_idx])
            keepers = (weighted_mask & class_mask.byte())
            sub_inds = torch.cat((sub_inds, keepers.nonzero())) 

    # Get a batch from the weighted sample and reduce
    batch_inds = sub_inds[torch.randperm(len(sub_inds))[:batch_size]]
    sub_loss = loss[batch_inds]

    if reduction=='mean':
        sub_loss = sub_loss.mean()
    elif reduction=='none':
        pass
    else:
        raise NotImplementedError()
    
    return sub_loss


if __name__ == '__main__':
    from datasets.mytransforms import ToFloat
    from datasets.nyu_depth import NYUDepthV2

    data_transforms = transforms.Compose([
        ToFloat(),
    ])

    dataset = NYUDepthV2('/scratch/klensink/data/nyu_depth_v2_labeled.mat',
        transform=data_transforms,
        target_transform=data_transforms
    )

    label = target_normalization_stats(dataset)
    img = dataset_normalization_stats(dataset)
    raise Exception(img, label)
    
