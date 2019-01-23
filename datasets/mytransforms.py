import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class ToFloat:
    def __call__(self, tensor):
        return tensor.float()/255

class WhiteNoise:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        noise = torch.randn_like(x)*self.scale
        # raise Exception(noise.max(), noise.min(), x.max(), x.min())
        x += noise

        return x

class GradientNoise:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        steps = x.shape[1]
        grad_vec = torch.linspace(0, 1, steps).unsqueeze(-1)
        grad = grad_vec.repeat(1, steps)
        grad -= grad.mean()
        x += grad*self.scale

        return x
class ResizeTensor:
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, tensor):
        up_img = F.upsample(tensor.unsqueeze(0), self.size, mode=self.mode)
        return up_img.squeeze(0)

class CenterCropTensor:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, tensor):
        C,H,W = tensor.shape
        mid_h, mid_w = (H//2, W//2)

        out = tensor[:, (mid_h-self.h//2):(mid_h+self.h//2), (mid_w-self.w//2):(mid_w+self.w//2)]

        return out

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)

class Squeeze:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, x):
        return torch.squeeze(x, dim=self.dim)

class Unsqueeze:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, x):
        return torch.unsqueeze(x, dim=self.dim)

class OneHot:

    def __init__(self, C):
        self.C = C

    def __call__(self, labels):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''


        one_hot = torch.LongTensor(self.C, labels.size(1), labels.size(2)).zero_()
        target = one_hot.scatter_(0, labels.data, 1)
        
        target = torch.autograd.Variable(target)
            
        return target
