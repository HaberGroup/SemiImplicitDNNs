import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from utils import plottable
from .network_utils import *
import matplotlib.pyplot as plt

class IMEXnet(nn.Module):

    def __init__(self, h, NG, use_gpu):
        super().__init__()

        _, self.depth = NG.shape
        self.NG = NG
        self.h = h
        self.L_mode = None
        self.use_gpu = use_gpu

    def init_weights(self, L_mode='rand'):
        nsteps = self.NG.shape[1]
        self.L_mode = L_mode
        K = []
        L = []
        self.BN = []

        if L_mode=='laplacian':
            lap_stencil = torch.Tensor([[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]])/6
            lap_stencil.unsqueeze_(0).unsqueeze_(0)

        for i in range(nsteps):  
            Ki  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)*1e-3
            # Ki  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)

            if L_mode=='rand':
                Li  = torch.rand(np.asscalar(self.NG[1,i]), 1, 3, 3)*1e0
                # Li  = torch.rand(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)*1e-3
            elif L_mode=='laplacian':
                Li = lap_stencil.repeat(self.NG[1,i], self.NG[0,i], 1, 1)*1e-2
            elif L_mode=='zero':
                Li  = torch.zeros(np.asscalar(self.NG[1,i]), np.asscalar(self.NG[0,i]),3,3)
            else:
                raise NotImplementedError()

            Ki   = projectTorchTensor(Ki)

            bni = nn.BatchNorm2d(self.NG[1,i])

            if self.use_gpu:
                bni = bni.cuda()
            self.BN.append(bni)
            K.append(Ki)
            L.append(Li)
        
        return K, L
    
    def num_params(self, kernel_size=3):
        explicit_params = (np.prod(self.NG, axis = 0)*kernel_size**2).sum()
        
        implicit_params = 0
        layers = self.NG[0,:]
        _, opening_layers = np.unique(layers, return_index=True)
        for i, nchannels in enumerate(layers):
            if i not in opening_layers:
                implicit_params += nchannels*(kernel_size**2)

        return explicit_params, implicit_params

    def forward(self,x,K,L):
    
        nt = len(K)
        
        # time stepping
        for j in range(nt):
            batchnorm = self.BN[j]
            
            # If same width
            if self.NG[0,j] == self.NG[1,j]: 

                # Zero kernel means
                Kj = projectTorchTensor(K[j])
                Lj = projectTorchTensor(L[j])

                # explicit step
                z = conv3x3(x,Kj)
                z = batchnorm(z)
                z  = F.relu(z)        
                z = conv3x3T(z, Kj)

                # q = convDiag(x,Lj)
                # if self.L_mode == 'rand':
                    # q = convDiagT(q,Lj)

                # X + h*K'f(N(KX)) + h*L'LX
                # x  = x + self.h*z + self.h*q
                x  = x + self.h*z# + self.h*q # Diffusion Reaction
                
                # Implicit Step
                if not self.L_mode == 'zero':
                    x = batchnorm(x)
                    x  = diagImpConvFFT(x, Lj, self.h, L_mode=self.L_mode)
    
            # Change number of channels/resolution    
            else:
                z1 = conv3x3(x, K[j])
                # z2 = convDiag(x, L[j])
                # raise Exception(x.shape, L[j].shape)

                z1 = batchnorm(z1)
                x  = F.relu(z1) # Instance norm on z2 applies BCs

        return x 

if __name__ == '__main__':

    import numpy as np
    def plottable(x):
        return np.moveaxis(x.cpu().detach().numpy(), 1, -1).squeeze()

    h = 1e-1
    NG = [ 3,3,3,3,3,3,3,3,3,
           3,3,3,3,3,3,3,3,3]
    # NG = [  3, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    #        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    NG = np.reshape(NG, (2,-1))

    net = IMEXnet(h, NG)
    K,L = net.init_weights(L_mode='laplacian')

    x = torch.zeros(1,3,32,32)
    # x = torch.zeros(1,3,128,128)
    x[:,:,8, 8] = 100
    plt.imshow(plottable(x))
    plt.show()

    X = net(x, K, L)

    raise Exception(net, net.depth)
