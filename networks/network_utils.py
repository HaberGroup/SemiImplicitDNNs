import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def diagImpConvFFT(x, K, h, L_mode=None):
    """convolution using FFT of (I + h*K'*K)^{-1}"""
    n = x.shape
    m = K.shape
    mid1 = (m[2] - 1) // 2
    mid2 = (m[3] - 1) // 2
    Kp = torch.zeros(m[0],n[2], n[3],device=K.device)
    # this code also flips up-down and left-right
    Kp[:, 0:mid1 + 1, 0:mid2 + 1] = K[:, 0, mid1:, mid2:]
    Kp[:, -mid1:, 0:mid2 + 1] = K[:, 0, 0:mid1, -(mid2 + 1):]
    Kp[:, 0:mid1 + 1, -mid2:] = K[:, 0, -(mid1 + 1):, 0:mid2]
    Kp[:, -mid1:, -mid2:] = K[:, 0, 0:mid1, 0:mid2]

    xh = torch.rfft(x, 2, onesided=False)
    Kh = torch.rfft(Kp, 2, onesided=False)
    xKh = torch.zeros(n[0],n[1],n[2], n[3], 2,device=K.device)

    # dont need semi pos def if L is laplacian
    if L_mode == 'laplacian':
        t = 1.0/(h * torch.abs(Kh[:, :, :, 0]) + 1.0) 
    else:
        t = 1.0/(h * (Kh[:, :, :, 0] ** 2 + Kh[:, :, :, 1] ** 2) + 1.0)

    for i in range(n[0]):
        xKh[i, :,:, :, 0] = xh[i, :, :, :, 0]*t
        xKh[i, :,:, :, 1] = xh[i, :, :, :, 1]*t
    xK = torch.irfft(xKh, 2, onesided=False)
    return xK

def conv1x1(x,K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=0)

def conv1x1T(x,K):
    """3x3 convolution transpose with padding"""
    #K = torch.transpose(K,0,1)
    return F.conv_transpose2d(x, K, stride=1, padding=0)

def conv3x3(x,K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=1)

def convSame(x,K):
    """Does padding for the same size output"""
    n = K.shape
    p = int(n[2]/2)    
    return F.conv2d(x, K, stride=1, padding=p)

def convSameT(x,K):
    """Does padding for the same size output"""
    n = K.shape
    p = int(n[2]/2)
    
    return F.conv_transpose2d(x, K, stride=1, padding=p)

def conv3x3T(x,K):
    """3x3 convolution transpose with padding"""
    #K = torch.transpose(K,0,1)
    return F.conv_transpose2d(x, K, stride=1, padding=1)

def projectTorchTensor(K): 
    n = K.data.shape
    K.data  = K.data.view(-1,9)
    M       = K.data.mean(1)
    for i in range(9):
        K.data[:,i] -= M
        
    K.data  = K.data.view(n[0],n[1],n[2],n[3])
    return K

def convDiag(x,K):
    n = K.shape
    return F.conv2d(x, K, stride=1, padding=1, groups=n[0])

def convDiagT(x,K):
    n = K.shape
    return F.conv_transpose2d(x, K, stride=1, padding=1, groups=n[0])

def smooth(x):
    nchan = x.shape[1]
    Ki    = torch.zeros((1,1,3,3)) 
    A = 1/16.0*np.array([[1, 2, 1],[2, 4, 2],[1,2,1]])
    Ki[0,0,:,:] = torch.tensor(A)
    K = torch.zeros((nchan,1,3,3))
    for i in range(nchan):
        K[i,0,:,:] = Ki
    
    z = convDiag(x,K)
    zout = x
    zout[:,:,1:-1,1:-1] = z[:,:,1:-1,1:-1]
    return zout

# dis = nn.CrossEntropyLoss()
def misfit(X,W,C, dis = nn.CrossEntropyLoss()):
    S = conv1x1(X,W)        
    S = S.view(S.shape[1],S.shape[2]*S.shape[3]).t()
    return dis(S,C), S   
# def misfitSamp(X,W,C,sampoints):
#     S = conv1x1(X,W)
#     S = S.view(S.shape[1],S.shape[2]*S.shape[3]).t()
    
#     Ssamp = S[sampoints]
#     Csamp = C.view(-1)[sampoints]
#     return dis(Ssamp,Csamp), S   
