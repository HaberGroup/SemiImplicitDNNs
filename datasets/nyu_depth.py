import numpy as np
import h5py
import torch.utils.data as data
import torch

class NYUDepthV2(data.Dataset):
    """NYU DepthV2 RGB-D dataset as descirbed at https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

    Load RBG and Depth images from the HDF5 (.mat) data structure.

    Input:
        matfile: Path to the .mat file.
        transform (optional): Data transforms to be applied to the images.
        target_tranforms (optional): Data transforms to be applied to the depth maps.
        transpose (optional): Transpose the image and depth map 

    Output:
        image: CxHxW RGB image
        target: HxW Float64 depth map
    """
    def __init__(self, matfile, transform=None, target_transform=None, transpose=True):

        # Load HDF5 datasets
        self.f = h5py.File(matfile)
        self.depths = self.f['depths'] #TODO: Check and optimize chunk sizes
        self.images = self.f['images']
        self.transform = transform
        self.target_transform = target_transform
        self.transpose = transpose

    def __getitem__(self, index):
        target = self.depths[index]
        img = self.images[index]

        #TODO: Do this once then re-write to disk
        if self.transpose: 
            target = target.T
            img = np.moveaxis(img, 2, 1)
        
        img = torch.tensor(img)
        target = torch.tensor(target).unsqueeze(0)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    dataset = NYUDepthV2('data/nyu_depth_v2_labeled.mat')

    import time
    start_time = time.time()
    inds = np.random.permutation(len(dataset))
    for i in inds[:100]:
        img, label = dataset[i]
    eps = 100/(time.time() - start_time)
    print('EPS: %f' % (eps))
