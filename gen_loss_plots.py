import matplotlib.pyplot as plt
import numpy as np
import os

resnet_fig_path = '/scratch/klensink/figs/paper/resnet/'
resnet_ckpt_dir = '/scratch/klensink/figs/resnet'
resnet_net_ckpt = os.path.join(resnet_ckpt_dir, 'net.ckpt')
resnet_state_ckpt = os.path.join(resnet_ckpt_dir, 'state.ckpt')
resnet_loss_hist = np.load(os.path.join(resnet_ckpt_dir, 'loss_hist.npz'))
resnet_train_hist = resnet_loss_hist['train_hist']
resnet_train_time = resnet_loss_hist['train_time']
resnet_val_hist = resnet_loss_hist['val_hist']
resnet_val_time = resnet_loss_hist['val_time']

imex_fig_path = '/scratch/klensink/figs/paper/imex/'
imex_ckpt_dir = '/scratch/klensink/figs/imex'
imex_net_ckpt = os.path.join(imex_ckpt_dir, 'net.ckpt')
imex_state_ckpt = os.path.join(imex_ckpt_dir, 'state.ckpt')
imex_loss_hist = np.load(os.path.join(imex_ckpt_dir, 'loss_hist.npz'))
imex_train_hist = imex_loss_hist['train_hist']
imex_train_time = imex_loss_hist['train_time']
imex_val_hist = imex_loss_hist['val_hist']
imex_val_time = imex_loss_hist['val_time']


plt.plot(imex_train_time, imex_train_hist)
plt.plot(resnet_train_time, resnet_train_hist)
plt.title('Training Loss')
plt.legend(['IMEX', 'ResNet'])
plt.ylim((0, 1.5))
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.savefig('/scratch/klensink/figs/demo/train_loss.png')

plt.figure()
plt.plot(imex_val_time, imex_val_hist)
plt.plot(resnet_val_time, resnet_val_hist)
plt.title('Validation Loss')
plt.legend(['IMEX', 'ResNet'])
plt.ylim((0, 1.5))
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.savefig('/scratch/klensink/figs/demo/val_loss.png')
# plt.show()