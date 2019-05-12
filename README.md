# SemiImplicitDNNs

Pytorch implementation of [IMEXnet - A Forward Stable Deep Neural Network](https://arxiv.org/abs/1903.02639) demonstrated on the synthetic Qtips dataset.

## Use
To run the ResNet baseline `python synth.py --net_type resnet`

To run IMEXnet `python synth.py --net_type resnet`

After running both models, you can compare the results with `gen_val_plots.py` and `gen_loss_plots.py`.
