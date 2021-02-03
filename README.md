# ConvLSTM_Pytorch
This repo contains a Pytorch implementation of ConvLSTM (Shi et al. 2015). 

Acknowledgement: This file is modified upon the implementation of [ndrplz](https://github.com/ndrplz/ConvLSTM_pytorch). Because that implementation was slightly different from the one in the [paper](https://arxiv.org/pdf/1506.04214.pdf), we modified it to make the implementation in full accordance with the . 

\[Shi et al. 2015\] Shi, X. et al. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. In NIPS, 2015. 

## How to install
This repo is implemented and designed according to Pytorch, and thus, it can be used just as an original Pytorch module. 

You will simply need to: 
- install Pytorch (My version is 1.6.0)
- put the `convlstm.py` under your project folder
- call `import convlstm` or something like `from convlstm import ConvLSTM`

## How to initialize
The `ConvLSTM` class should be initialized as follows: 

`clstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first = False, bias = True, return_all_layers = False):`

Parameters: 
- input_dim: the number of channels in the input
- hidden_dim: the number of channels in the hidden features
- kernel_size: size of kernel in convolutions. We recommend square kernels with sizes being odd numbers
- num_layers: the number of ConvLSTM layers stacked upon each other
- batch_first: whether or not the input is in batch first mode. Detailed later. 
- bias: whether to use bias for convolution
- return_all_layers: whether to return computations from all layers or not. Detailed later. 

## How to call
You simply need to invoke `out = clstm(X)` where `X` is some tensor with valid shape.  

Input: 
- X: a tensor of shape \[B, T, C, H, W\] if `batch_first = True`. Otherwise, \[T, B, C, H, W\].
  - B: batch_size
  - T: length of the sequence
  - C: channels
  - H, W: height and width
  
## Return values
The return values will be a tuple of two elements `out = [h, c]`, where: 
- h: a list of length `num_layers` if `return_all_layers = True`, else a list of length 1. 
