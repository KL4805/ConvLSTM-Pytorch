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

## How to use
The `ConvLSTM` class is implemented as follows: 

`class ConvLSTM(nn.Module):`
`    Parameters: input_dim: Number of channels in input
                 hidden_dim: Number of hidden channels`
