# Build a small CNN model consisting of 555 convolution layers. Each convolution layer would be followed by an activation and a max-pooling layer.

# After 555 such conv-activation-maxpool blocks, you should have one dense layer followed by the output layer containing 101010 neurons (111 for each of the 101010 classes). The input layer should be compatible with the images in the iNaturalist dataset dataset.

# The code should be flexible such that the number of filters, size of filters, and activation function of the convolution layers and dense layers can be changed. You should also be able to change the number of neurons in the dense layer.

#     What is the total number of computations done by your network? (assume mmm filters in each layer of size k×kk\times kk×k and nnn neurons in the dense layer)
#     What is the total number of parameters in your network? (assume mmm filters in each layer of size k×kk\times kk×k and nnn neurons in the dense layer)

import torch
import lightning as pl
import torchmetrics.functional as metricfunctions

pytorch_activations = {
    "ReLU": torch.nn.ReLU(),
    "ReLU6": torch.nn.ReLU6(),
    "ELU": torch.nn.ELU(),
    "SELU": torch.nn.SELU(),
    "CELU": torch.nn.CELU(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "PReLU": torch.nn.PReLU(),
    "Tanh": torch.nn.Tanh(),
    "Sigmoid": torch.nn.Sigmoid(),
    "LogSigmoid": torch.nn.LogSigmoid(),
    "Hardtanh": torch.nn.Hardtanh(),
    "Tanhshrink": torch.nn.Tanhshrink(),
    "Softplus": torch.nn.Softplus(),
    "Softsign": torch.nn.Softsign(),
    "Softmin": torch.nn.Softmin(),
    "Softmax": torch.nn.Softmax(dim=1),
    "LogSoftmax": torch.nn.LogSoftmax(dim=1),
    "GELU": torch.nn.GELU(),
    "SiLU": torch.nn.SiLU(),
    "Mish": torch.nn.Mish(),
    "Hardswish": torch.nn.Hardswish(),
    "Hardshrink": torch.nn.Hardshrink(),
    "Threshold": torch.nn.Threshold(threshold=0, value=0),
}

# TODO: Add dropout layers to the conv blocks

# Adding dropout on conv block neurons makes
# those neurons' receptive field in the image
# not participate in the training process
# So this could be useful when the subject
# in the image is pretty localized in the image
# the receptive field of the dropped neurons
# may be those that do not capture the subject
def conv_block(
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        padding: int,
        batchNorm: bool,
        activation: str,
        pool_kernel_size: int
) -> torch.nn.Sequential:
    """Build a customisable conv-activation-maxpool layered block"""

    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=conv_kernel_size,
                        padding=padding)
    )
    if batchNorm:
        block.add_module(torch.nn.BatchNorm2d(out_channels))
    
    block.add_module(pytorch_activations[activation])
    block.add_module(torch.nn.MaxPool2d(kernel_size=pool_kernel_size))
    
    return block

class SimpleCNN(pl.LightningModule):
    def __init__(self,
                 lr = 1e-3,
                 conv_block_1_config: tuple[int, int, int, int, bool, str, int] = ( 3,  8, 3, True, "ReLU", 2),
                 conv_block_2_config: tuple[int, int, int, int, bool, str, int] = ( 8, 16, 3, True, "ReLU", 2),
                 conv_block_3_config: tuple[int, int, int, int, bool, str, int] = (16, 32, 3, True, "ReLU", 2),
                 conv_block_4_config: tuple[int, int, int, int, bool, str, int] = (32, 64, 3, True, "ReLU", 2),
                 conv_block_5_config: tuple[int, int, int, int, bool, str, int] = (64, 128,3, True, "ReLU", 2),
                 hidden_dense_activation: str = "ReLU",
                 hidden_dense_neurons: int = 512,
                 output_activation: str = "Softmax"
                 ):
        super(SimpleCNN).__init__()
