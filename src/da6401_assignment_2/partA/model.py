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

# Adding dropout on conv block neurons makes
# those neurons' receptive field in the image
# not participate in the training process
# So this could be useful when the subject
# in the image is pretty localized in the image
# the receptive field of the dropped neurons
# may be those that do not capture the subject
def conv_block(
        dropout_rate: float,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        padding: int,
        batchNorm: bool,
        activation: str,
        pool_kernel_size: int,
        pool_stride: int
) -> torch.nn.Sequential:
    """Build a customisable conv-activation-maxpool layered block"""

    block = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_rate),
        torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=conv_kernel_size,
                        padding=padding)
    )
    if batchNorm:
        block.append(torch.nn.BatchNorm2d(out_channels))
    
    block.append(pytorch_activations[activation])
    block.append(torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
    
    return block

# Find image size after being passed through the conv blocks:
def get_image_size_after_conv_block(
        image_size: int,
        conv_kernel_size: int,
        padding: int,
        pool_kernel_size: int,
        pool_stride: int
) -> int:
    """Calculate the image size after being passed through a conv block"""
    # Formula for conv layer output size:
    # (input_size - kernel_size + 2 * padding) / stride + 1
    # Formula for maxpool layer output size:
    # (input_size - kernel_size) / pool_stride + 1

    # Assuming stride = 1 for conv layers
    conv_output_size = (image_size - conv_kernel_size + 2 * padding) + 1
    pool_output_size = (conv_output_size - pool_kernel_size)/pool_stride + 1

    return pool_output_size

def get_image_after_all_conv_blocks(
        image_size: int,
        conv_block_configs: list[tuple[float, int, int, int, int, bool, str, int, int]]
) -> int:
    """Calculate the image size after being passed through all conv blocks"""

    for config in conv_block_configs:
        image_size = get_image_size_after_conv_block(
            image_size,
            config[3],
            config[4],
            config[7],
            config[8]
        )
    return image_size


class SimpleCNN(pl.LightningModule):
    def __init__(self,
                 image_size: int,
                 lr = 1e-3,
                 optimizer: str = "adam",

                 conv_block_1_config: tuple[float, int, int, int, int, bool, str, int, int] = (0.2, 3,  8, 3, 1, True, "ReLU", 2, 2),
                 conv_block_2_config: tuple[float, int, int, int, int, bool, str, int, int] = (0.2, 8, 16, 3, 1, True, "ReLU", 2, 2),
                 conv_block_3_config: tuple[float, int, int, int, int, bool, str, int, int] = (0.2,16, 32, 3, 1, True, "ReLU", 2, 2),
                 conv_block_4_config: tuple[float, int, int, int, int, bool, str, int, int] = (0.2,32, 64, 3, 1, True, "ReLU", 2, 2),
                 conv_block_5_config: tuple[float, int, int, int, int, bool, str, int, int] = (0.2,64,128, 3, 1, True, "ReLU", 2, 2),

                 hidden_dense_neurons: int = 512,
                 hidden_dense_dropout: float = 0.5,
                 hidden_dense_activation: str = "ReLU",
                 hidden_dense_batchNorm: bool = True,
                 output_activation: str = "Softmax"
                 ):
        super(SimpleCNN, self).__init__()
        self.image_size = image_size
        self.lr = lr
        self.optimizer = optimizer

        # Calculate the output size after the conv blocks
        self.conv_output_size = get_image_after_all_conv_blocks(
            image_size = image_size,
            conv_block_configs = [conv_block_1_config,
                                    conv_block_2_config,
                                    conv_block_3_config,
                                    conv_block_4_config,
                                    conv_block_5_config]
        )
        self.conv_output_size = int(self.conv_output_size)

        # Build the conv blocks
        self.conv_blocks = torch.nn.Sequential(
            conv_block(*conv_block_1_config),
            conv_block(*conv_block_2_config),
            conv_block(*conv_block_3_config),
            conv_block(*conv_block_4_config),
            conv_block(*conv_block_5_config)
        )

        # Build the flatten layer
        self.flatter = torch.nn.Flatten()

        # Build the dense layer
        self.hidden_dense_neurons = hidden_dense_neurons
        self.hidden_dense_dropout = hidden_dense_dropout
        self.hidden_dense_activation = hidden_dense_activation
        self.hidden_dense_batchNorm = hidden_dense_batchNorm
        self.hidden_dense = torch.nn.Sequential(
            torch.nn.Dropout(p=self.hidden_dense_dropout),
            torch.nn.Linear((self.conv_output_size**2) * conv_block_5_config[2], self.hidden_dense_neurons),
            pytorch_activations[self.hidden_dense_activation]
        )
        if self.hidden_dense_batchNorm:
            self.hidden_dense.append(torch.nn.BatchNorm1d(self.hidden_dense_neurons))
        
        # Build the output layer
        self.output_activation = output_activation
        self.output_layer = torch.nn.Linear(self.hidden_dense_neurons, 10)

        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input through the conv blocks
        x = self.conv_blocks(x)

        # Flatten the input
        x = self.flatter(x)

        # Pass the input through the dense layers
        x = self.hidden_dense(x)
        x = self.output_layer(x)

        # Returns the logits as pytorch cross entropy itself uses logits
        # and not the probabilities
        return x
    
    def _get_loss_and_metrics(self, batch: torch.Tensor):
        X, Y = batch

        # Convert one hot encoded vector to integer label for each sample in the batch
        # Shape becomes from (batch_size, num_classes) to (batch_size,)
        targets = torch.argmax(Y, dim = 1)
        
        # Find the network's predictions
        logits = self(X)
        
        # Probability vectors for record keeping
        Y_hat = torch.nn.functional.softmax(logits, dim=1)
        
        # Find the predicted classes
        # Shape becomes from (batch_size, num_classes) to (batch_size,)
        preds = torch.argmax(Y_hat, dim=1)

        # Calculate loss
        loss = torch.nn.functional.cross_entropy(logits, targets)

        # Calculate metrics
        acc = metricfunctions.accuracy(preds, targets, task='multiclass', num_classes=10)
        precision = metricfunctions.precision(preds, targets, task='multiclass', num_classes=10, average='macro')
        recall = metricfunctions.recall(preds, targets, task='multiclass', num_classes=10, average='macro')
        f1 = metricfunctions.f1_score(preds, targets, task='multiclass', num_classes=10, average='macro')

        return Y_hat, loss, acc, precision, recall, f1
    
    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "nesterov":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, loss, acc, precision, recall, f1 = self._get_loss_and_metrics(batch)
        batch_size = len(batch[0])

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train_precision", precision, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train_recall", recall, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("train_f1score", f1, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, loss, acc, precision, recall, f1 = self._get_loss_and_metrics(batch)
        batch_size = len(batch[0])

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val_precision", precision, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val_recall", recall, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("val_f1score", f1, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, loss, acc, precision, recall, f1 = self._get_loss_and_metrics(batch)
        batch_size = len(batch[0])

        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("test_accuracy", acc, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("test_precision", precision, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("test_recall", recall, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log("test_f1score", f1, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    