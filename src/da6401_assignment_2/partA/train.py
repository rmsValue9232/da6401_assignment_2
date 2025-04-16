from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from . import model
from ..dataloader import INaturalistDataloader

activations = [
    "ReLU",
    "ReLU6",
    "ELU",
    "SELU",
    "CELU",
    "LeakyReLU",
    "PReLU",
    "Softplus",
    "GELU",
    "SiLU",
    "Mish",
    "Hardswish",
    "Hardshrink"
]
params = dict() # Parameters for the model

# General model parameters
params['learning_rate'] = {
        'values': [1.e-6, 1.e-3, 1.e-2, 0.1]
    }
params['batch_size'] = {
        'values': [16, 128]
    }
params['augment_train_data'] = {
    'values': [True, False]
}
params['num_epochs'] = {
        'values': [10]
    }
params['optimizer'] = {
        'values': ['adam', 'nesterov']
    }

# Hidden dense layer parameters
params['hidden_dense_neurons'] = {
        'values': [128, 256, 512]
    }
params['hidden_dense_dropout'] = {
        'values': [0.0, 0.2, 0.3, 0.5]
    }
params['hidden_dense_activation'] = {
        'values': activations
    }
params['hidden_dense_batch_norm'] = {
        'values': [True, False]
    }


# Convolutional block parameters
for i in range(1, 6):
    params[f'num_filters_{i}'] = {
            'values': [8, 64, 128]
        }
    params[f'kernel_size_{i}'] = {
            'values': [3, 5, 7, 9]
        }
    params[f'activation_{i}'] = {
            'values': activations
        }
    params[f'pool_size_{i}'] = {
            'values': [1, 2, 3, 4]
        }
    params[f'pool_stride_{i}'] = {
            'values': [1, 2, 3, 4]
        }
    params[f'dropout_{i}'] = {
            'values': [0.0, 0.2, 0.3, 0.5]
        }
    params[f'batch_norm_{i}'] = {
            'values': [True, False]
        }

# Hyperparameter sweep configuration
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': params
}

def train(params: dict = None) -> None:
    """Train the model using the given current sweep configuration."""
    if params is None:
        # When using this for hyperparameter sweep
        wandb.init()
    else:
        # When using this for standalone training
        wandb.init(config = params)

    data_module = INaturalistDataloader(
        batch_size = wandb.config['batch_size'],
        num_workers = 2,
        augment_train_data= wandb.config['augment_train_data']
    )
    
    data_module.setup(stage = 'fit')

    wandb_logger = WandbLogger(save_dir='runs')
    wandb_logger.log_hyperparams(wandb.config)

    # Convolutional block configurations for blocks 2 to 5
    conv_block_configs = [
        (
            wandb.config[f'dropout_{i}'], # dropout
            wandb.config[f'num_filters_{i-1}'], # in_channels
            wandb.config[f'num_filters_{i}'], # out_channels
            wandb.config[f'kernel_size_{i}'], # kernel_size
            (wandb.config[f'kernel_size_{i}'] - 1)//2, # padding
            wandb.config[f'batch_norm_{i}'], # batch_norm
            wandb.config[f'activation_{i}'], # activation
            wandb.config[f'pool_size_{i}'], # pool_size
            wandb.config[f'pool_stride_{i}'] # pool_stride
        ) for i in range(2, 6)
    ]

    # Define the model
    my_CNN = model.SimpleCNN(
        image_size = 128, # According to the data module definition
        lr = wandb.config['learning_rate'],
        optimizer = wandb.config['optimizer'],

        conv_block_1_config = (
            wandb.config['dropout_1'],
            3,
            wandb.config['num_filters_1'],
            wandb.config['kernel_size_1'],
            (wandb.config['kernel_size_1'] - 1)//2,
            wandb.config['batch_norm_1'],
            wandb.config['activation_1'],
            wandb.config['pool_size_1'],
            wandb.config['pool_stride_1']
            ),
        conv_block_2_config = conv_block_configs[0],
        conv_block_3_config = conv_block_configs[1],
        conv_block_4_config = conv_block_configs[2],
        conv_block_5_config = conv_block_configs[3],

        hidden_dense_neurons = wandb.config['hidden_dense_neurons'],
        hidden_dense_dropout = wandb.config['hidden_dense_dropout'],
        hidden_dense_activation = wandb.config['hidden_dense_activation'],
        hidden_dense_batchNorm = wandb.config['hidden_dense_batch_norm'],
    )

    # Watch the model
    wandb_logger.watch(my_CNN, log_graph=True)

    # Define the callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_accuracy',
        mode = 'max',
        filename = 'val_acc_{val_accuracy:.2f}',
    )

    early_stopping_callback = EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        patience= round(wandb.config['num_epochs'] * 0.25) # 25% of the epochs
    )

    # Define the trainer
    trainer = Trainer(
        max_epochs = wandb.config['num_epochs'],
        logger = wandb_logger,
        callbacks = [checkpoint_callback, early_stopping_callback],
        devices = 2,
    )

    # Train the model
    trainer.fit(model = my_CNN, datamodule = data_module)

    wandb.finish()



def sweeper(sweep_config: dict, max_runs: int = 5) -> None:
    """Initialize the sweep and start the training."""
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep = sweep_config, project = 'DA6401_assign_2_part_A')
    # Start the sweep agent
    wandb.agent(sweep_id, function = train, count = max_runs)

def main():
    wandb.login()
    sweeper(
        sweep_config = sweep_configuration,
        max_runs = 20 # Number of runs for the sweep
    )

