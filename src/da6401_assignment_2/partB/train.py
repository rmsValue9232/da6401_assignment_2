from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from .model import ResNet50FineTune
from ..dataloader import INaturalistDataloader


params = {
    'learning_rate': 1.e-3,
    'batch_size': 32,
    'max_epochs': 50,
}

def train(params: dict = None, name: str = 'ResNet50FineTune') -> None:
    wandb.init(
        project="DA6401_assign_2_part_B",
        name=name,
        config=params,
    )

    dm = INaturalistDataloader(
        batch_size=wandb.config['batch_size'],
        num_workers=2,
        augment_train_data=False, # Since using a pretrained model, no augmentation
    )

    dm .setup(stage = 'fit')

    wandb_logger = WandbLogger(
        save_dir='runs',
        project="DA6401_assign_2_part_B",
        name=name,
    )

    wandb_logger.log_hyperparams(wandb.config)

    model = ResNet50FineTune(
        img_size=224,
        lr=wandb.config['learning_rate']
    )

    wandb_logger.watch(model, log='all', log_graph=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        filename='val_acc_{val_accuracy:.2f}',
        save_top_k=3,
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
    )

    trainer = Trainer(
        max_epochs=wandb.config['max_epochs'],
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)

    return model, dm, trainer

