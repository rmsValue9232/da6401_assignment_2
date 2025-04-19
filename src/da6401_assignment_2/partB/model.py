import torch
import lightning as pl
import torchmetrics.functional as metricfunctions
import torchvision.models as models

class ResNet50FineTune(pl.LightningModule):
    """Use ResNet50 model trained on ImageNet1K_V2 for fine-tuning over iNaturalist12K."""
    def __init__(self, img_size = 224, num_classes: int = 10, lr: float = 0.001):
        super().__init__()
        self.img_size = img_size
        self.lr = lr
        self.num_classes = num_classes
        # initialize the pretrained ResNet50 model
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Input size in the last fully connected layer
        num_filters = backbone.fc.in_features
        # Remove the last fully connected layer
        layers = list(backbone.children())[:-1]
        # Build the feature extractor
        self.backbone = torch.nn.Sequential(*layers)
        # Freeze the feature extractor's probabilistic layers like Dropout and BatchNorm 
        self.backbone.eval()
        # Freeze all layers except the last one from participating in training
        self.backbone.requires_grad_(False)
        
        self.classifier = torch.nn.Linear(num_filters, self.num_classes)
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        with torch.no_grad():
            x = self.backbone(x)
            x = torch.flatten(x, 1)
        
        # Pass the features through the classifier
        x = self.classifier(x)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
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
        