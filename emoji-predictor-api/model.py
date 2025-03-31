import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

class EmojiClassifier(pl.LightningModule):
    def __init__(self, num_classes=18, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv = nn.Sequential(
            # 3 channels (RGB)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Assuming input images are eventually resized to 128x128 for example:
        # 128 -> MaxPool -> 64 -> MaxPool -> 32 -> MaxPool -> 16 -> MaxPool -> 8
        # Output size = 256 * 8 * 8
        self.flattened_size = 256 * 8 * 8  # Adjust if your input processing differs

        self.fully_connected = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes)  # output layer for 18 classes
        )

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = nn.Flatten()(x)  # Flatten the output of conv layers
        x = self.fully_connected(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy(out, y), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy(out, y), on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_accuracy(out, y), on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

# Helper function to load the model (can be called from main.py)
def load_emoji_model(checkpoint_path="emoji_model.ckpt", num_classes=18):
    """Loads the EmojiClassifier model from a checkpoint."""
    # Ensure the model class is defined before loading
    model = EmojiClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    model.eval()  # Set the model to evaluation mode
    return model 