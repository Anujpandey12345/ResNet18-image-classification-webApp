import torch
import torch.nn as nn
import torch.optim as optim   # Optimizers (SGD, Adam)


# Import torchvision for datasets and models
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Import PyTorch Lightning
import pytorch_lightning as pl


"""1. Data Preprocessing"""
# Define transformations applied to images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(  # Normalize using ImageNet stats
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
])

# Load CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root = "./dataT",
    train = True,
    download = True,
    transform = transform,
)

# Load CIFAR-10 validation dataset
val_dataset = torchvision.datasets.CIFAR10(
    root = "./dataT",
    train = False,
    download = True,
    transform = transform,
)


# Create DataLoader for training
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)
# Create DataLoader for validation
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)


"""2. Lightning Model"""

class ResNet18Model(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()


        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)
        # Loss Funstion
        self.lossFun = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Get images and labels from batch
        images, labels = batch
        # Forward pass
        outputs = self(images)
        # Loss calculate
        loss = self.lossFun(outputs, labels)
        # Log training loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.lossFun(outputs, labels)
        # Get predicted class
        preds = torch.argmax(outputs, dim=1)
        # Calculate accuracy
        acc = (preds==labels).float().mean()
        # Log metrics
        self.log("val_loss", loss)
        self.log("loss_acc", acc)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
"""3. Training"""
model = ResNet18Model(num_classes=10)

# Initialize trainer
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1
)
# Start training
trainer.fit(model, train_loader, val_loader)




# Save you model
torch.save(model.model.state_dict(), "resnet18_model.pth")