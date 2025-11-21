import torch
import torch.nn as nn
from dataset import create_dataloaders
from models.resnet import create_resnet18

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device='cpu', save_path='best_model.pth'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.model.to(self.device)
        self.save_path = save_path
        self.best_val_loss = float('inf')

    def train_epoch(self):
        running_loss = 0.0
        self.model.train()
        for Xb, yb in self.train_loader:
            self.optimizer.zero_grad()
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            logits = self.model(Xb)
            loss = self.criterion(logits, yb)
            running_loss += loss.item() * Xb.size(0)
            loss.backward()
            self.optimizer.step()
        running_loss /= len(self.train_loader.dataset)
        return running_loss
    
    def validate(self):
        running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for Xb, yb in self.val_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                logits = self.model(Xb)
                loss = self.criterion(logits, yb)
                running_loss += loss.item() * Xb.size(0)
        running_loss /= len(self.val_loader.dataset)
        return running_loss
    
    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Saved best model with val_loss={val_loss:.4f}")


batch_size = 32
num_workers = 0
lr = 5e-2
epochs = 5
root = 'data'
save_path = 'outputs/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_resnet18(num_classes=10, pretrained=True)
train_loader, val_loader = create_dataloaders(
    batch_size=batch_size, 
    num_workers=num_workers, 
    root=root, 
    download=True
)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
trainer = Trainer(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    optimizer=optimizer, 
    criterion=criterion, 
    device=device, 
    save_path=save_path
)

trainer.fit(epochs)