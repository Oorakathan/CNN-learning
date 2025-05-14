import torch
from torch import nn
from model import MNISTmodel
from training_dataset_maker import get_dataloaders
import matplotlib.pyplot as plt

# Parameters
csv_path = "train_labels.csv"
epochs = 10
batch_size = 80
lr = 0.0001
model_save_path = "trained_mnist_model.pth"

# Load data
train_loader, val_loader = get_dataloaders(csv_path, batch_size=batch_size)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTmodel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []

def train_epoch():
    model.train()
    running_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate():
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            running_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy

# Training Loop
for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss, val_acc = validate()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}")

# Plot Loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.show()
