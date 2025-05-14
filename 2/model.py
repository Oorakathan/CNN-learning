import torch
from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
#import matplotlib.pyplot as plt

'''
# Load Data
training_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

# Dataloaders
batch_size = 80
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
'''
# Model 
class MNISTmodel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st conv: 1 ip ch, 32 op ch , 3x3 kernel, stride=1, padding=1 : op dim same
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # 2x2 max pooling, stride=2 -> halves op dimention dimensions: 28x28 -> 14x14
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd conv: 32 ip ch -> 64 op ch : op dim same
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # 2x2 max pooling:half op dimentions
        #14x14 -> 7x7
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattens output from shape (64, 7, 7) to a vector of size 64×7×7 = 3136
        self.flatten = nn.Flatten()

        # Fully connected layer: 3136 -> 128
        self.l1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        # Final output layer: 128 -> 10 (for 10-class classification)
        self.l2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)      # (1, 28, 28)  -> (32, 28, 28)
        x = self.relu1(x)
        x = self.maxpool1(x)   # (32, 28, 28) -> (32, 14, 14)
            
        x = self.conv2(x)      # (32, 14, 14) -> (64, 14, 14)
        x = self.relu2(x)
        x = self.maxpool2(x)   # (64, 14, 14) -> (64, 7, 7)
           
        x = self.flatten(x)    # (64, 7, 7)->(3136)
        x = self.l1(x)         # (3136)->(128)
        x = self.relu3(x)
        x = self.l2(x)         # (128)->(10)
        return x

'''
model = MNISTmodel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Using a more common learning rate

# Lists to store loss values
train_losses = []
test_losses = []

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        if batch % 100 == 0:
            loss_item, current = loss.item(), batch * len(X)
            print(f"Train loss: {loss_item:>7f}  [{current:>5d}/{size:>5d}]")
    train_losses.append(epoch_loss / len(dataloader)) # Average loss for the epoch

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    test_losses.append(test_loss) # Average loss for the epoch
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# training and testing loops and plot
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()
'''