
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# variables to plot losses
batch_loss=[]
epoch_losses=[]

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.Encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU()
        )
        self.Decoder = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28))  # (batch size,(1*28*28)-> (channel,hight,width) )
            
        )
        
    def forward(self,x):    
        x = self.Encoder(x)
        x = self.Decoder(x)

        return x # -> this is the reconstructed output after encoding and decodeing. and it has encoded featers too

model = Autoencoder().to(device) #call the model and tell where to store and train the model. in cuda or cpu

criterion = nn.BCELoss() # type of loss function you need
optimizer = torch.optim.Adam(model.parameters(),lr = 0.003) # -> what optimizer you want and learning rate you are going to use

# train loop the model
print("Training loop is started")
epochs = 8
for epoch in range(epochs):  # Just for 5 epochs as an example
    for images, _ in train_loader:
        images = images.to(device)  # sending the image to cpu
        outputs = model(images) 
        loss = criterion(outputs, images) 
        batch_loss.append(loss.item())
        optimizer.zero_grad()  
        loss.backward()  # back propogate based on the losses you obtained
        optimizer.step()  # update the model's parameters calculated from backpropogation. 

    epoch_losses.append(sum(batch_loss[-len(train_loader):]) / len(train_loader))

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")  # pring the process after each epoch is done
    
# to save
torch.save(model.state_dict(),"autoencoder.pth")

plt.plot(batch_loss,color='green')
plt.title("Batch Loss Over Time")
plt.xlabel("Batch #")
plt.ylabel("Loss")
plt.grid()
plt.show()
plt.grid()

plt.plot(epoch_losses,color='red',marker='o',linestyle='dashed')
plt.title("Epoch Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()


# to print the weight and bias layer by layer
'''
for name, param in model.named_parameters():
    print(f"Layer: {name}")
    print(f"Shape: {param.shape}")
    print(f"Values:\n{param.data}\n")
'''



        