'''
# input output difference seeing
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Simple Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),  # Output between 0 and 1
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().to(device)
criterion = nn.BCELoss()  # Binary cross-entropy for images in [0,1]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
num_epochs = 5
for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Visualize: Original vs Reconstructed
def show_images(original, reconstructed, num=5):
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(num):
        # Original
        plt.subplot(2, num, i + 1)
        plt.imshow(original[i][0], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, num, i + 1 + num)
        plt.imshow(reconstructed[i][0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Take a batch and visualize
test_images, _ = next(iter(train_loader))
test_images = test_images.to(device)
reconstructed = model(test_images)
show_images(test_images, reconstructed)

'''


'''
# feature map seeing
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Modified Autoencoder for visualization
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),  # Output between 0 and 1
            nn.Unflatten(1, (1, 28, 28))  # Reshape back to 28x28
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        # For feature map visualization, we return the encoder output here
        encoded = x
        # Decoder
        x = self.decoder(x)
        return x, encoded  # Return both the output and feature map

model = Autoencoder().to(device)
criterion = nn.BCELoss()  # Binary cross-entropy for images in [0,1]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)

        # Forward
        outputs, encoded = model(images)
        loss = criterion(outputs, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Visualize: Feature map from the encoder (Before reconstruction)
def show_feature_map(encoded, num=5):
    encoded = encoded.cpu().detach().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(num):
        # Visualize the feature map as a 2D grid
        plt.subplot(1, num, i + 1)
        plt.imshow(encoded[i].reshape(4, 8), cmap='viridis')  # Reshape to a grid for better visualization
        plt.title("Feature Map")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Take a batch and visualize feature maps
test_images, _ = next(iter(train_loader))
test_images = test_images.to(device)
_, encoded = model(test_images)
show_feature_map(encoded)
'''




'''
What you can do with the model other than training and testing and getting the O/P
model parameters
model outputs
feature maps
loss
latent space



'''

'''
#training,saving, reusing, testing and visulizing the model and I/O images
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.module):
    def __init__(nn.Module):
        super(Autoencoder,self).__init__()
        #define your encoder and decoder layers
        
    def forward(self,x):
        #Encoder and Decoder layers. That is call the in the order you want
        return x # -> this is the reconstructed output after encoding and decodeing. and it has encoded featers too

model = Autoencoder().to(device) #call the model and tell where to store and train the model. in cuda or cpu

criterion = nn.BCELoss() # type of loss function you need
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3) # -> what optimizer you want and learning rate you are going to use

# train loop the model
for epoch in range(5):  # Just for 5 epochs as an example
    for images, _ in train_loader:  # iterating over data in train_loader which has batch of images in it.. images means current batch
        images = images.to(device)  # sending the image to cpu
        outputs = model(images) # feeding the data to the model for forward propagation. the result is stored in output variable
        loss = criterion(outputs, images) # checking the loss by output by model vs the input. because we are reconstructing the data again. so we should know how much reconstruction process is lost 
        print(f"Epoch [{epoch+1}/5], Batch Loss: {loss.item():.4f}")  # printing loss after each batch is over
        # Backpropagation is done for each batch and done for 5 times for all the batch. 
        optimizer.zero_grad()  # zero outing the gradient from previous batch training. for 1st loop it is a random by default. so 
        loss.backward()  # back propogate based on the losses you obtained
        optimizer.step()  # update the model's parameters calculated from backpropogation. 

    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")  # pring the process after each epoch is done
    
# to save
torch.save(model.state_dict(),"autoencoder.pth")



# to load the model and use it for testing

# create the class. it should contain same architecture as saved model.
# so copy paste it from where it is trained will be best way
class Autoencoder(nn.module):
    def __init__(nn.Module):
        super(Autoencoder,self).__init__()
        #define your encoder and decoder layers
        
    def forward(self,x):
        #Encoder and Decoder layers. That is call the in the order you want
        return x # -> this is the reconstructed output after encoding and decodeing. and it has encoded featers too

# initilize the model
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth'))  # load the saved parameters from the saved model

# custom function to see visulize the images before and after
def visulize_images(orginal,reconstructed,num=5):
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    plt.figure(figsize = (10,4))
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(original[i][0], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, num, i + 1 + num)
        plt.imshow(reconstructed[i][0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()



model.eval()  # put the model in evaluation mode

with torch.no_grad():    # tell the model not to use any gradient changing
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        visulize_images(images,outputs)
        
'''

