
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# to load the model and use it for testing

# create the class. it should contain same architecture as saved model.
# so copy paste it from where it is trained will be best way\
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.ToTensor()
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)



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

        return x

# initilize the model
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth'))  # load the saved parameters from the saved model
model.eval() # put the model in evaluation mode

# custom function to see visulize the images before and after
def visualize_images(original, reconstructed, num=5):
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    plt.figure(figsize=(10, 4))
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




criterion = nn.BCELoss()

# Inference
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, images)
        print(f"Test Loss: {loss.item():.4f}")

        visualize_images(images, outputs)
        break
