import torch
from torchvision import transforms
from PIL import Image
from model import MNISTmodel

# Load trained model
model = MNISTmodel()
model.load_state_dict(torch.load("trained_mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transform: same as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Shape: (1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, 1).item() # <- read this
    
    return predicted

if __name__ == "__main__":
    img_path = "test_digit.png"
    prediction = predict_image(img_path)
    print(f"Predicted digit: {prediction}")
