from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image_path):
    """
    Preprocess a single image for testing. Resizes it to 28x28 and normalizes it.
    
    Args:
        image_path (str): Path to the image.
    
    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    image = Image.open(image_path).convert('L')  # Convert image to grayscale (if not already)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),        # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize for MNIST (mean=0.5, std=0.5)
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)

    return image

# Example usage:
image_tensor = preprocess_image("test_image.jpg")
