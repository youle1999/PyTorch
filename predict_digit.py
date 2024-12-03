import torch
import torchvision.transforms as transforms
from PIL import Image
from models.simple_cnn import SimpleCNN

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('./outputs/model.pth', map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

def process_image(image_path):
    # Transformations to match MNIST format
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_digit(image_path):
    # Process the image
    image = process_image(image_path)

    # Predict with the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")  # User input for image path
    prediction = predict_digit(image_path)
    print(f"The model predicts the digit is: {prediction}")
