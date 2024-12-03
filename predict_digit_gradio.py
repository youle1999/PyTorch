import torch
import torchvision.transforms as transforms
from PIL import ImageOps, Image
import numpy as np  # Ensure NumPy is imported
from models.simple_cnn import SimpleCNN
import gradio as gr


# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('./outputs/model.pth', map_location=torch.device('cpu')))
model.eval()

def process_image(image):
    # Convert NumPy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure the image has a white background if it has transparency
    if image.mode == 'RGBA':
        image = Image.alpha_composite(Image.new("RGBA", image.size, "WHITE"), image).convert("RGB")
    elif image.mode != 'RGB':
        image = image.convert("RGB")
    
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)

    # Resize to fit within a 28x28 box while preserving aspect ratio
    image.thumbnail((28, 28), Image.Resampling.LANCZOS)  # Updated constant here

    # Create a blank 28x28 white canvas
    new_image = Image.new("L", (28, 28), 255)  # "L" mode for grayscale, white background

    # Paste the resized image onto the canvas, centering it
    paste_position = ((28 - image.size[0]) // 2, (28 - image.size[1]) // 2)
    new_image.paste(image, paste_position)

    # Invert colors if necessary (MNIST has black digits on a white background)
    new_image = ImageOps.invert(new_image)

    # Transform to Tensor and Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    image_tensor = transform(new_image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_digit_gradio(image):
    try:
        image_tensor = process_image(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return f"The model predicts the digit is: {predicted.item()}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
interface = gr.Interface(fn=predict_digit_gradio, inputs="image", outputs="text")
interface.launch()
