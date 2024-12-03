import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN

# Step 1: Load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Load Model
model = SimpleCNN()
model.load_state_dict(torch.load('./outputs/model.pth', map_location=torch.device('cpu')))
model.eval()

# Step 3: Visualize Predictions
dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Plot some images with their predictions
plt.figure(figsize=(12, 4))  # Adjust plot size
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"Pred: {predicted[i].item()}")
    plt.axis('off')

# Save the figure
plt.savefig('./outputs/predictions.png')
plt.show()
