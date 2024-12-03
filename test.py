import torch
import torchvision
import torchvision.transforms as transforms
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
model.load_state_dict(torch.load('./outputs/model.pth'))
model.eval()

# Step 3: Evaluate Model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
