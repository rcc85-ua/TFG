import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Crear carpeta donde guardar modelos
save_dir = "batchsize"
os.makedirs(save_dir, exist_ok=True)

# Definir CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Dividir en entrenamiento y validación
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Lista de batch sizes a probar
batch_sizes = [16, 32, 64, 128, 256]

results = []

# Entrenar con cada batch size
for batch_size in batch_sizes:
    print(f"\nEntrenando con batch size = {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy con batch size {batch_size}: {accuracy:.2f}%")
    results.append((batch_size, accuracy))

    # Guardar modelo
    model_path = os.path.join(save_dir, f"modelo_batch{batch_size}.pth")
    torch.save(model.state_dict(), model_path)

# Graficar resultados
batch_sizes_plot, accuracies_plot = zip(*results)
plt.figure()
plt.plot(batch_sizes_plot, accuracies_plot, marker='o')
plt.title("Accuracy en validación vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("batchsize_results.png")
print("Gráfica guardada como batchsize_results.png")