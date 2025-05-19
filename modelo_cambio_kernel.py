import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Asegurar reproducibilidad
torch.manual_seed(42)

# Crear carpeta para guardar modelos
os.makedirs("modelo_kernel", exist_ok=True)

# Transformación de imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar y dividir el dataset
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Definir la arquitectura de la CNN
class CNN(nn.Module):
    def __init__(self, kernel_size):
        super(CNN, self).__init__()
        padding = kernel_size // 2  # Para mantener las dimensiones
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Calcular el tamaño de la salida después de las capas convolucionales
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Entrenamiento y evaluación
kernel_sizes = [3, 5, 7]
val_accuracies = []

for k in kernel_sizes:
    print(f"\nEntrenando modelo con kernel size {k}x{k}")
    model = CNN(kernel_size=k)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    print(f"Precisión de validación: {accuracy:.2f}%")

    # Guardar modelo
    model_path = f"modelo_kernel/modelo_kernel{k}.pth"
    torch.save(model.state_dict(), model_path)

# Gráfico de resultados
plt.figure(figsize=(8, 5))
plt.plot(kernel_sizes, val_accuracies, marker='o')
plt.title("Precisión vs Tamaño del Kernel")
plt.xlabel("Tamaño del Kernel")
plt.ylabel("Precisión de Validación (%)")
plt.xticks(kernel_sizes)
plt.grid(True)
plt.savefig("modelo_kernel/resultados_kernel.png")
plt.show()