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
os.makedirs("modelo_capas", exist_ok=True)

# Transformación de imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset completo
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Modelo con capas dinámicas
class CNN(nn.Module):
    def __init__(self, num_conv_layers):
        super(CNN, self).__init__()
        layers = []
        input_channels = 1
        output_channels = 32
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = output_channels
            output_channels *= 2

        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Linear(1, 1)  # Inicialización dummy, se sobrescribirá
        self.fc2 = nn.Linear(128, 10)

        # Se hace dummy forward para calcular el tamaño del plano tras convoluciones
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]
            self.fc1 = nn.Linear(self.flattened_size, 128)  # redefinir con tamaño correcto

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # dinámico según el batch
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Entrenamiento y evaluación
val_accuracies = []

for num_layers in [1,2,3]:
    print(f"\nEntrenando modelo con {num_layers} capa(s) convolucional(es)")
    model = CNN(num_conv_layers=num_layers)
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
    print(f"Accuracy de validación: {accuracy:.2f}%")

    # Guardar modelo
    model_path = f"modelo_capas/modelo_{num_layers}capas.pth"
    torch.save(model.state_dict(), model_path)

# Gráfico
plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3], val_accuracies, marker='o')
plt.title("Precisión vs Número de capas convolucionales")
plt.xlabel("Número de capas convolucionales")
plt.ylabel("Precisión de validación (%)")
plt.xticks([1, 2, 3])
plt.grid(True)
plt.savefig("modelo_capas/resultados_capas.png")
plt.show()
