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
os.makedirs("modelo_filtros", exist_ok=True)

# Transformación de imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Modelo con filtros dinámicos
class CNN(nn.Module):
    def __init__(self, base_filters):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Valores de filtros a probar
filtro_list = [8, 16, 32, 64]
val_accuracies = []

for filtros in filtro_list:
    print(f"\nEntrenando modelo con {filtros} filtros base...")
    model = CNN(base_filters=filtros)
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
    model_path = f"modelo_filtros/modelo_{filtros}filtros.pth"
    torch.save(model.state_dict(), model_path)

# Gráfico de resultados
plt.figure(figsize=(8, 5))
plt.plot(filtro_list, val_accuracies, marker='o')
plt.title("Precisión vs Número de filtros")
plt.xlabel("Número de filtros (1ª capa)")
plt.ylabel("Precisión de validación (%)")
plt.grid(True)
plt.savefig("modelo_filtros/resultados_filtros.png")
plt.show()
