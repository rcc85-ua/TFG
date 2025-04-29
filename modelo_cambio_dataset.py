import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
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

# Config
batch_size = 64
epochs = 5
fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
val_accuracies = []

# Transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

random.seed(42)
torch.manual_seed(42)

for frac in fractions:
    size = int(frac * len(full_train_dataset))
    subset_indices = list(range(size))
    random.shuffle(subset_indices)

    train_split = int(0.8 * size)
    train_indices = subset_indices[:train_split]
    val_indices = subset_indices[train_split:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nEntrenando con {len(train_dataset)} muestras ({int(frac*100)}%)")

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        correct_train, total_train, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        correct_val, val_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / len(val_dataset)
        print(f"Época {epoch+1}/{epochs} - Val Accuracy: {val_accuracy:.2f}%")

    val_accuracies.append(val_accuracy)

    # Guardar y exportar modelo ONNX
    torch.save(model.state_dict(), f"modelo_{int(frac*100)}porc.pth")
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, dummy_input, f"modelo_{int(frac*100)}porc.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# 📈 GRAFICAR RESULTADOS
plt.figure(figsize=(8, 6))
plt.plot([int(f*100) for f in fractions], val_accuracies, marker='o', linestyle='-', color='blue')
plt.title("Precisión en Validación vs Tamaño del Dataset")
plt.xlabel("Porcentaje del Dataset de Entrenamiento")
plt.ylabel("Precisión en Validación (%)")
plt.grid(True)
plt.xticks([int(f*100) for f in fractions])
plt.tight_layout()
plt.savefig("grafico_resultados.png")
plt.show()
