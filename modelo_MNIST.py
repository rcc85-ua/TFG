import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets
#import torch.utils.data
from sklearn.model_selection import train_test_split
import numpy as np
import onnx





class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1) #capa convolucional
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #capa de pooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128) #capa completamente conectada
        self.fc2 = nn.Linear(128, 10) #capa de salida
        self.p2 = nn.Linear(3,1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) #Conv -> ReLu -> Pool
        x = self.pool(torch.relu(self.conv2(x))) #Conv -> ReLu -> Pool
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
modelo = CNN()
print(modelo)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#SEPARACIÓN DEL DATASET

train_size = int(0.6 * len(train_dataset))
val_size = int(0.2*len(train_dataset))
test_size = len(train_dataset) - train_size - val_size 
train_dataset, val_dataset, _ = random_split(train_dataset, [train_size, val_size, test_size])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 64, shuffle=False)

print("Entrenamiento con " + "{}".format(len(train_dataset)) + " muestras")
print("Validación con " + "{}".format(len(val_dataset)) + " muestras")
print("Prueba con " + "{}".format(len(test_dataset)) + " muestras")

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        optimizer.zero_grad() # Limpia gradientes
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train/total_train




    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct_val / total_val

    print("Época: " + str(epoch+1) + "/" + str(epochs))
    print("Train Loss: " + "{:.4f}".format(train_loss) + " | Train Accuracy: " + "{:.2f}".format(train_accuracy) + "%")
    print("Val Loss: " + "{:.4f}".format(val_loss) + " | Val Accuracy: " + "{:.2f}".format(val_accuracy) + "%")

    print(f'Accuracy: {100*correct_val/total_val:.2f}%')

    torch.save(model.state_dict(), "modelo_entrenado.pth")


# Exportar el modelo entrenado a ONNX
dummy_input = torch.randn(1, 1, 28, 28)  # Entrada simulada del tamaño correcto
torch.onnx.export(model, dummy_input, "modelo.onnx", 
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print("Modelo exportado a ONNX exitosamente: modelo.onnx")
