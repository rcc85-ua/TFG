import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import functional as TF
from muscima.io import parse_cropobject_list
from collections import Counter

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas y carga de datos
print("Cargando documentos MUSCIMA...")
start = time.time()
CROPOBJECT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop/TFG-Info/ImageFolder/MUSCIMA-pp_v1.0/v1.0/data/cropobjects_manual')
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]
print(f"Documentos cargados en {time.time() - start:.2f} segundos.")

# Extraer clases
class_to_objects = {}
for doc in docs:
    for c in doc:
        class_to_objects.setdefault(c.clsname, []).append(c)

# Filtrar clases con suficientes muestras
min_samples = 200  # Aumentamos el número mínimo de muestras por clase
filtered_classes = {k: v for k, v in class_to_objects.items() if len(v) >= min_samples}
label_map = {clsname: i for i, clsname in enumerate(filtered_classes.keys())}
print("\nClases usadas:", label_map)

# Función para convertir CropObject a imagen (con augmentación opcional)
def get_image_single(cropobj, margin=1, augment=False):
    h, w = cropobj.height + 2 * margin, cropobj.width + 2 * margin
    canvas = np.zeros((h, w), dtype='uint8')
    canvas[margin:margin + cropobj.height, margin:margin + cropobj.width] = cropobj.mask
    canvas[canvas > 0] = 255
    img = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0)
    img = TF.resize(img, [80, 20])  # Tamaño aumentado

    if augment:
        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        if torch.rand(1).item() > 0.5:
            img = TF.vflip(img)
        angle = torch.randint(-10, 10, (1,)).item()
        img = TF.rotate(img, angle)
        img = TF.affine(img, angle=0, translate=(2, 2), scale=1.0, shear=(2, 2))

    img = (img > 127).float()
    return img.squeeze(0).numpy()

# Generar dataset con augmentación parcial
print("Generando datos...")
X, y = [], []
for clsname, objects in filtered_classes.items():
    for i, c in enumerate(objects[:min_samples]):
        img = get_image_single(c, augment=(i < min_samples * 0.5))
        X.append(img)
        y.append(label_map[clsname])

X = np.array(X).reshape(-1, 1, 80, 20).astype(np.float32)
y = np.array(y).astype(np.int64)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Dataset y dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Ponderación de clases para la pérdida
label_counts = Counter(y)
weights = [1.0 / label_counts[i] for i in range(len(label_map))]
weights = torch.tensor(weights).to(device)

# Modelo CNN mejorado (sin Transformer)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 40x10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x2
            nn.Dropout(0.3)
        )
        self.flatten_dim = 128 * 10 * 2
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Aplana la salida
        return self.classifier(x)

# Entrenamiento
model = SimpleCNN(input_channels=1, num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

patience = 10
best_loss = float('inf')
epochs_without_improvement = 0
num_epochs = 100

print("Entrenando el modelo...")
start = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Deteniendo por early stopping.")
            break

print(f"Entrenamiento terminado en {time.time() - start:.2f} segundos.")

# Evaluación
model.eval()
correct = 0
total = 0
class_correct = [0] * len(label_map)
class_total = [0] * len(label_map)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

print(f"\nPrecisión global: {100 * correct / total:.2f}%")
for i, classname in enumerate(label_map.keys()):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"Clase '{classname}': Precisión: {acc:.2f}%")
