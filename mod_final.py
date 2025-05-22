import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import functional as TF
from torchvision import transforms
from muscima.io import parse_cropobject_list
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from datetime import datetime

# Configuración
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_samples = 200
        self.image_size = (80, 20)
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.patience = 15
        self.augment_ratio = 0.6
        self.dropout_rate = 0.4
        self.weight_decay = 1e-4

config = Config()
print(f"Usando dispositivo: {config.device}")

# Transformaciones de augmentación más sofisticadas
class AdvancedAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
            
        # Rotación suave
        if torch.rand(1).item() > 0.7:
            angle = torch.randint(-15, 16, (1,)).item()
            img = TF.rotate(img, angle, fill=0)
        
        # Transformación afín suave
        if torch.rand(1).item() > 0.7:
            translate = (torch.randint(-3, 4, (1,)).item(), torch.randint(-2, 3, (1,)).item())
            scale = 0.9 + torch.rand(1).item() * 0.2  # 0.9 a 1.1
            shear = torch.randint(-5, 6, (1,)).item()
            img = TF.affine(img, angle=0, translate=translate, scale=scale, shear=shear, fill=0)
        
        # Flip horizontal ocasional
        if torch.rand(1).item() > 0.8:
            img = TF.hflip(img)
            
        # Añadir ruido gaussiano suave
        if torch.rand(1).item() > 0.8:
            noise = torch.randn_like(img) * 0.05
            img = torch.clamp(img + noise, 0, 1)
        
        return img

# Módulos de red más modernos
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class ImprovedOMRNet(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_rate=0.4):
        super(ImprovedOMRNet, self).__init__()
        
        # Encoder con bloques residuales
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)  # 40x10
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)  # 20x5
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 10x2
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 5x1
        
        # Módulo de atención
        self.attention = AttentionModule(256)
        
        # Global Average Pooling + Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier con capas adicionales
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Aplicar atención
        x = self.attention(x)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        return self.classifier(x)

# Función mejorada para generar imágenes
def get_improved_image(cropobj, target_size=(80, 20), margin=2):
    h, w = cropobj.height + 2 * margin, cropobj.width + 2 * margin
    canvas = np.zeros((h, w), dtype='uint8')
    canvas[margin:margin + cropobj.height, margin:margin + cropobj.width] = cropobj.mask
    canvas[canvas > 0] = 255
    
    # Convertir a tensor y redimensionar
    img = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0)
    img = TF.resize(img, target_size, antialias=True)
    
    # Normalizar
    img = img / 255.0
    
    # Binarizar con umbral adaptativo
    img = (img > 0.5).float()
    
    return img.squeeze(0).numpy()

# Función de entrenamiento mejorada
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# Carga de datos mejorada
print("Cargando documentos MUSCIMA...")
start = time.time()
CROPOBJECT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop/TFG-Info/ImageFolder/MUSCIMA-pp_v1.0/v1.0/data/cropobjects_manual')
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]
print(f"Documentos cargados en {time.time() - start:.2f} segundos.")

# Extraer y filtrar clases
class_to_objects = {}
for doc in docs:
    for c in doc:
        class_to_objects.setdefault(c.clsname, []).append(c)

filtered_classes = {k: v for k, v in class_to_objects.items() if len(v) >= config.min_samples}
label_map = {clsname: i for i, clsname in enumerate(filtered_classes.keys())}
print(f"\nClases usadas ({len(label_map)}): {list(label_map.keys())}")

# Generar dataset con augmentación mejorada
print("Generando datos con augmentación avanzada...")
augment = AdvancedAugmentation(p=0.7)

X, y = [], []
for clsname, objects in filtered_classes.items():
    for i, c in enumerate(objects[:config.min_samples]):
        img = get_improved_image(c, config.image_size)
        
        # Imagen original
        X.append(img)
        y.append(label_map[clsname])
        
        # Imagen aumentada (para algunos casos)
        if i < config.min_samples * config.augment_ratio:
            img_tensor = torch.tensor(img).unsqueeze(0)
            aug_img = augment(img_tensor).squeeze(0).numpy()
            X.append(aug_img)
            y.append(label_map[clsname])

X = np.array(X).reshape(-1, 1, config.image_size[0], config.image_size[1]).astype(np.float32)
y = np.array(y).astype(np.int64)

print(f"Dataset generado: {X.shape[0]} muestras, {len(label_map)} clases")

# Crear tensores y datasets
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

# Ponderación de clases
label_counts = Counter(y)
weights = torch.tensor([1.0 / label_counts[i] for i in range(len(label_map))]).to(config.device)

# Modelo y optimización
model = ImprovedOMRNet(input_channels=1, num_classes=len(label_map), 
                      dropout_rate=config.dropout_rate).to(config.device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"\nModelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")

if __name__ == '__main__':

    # Entrenamiento mejorado
    print("Iniciando entrenamiento...")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Entrenamiento
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Validación
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Guardar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        # Early stopping mejorado
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            # Guardar mejor modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_map': label_map
            }, 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping en época {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time:.2f} segundos")
    print(f"Mejor precisión de validación: {best_val_acc:.2f}%")

    # Cargar mejor modelo para evaluación
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluación final
    print("\nEvaluación final...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Métricas finales
    final_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Precisión final: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    # Matriz de confusión mejorada
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(all_labels, all_preds)
    class_names = list(label_map.keys())

    # Normalizar matriz de confusión
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title(f"Matriz de Confusión Normalizada - Precisión: {final_accuracy:.3f}")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Gráficas de entrenamiento
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    # Pérdida
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución de la Pérdida')
    ax1.legend()
    ax1.grid(True)

    # Precisión
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión (%)')
    ax2.set_title('Evolución de la Precisión')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Reporte de clasificación detallado
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n" + "="*80)
    print("REPORTE DE CLASIFICACIÓN DETALLADO")
    print("="*80)
    print(report)

    # Guardar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'final_accuracy': final_accuracy,
        'best_val_accuracy': best_val_acc,
        'num_classes': len(label_map),
        'num_samples': len(X),
        'label_map': label_map,
        'config': vars(config),
        'training_time': total_time
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados guardados en 'training_results.json'")
    print(f"Modelo guardado en 'best_model.pth'")
    print(f"Gráficas guardadas como 'confusion_matrix.png' y 'training_curves.png'")
