import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from muscima.io import parse_cropobject_list

print("Cargando documentos MUSCIMA...")
start = time.time()

# Rutas
CROPOBJECT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop/TFG-Info/ImageFolder/MUSCIMA-pp_v1.0/v1.0/data/cropobjects_manual')
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

print(f"Documentos cargados en {time.time() - start:.2f} segundos.")

# Función para convertir un solo objeto a imagen
def get_image_single(cropobj, margin=1):
    h, w = cropobj.height + 2 * margin, cropobj.width + 2 * margin
    canvas = np.zeros((h, w), dtype='uint8')
    canvas[margin:margin + cropobj.height, margin:margin + cropobj.width] = cropobj.mask
    canvas[canvas > 0] = 1
    return resize(canvas, (40, 10))

print("Clasificando objetos por clase...")
start = time.time()

# Extraer todas las clases
class_to_objects = {}
for doc in docs:
    for c in doc:
        if c.clsname not in class_to_objects:
            class_to_objects[c.clsname] = []
        class_to_objects[c.clsname].append(c)

# Mostrar recuento de clases
for cls, objs in class_to_objects.items():
    print(f"{cls}: {len(objs)}")

print(f"Clasificación completada en {time.time() - start:.2f} segundos.")

# Filtrar clases con suficientes ejemplos (por ejemplo, >100)
min_samples = 100
filtered_classes = {k: v for k, v in class_to_objects.items() if len(v) >= min_samples}
label_map = {clsname: i for i, clsname in enumerate(filtered_classes.keys())}
print("\nClases usadas:", label_map)

print("Generando vectores de imagen (X, y)...")
start = time.time()

# Generar datos (X, y)
X, y = [], []
for clsname, objects in filtered_classes.items():
    for c in objects:
        img = get_image_single(c)
        X.append(img.flatten())
        y.append(label_map[clsname])

print(f"Generación completada en {time.time() - start:.2f} segundos.")
print(f"Total de muestras: {len(X)}")

# Entrenamiento
print("Partiendo en train/test y normalizando...")
start = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Partición y escalado completados en {time.time() - start:.2f} segundos.")

print("Aplicando SMOTE para rebalanceo...")
start = time.time()

# Rebalanceo
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"SMOTE completado en {time.time() - start:.2f} segundos.")
print(f"Nuevo tamaño del conjunto de entrenamiento: {len(X_train_res)}")

# Modelo
print("Entrenando MLPClassifier...")
start = time.time()

clf = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42, verbose = True)
clf.fit(X_train_res, y_train_res)

print(f"Entrenamiento completado en {time.time() - start:.2f} segundos.")

# Evaluación
print("Evaluando el modelo...")
start = time.time()

y_pred = clf.predict(X_test)

print(f"Predicción completada en {time.time() - start:.2f} segundos.")
target_names = list(label_map.keys())
labels = list(label_map.values())
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names, labels=labels))

