import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import SMOTE
from muscima.io import parse_cropobject_list

# Rutas
CROPOBJECT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop/TFG-Info/ImageFolder/MUSCIMA-pp_v1.0/v1.0/data/cropobjects_manual')
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

# Generar imagen
def get_image(cropobjs, margin=1):
    top = min(c.top for c in cropobjs)
    left = min(c.left for c in cropobjs)
    bottom = max(c.bottom for c in cropobjs)
    right = max(c.right for c in cropobjs)

    h, w = bottom - top + 2 * margin, right - left + 2 * margin
    canvas = np.zeros((h, w), dtype='uint8')
    for c in cropobjs:
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        canvas[_pt:_pt + c.height, _pl:_pl + c.width] += c.mask
    canvas[canvas > 0] = 1
    return resize(canvas, (40, 10))

# Extraer nota + tallo + bandera
def extract_notes(cropobjects):
    crop_dict = {c.objid: c for c in cropobjects}
    notes = {'quarter': [], 'half': [], 'eighth': [], 'sixteenth': [], 'whole': []}

    for c in cropobjects:
        if c.clsname in ['notehead-full', 'notehead-empty']:
            out_objs = [crop_dict[o] for o in c.outlinks]
            stem = next((o for o in out_objs if o.clsname == 'stem'), None)
            flags = [o.clsname for o in out_objs if 'flag' in o.clsname]

            if c.clsname == 'notehead-full' and stem and not flags:
                notes['quarter'].append((c, stem))
            elif c.clsname == 'notehead-full' and stem and 'flag8' in flags:
                flag_obj = next(o for o in out_objs if o.clsname == 'flag8')
                notes['eighth'].append((c, stem, flag_obj))
            elif c.clsname == 'notehead-full' and stem and 'flag16' in flags:
                flag_obj = next(o for o in out_objs if o.clsname == 'flag16')
                notes['sixteenth'].append((c, stem, flag_obj))
            elif c.clsname == 'notehead-empty' and stem:
                notes['half'].append((c, stem))
            elif c.clsname == 'notehead-empty' and not stem:
                notes['whole'].append((c,))
    return notes

# Obtener todas las notas
all_notes = {'quarter': [], 'half': [], 'eighth': [], 'sixteenth': [], 'whole': []}
for doc in docs:
    notes = extract_notes(doc)
    for k in all_notes:
        all_notes[k].extend(notes[k])

# Visualizar recuento
for k in all_notes:
    print(f"{k}: {len(all_notes[k])}")

# Preprocesar imágenes y etiquetas
X, y = [], []
label_map = {'quarter': 0, 'half': 1, 'eighth': 2, 'sixteenth': 3, 'whole': 4}
for cls, samples in all_notes.items():
    for group in samples:
        img = get_image(group)
        X.append(img.flatten())
        y.append(label_map[cls])

# Validar si hay clases vacías
for cls, label in label_map.items():
    if label not in y:
        print(f"⚠️ Aviso: No hay muestras de la clase '{cls}' (etiqueta {label})")

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Rebalanceo con SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Modelo MLP
clf = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42)
clf.fit(X_train_res, y_train_res)

# Resultados
y_pred = clf.predict(X_test)

# Reporte robusto: solo las clases presentes
present_labels = sorted(unique_labels(y_test, y_pred))
label_names = [k for k, v in label_map.items() if v in present_labels]

print(classification_report(
    y_test,
    y_pred,
    labels=present_labels,
    target_names=label_names
))
