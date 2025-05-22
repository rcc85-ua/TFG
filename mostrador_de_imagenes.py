import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

# Ruta a MUSCIMA++ (adaptada a tu variable de entorno y ruta)
CROPOBJECT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop/TFG-Info/ImageFolder/MUSCIMA-pp_v1.0/v1.0/data/cropobjects_manual')

# --- Cargar MNIST ---
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
imgs_mnist = [mnist_train[i][0].squeeze().numpy() for i in range(5)]  # primeras 5 imágenes

# --- Cargar MUSCIMA++ ---
imgs_muscima = []
for i, filename in enumerate(sorted(os.listdir(CROPOBJECT_DIR))):
    if i >= 5:
        break
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(CROPOBJECT_DIR, filename)
        img = Image.open(img_path).convert('L')  # escala de grises
        img = np.array(img) / 255.0
        imgs_muscima.append(img)

# Función para mostrar imágenes en matplotlib
def show_images(images, title):
    plt.figure(figsize=(15, 3))
    plt.suptitle(title, fontsize=16)
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

# Mostrar MNIST
show_images(imgs_mnist, 'Ejemplos de MNIST')

# Mostrar MUSCIMA++
show_images(imgs_muscima, 'Ejemplos de MUSCIMA++')


