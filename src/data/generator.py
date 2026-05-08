import cv2
import numpy as np
import os
import random
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
DIR_OCR = os.path.join(project_root, 'data', 'processed', 'train')

if os.path.exists(DIR_OCR): 
    shutil.rmtree(DIR_OCR)
os.makedirs(DIR_OCR, exist_ok=True)

CARACTERES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FUENTES_SINTETICAS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]

def aplicar_shear(image):
    """Aplica transformación de corte (shear) para simular perspectiva.
    
    El factor de corte aleatorio [-0.15, 0.15] modela variaciones de ángulo
    que ocurren cuando las placas se capturan desde posiciones no frontales,
    mejorando la robustez del modelo ante distorsiones geométricas.
    """
    rows, cols, ch = image.shape
    shear_factor = random.uniform(-0.15, 0.15)
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))

print("Generando dataset sintético de caracteres...")

for char in CARACTERES:
    char_dir = os.path.join(DIR_OCR, char)
    os.makedirs(char_dir, exist_ok=True)
    
    # Se generan 250 muestras por carácter. Aunque modesto, este volumen es suficiente
    # cuando los datos son sintéticos de alta calidad y se aplican aumentaciones robustas
    for i in range(250):
        # Lienzo negro (0) como fondo, simulando placas oscuras
        img = np.zeros((96, 96, 3), dtype=np.uint8)

        fuente = random.choice(FUENTES_SINTETICAS)
        # Escala 2.5-3.5 para caracteres grandes que ocupen ~70% de la imagen
        escala = random.uniform(2.5, 3.5)
        grosor = random.randint(2, 6)
        x = random.randint(10, 25)
        y = random.randint(70, 85)

        # Texto blanco puro (255), máximo contraste
        cv2.putText(img, char, (x, y), fuente, escala, (255, 255, 255), grosor)
        img = aplicar_shear(img)

        # Erosión y dilatación simulan el efecto de degradación por luz solar,
        # reflejándose en la placa real. Se aplican de forma estocástica
        # para crear variabilidad sin romper completamente los caracteres
        if random.random() > 0.5:
            kernel = np.ones((random.randint(2,3), random.randint(2,3)), np.uint8)
            if random.random() > 0.5:
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)

        cv2.imwrite(os.path.join(char_dir, f"{char}_{i}.jpg"), img)

print(f"Dataset generado en {DIR_OCR}")