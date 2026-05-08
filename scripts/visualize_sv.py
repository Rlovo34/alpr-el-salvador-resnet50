import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.inference.predict import PlatePredictor
from tensorflow.keras.applications.resnet50 import preprocess_input

def visualize_sv_pipeline():
    """Genera visualizaciones del pipeline OCR para análisis y debugging.
    
    Procesa imágenes de placas salvadoreñas a través de cada etapa del pipeline
    (preprocesamiento, extracción de caracteres, clasificación) y genera
    dashboards visuales que muestran transformaciones intermedias y predicciones.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    sys.path.append(project_root)
    
    model_path = os.path.join(project_root, 'models', 'trained_model.h5')
    valid_dir = os.path.join(project_root, 'data', 'raw', 'valid', '1_EL _SALVADOR')
    output_dir = os.path.join(project_root, 'output_visuals')
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(valid_dir):
        print(f"Error: No se encuentra la carpeta en: {valid_dir}")
        return

    print("Iniciando visualización del pipeline OCR...")
    predictor = PlatePredictor(model_path)

    # Filtrar solo archivos de imagen
    image_files = [f for f in os.listdir(valid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(valid_dir, img_name)
        
        thresh, img_std = predictor.preprocess_image(img_path)
        char_images = predictor.extract_characters(thresh, img_std)
        
        final_text = ""
        char_preds = []
        if char_images:
            batch = np.array(char_images, dtype=np.float32)
            batch = preprocess_input(batch)
            predictions = predictor.model.predict(batch, verbose=0)
            for pred in predictions:
                char = predictor.classes[np.argmax(pred)]
                final_text += char
                char_preds.append(char)
        else:
            final_text = "NO_DETECTADO"

        num_chars = len(char_images)
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(f"Pipeline OCR | Imagen: {img_name} | Resultado: {final_text}", 
                     fontsize=16, fontweight='bold')

        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB))
        ax1.set_title("Imagen Estandarizada (600px)", fontsize=12)
        ax1.axis('off')

        ax2 = plt.subplot(2, 2, 2)
        ax2.imshow(thresh, cmap='gray')
        ax2.set_title("Máscara de Extracción (Black-Hat + Otsu)", fontsize=12)
        ax2.axis('off')

        if num_chars > 0:
            for i in range(num_chars):
                ax_char = plt.subplot(2, num_chars, num_chars + i + 1)
                ax_char.imshow(char_images[i])
                ax_char.set_title(f"'{char_preds[i]}'", fontsize=14, fontweight='bold', color='blue')
                ax_char.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(output_dir, f"visual_{img_name}")
        plt.savefig(save_path)
        print(f"Dashboard guardado: {save_path}")
        plt.show()

if __name__ == "__main__":
    visualize_sv_pipeline()