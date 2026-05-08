import os
import sys
import glob

from inference.predict import PlatePredictor

def evaluate_bulk():
    """Ejecuta evaluación masiva de placas sobre múltiples jurisdicciones geográficas.
    
    Realiza búsqueda recursiva en data/raw para procesar todas las imágenes JPEG
    de diferentes estados/territorios, extrayendo OCR de placas vehiculares.
    Los resultados se presentan en formato tabular con estado y predicción.
    """
    # Resolución de rutas del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    sys.path.append(os.path.join(project_root, "src"))
    
    raw_dir = os.path.join(project_root, "data", "raw")
    model_path = os.path.join(project_root, "models", "trained_model.h5")
    
    # Se utiliza glob con patrón recursivo (**/) para acceder a subdirectorios
    # de múltiples niveles (ej: valid/ALABAMA/image.jpg), permitiendo procesamiento
    # unificado sin dependencia de estructura de directorios específica
    search_pattern = os.path.join(raw_dir, "**", "*.jpg")
    image_paths = glob.glob(search_pattern, recursive=True)
    
    if not image_paths:
        print(f"❌ Error: No se encontraron imágenes en {raw_dir}")
        print("Asegúrate de que las carpetas /valid/alabama/... estén dentro de data/raw/")
        return
        
    print(f"--- Iniciando Evaluación Masiva ---")
    print(f"Imágenes detectadas: {len(image_paths)}")
    
    try:
        predictor = PlatePredictor(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    print(f"\n{'ESTADO/CARPETA':<20} | {'ARCHIVO':<20} | {'PREDICCIÓN'}")
    print("-" * 70)
    
    # Se limita a 100 imágenes para mantener tiempo de ejecución razonable
    # en evaluaciones exploratorias sobre datasets grandes
    for img_path in image_paths[:100]:
        partes = img_path.split(os.sep)
        estado = partes[-2]
        filename = partes[-1]
        
        try:
            result = predictor.run(img_path)
            texto = result['text'] if result['text'] else "[VACÍO]"
            print(f"{estado[:20]:<20} | {filename[:20]:<20} | {texto}")
        except Exception as e:
            print(f"{estado[:20]:<20} | {filename[:20]:<20} | ERROR TÉCNICO")
            
    print("-" * 70)
    print(f"Evaluación completada.")
    

if __name__ == "__main__":
    evaluate_bulk()