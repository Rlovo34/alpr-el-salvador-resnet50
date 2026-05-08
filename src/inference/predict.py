import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

class PlatePredictor:
    """Motor de inferencia para OCR de placas vehiculares salvadoreñas.
    
    Orquesta el pipeline completo: preprocesamiento (morfología), segmentación
    de caracteres (geometría especializada) y clasificación (ResNet50 preentrenada).
    Diseñado específicamente para las características visuales de placas SV:
    - Contraste alto (negro/blanco)
    - Geometría de caracteres predecible
    - Presencia de texto adicional (EL SALVADOR, CENTRO AMERICA)
    """
    
    def __init__(self, model_path):
        """Inicializa predictor cargando modelo preentrenado.
        
        Args:
            model_path: Ruta al modelo entrenado (formato .h5)
        """
        self.model = tf.keras.models.load_model(model_path)
        self.classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def standardize_image(self, img):
        """Normaliza el ancho de imagen a 600px manteniendo proporción.
        
        Se utiliza 600px como estandar para asegurar consistencia en el
        procesamiento morfológico posterior (kernels de Black-Hat).
        El redimensionamiento mantiene la relación de aspecto para evitar
        distorsión de caracteres en el eje vertical.
        """
        height, width = img.shape[:2]
        new_width = 600
        new_height = int((new_width / float(width)) * height)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def preprocess_image(self, image_path):
        """Extrae máscara binaria de caracteres usando morfología matemática.
        
        Secuencia de operaciones:
        1. Estandarización de escala (600px de ancho)
        2. Conversión a escala de grises
        3. Filtro Gaussiano (suavizado para reducir ruido de reflejos metálicos)
        4. Black-Hat: Extrae características oscuras sobre fondo claro
           - Kernel 23x23 seleccionado empíricamente para placas SV
        5. Umbralización de Otsu: Conversión automática a binario
        6. Dilatación controlada: Suelda trazos rotos sin perder precisión
        
        Args:
            image_path: Ruta a la imagen JPEG
            
        Returns:
            thresh_clean: Máscara binaria (0=fondo, 255=carácter)
            img: Imagen original estandarizada
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error cargando imagen: {image_path}")
        
        img = self.standardize_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Filtro Gaussiano suaviza sin perder definición de bordes
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
        
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh_clean = cv2.dilate(thresh, kernel_dil, iterations=1)
                                           
        return thresh_clean, img

    def _format_char_image(self, char_roi, h):
        """Transforma región de carácter a tensor normalizado de entrada (96x96 RGB).
        
        Estrategia:
        1. Padding dinámico (18% arriba/abajo, 40% izquierda/derecha) para
           otorgar espacio contextual al clasificador
        2. Creación de lienzo cuadrado: Evita deformación al redimensionar
           rect-angular a cuadrado (mantiene proporciones del carácter)
        3. Redimensión a 96x96 (entrada de ResNet50)
        4. Conversión a RGB (ResNet espera 3 canales)
        
        Los coeficientes de padding fueron calibrados empíricamente
        en dataset de validación de placas salvadoreñas.
        
        Args:
            char_roi: Región binaria del carácter extraído
            h: Alto de la región (usado para calcular padding dinámico)
            
        Returns:
            char_resized: Tensor 96x96 RGB normalizado para clasificación
        """
        pad_y = int(h * 0.18)
        pad_x = int(h * 0.40)
        padded_char = cv2.copyMakeBorder(char_roi, pad_y, pad_y, pad_x, pad_x, 
                                         cv2.BORDER_CONSTANT, value=0)
        
        ph, pw = padded_char.shape
        sq_size = max(ph, pw)
        fondo = np.zeros((sq_size, sq_size), dtype=np.uint8)
        
        y_off = (sq_size - ph) // 2
        x_off = (sq_size - pw) // 2
        fondo[y_off:y_off+ph, x_off:x_off+pw] = padded_char
        
        char_resized = cv2.resize(fondo, (96, 96), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(char_resized, cv2.COLOR_GRAY2RGB)

    def extract_characters(self, thresh, img_original):
        """Segmenta caracteres individuales usando análisis de contornos y filtros geométricos.
        
        Filtra contornos candidatos mediante 3 criterios:
        
        FILTRO 1 - Área mínima (1500 px²):
            Desecha ruido microscópico y artefactos de binarización que
            no pueden ser caracteres válidos (ancho mínimo ~20px)
        
        FILTRO 2 - Geometría esperada de letra:
            - Alto: entre 30%-75% del alto total de imagen
            - Proporción: 0.15 < ancho/alto < 0.95
            Estos rangos se calibraron contra muestras reales de placas SV
        
        FILTRO 3 - Banda de oro (posición vertical):
            Solo acepta caracteres cuyo centro de gravedad esté en la 
            franja central 35%-65% de la imagen. Esto rechaza el texto
            decorativo ("EL SALVADOR", "CENTRO AMERICA") que típicamente
            aparece arriba o abajo de los caracteres de placa.
        
        Poda final: Las placas SV tienen máximo 8 caracteres.
        Si se detectan más, se retienen los 8 de mayor área.
        
        Args:
            thresh: Máscara binaria del preprocesamiento
            img_original: Imagen original (para obtener dimensiones)
            
        Returns:
            characters: Lista de arrays 96x96 RGB en orden izquierda-derecha
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        characters = []
        img_h, img_w = img_original.shape[:2]

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            if area < 1500:
                continue
                
            aspect_ratio = w / float(h)
            center_y = y + (h / 2.0)

            if (img_h * 0.30) < h < (img_h * 0.75) and 0.15 < aspect_ratio < 0.95:
                if (img_h * 0.35) < center_y < (img_h * 0.65):
                    char_roi = thresh[y:y+h, x:x+w]
                    char_rgb = self._format_char_image(char_roi, h)
                    characters.append((x, char_rgb, area))
                
        if len(characters) > 8:
            characters.sort(key=lambda item: item[2], reverse=True)
            characters = characters[:8]
            
        characters.sort(key=lambda item: item[0])
        
        return [char[1] for char in characters]

    def run(self, image_path):
        """Ejecuta el pipeline completo de OCR: preprocesamiento -> segmentación -> clasificación.
        
        Args:
            image_path: Ruta a imagen JPEG de placa
            
        Returns:
            dict: {
                'text': Texto extraído (ej: "ABC1234") o "NO_DETECTADO",
                'confidence': Confianza promedio del clasificador [0, 1]
            }
        """
        thresh, img = self.preprocess_image(image_path)
        char_images = self.extract_characters(thresh, img)
        
        if not char_images:
            return {"text": "NO_DETECTADO", "confidence": 0.0}

        batch = np.array(char_images, dtype=np.float32)
        batch = preprocess_input(batch)
        
        predictions = self.model.predict(batch, verbose=0)
        
        final_text = ""
        total_confidence = 0.0
        
        for pred in predictions:
            conf = np.max(pred)
            pred_class = self.classes[np.argmax(pred)]
            final_text += pred_class
            total_confidence += conf
            
        avg_confidence = total_confidence / len(char_images)
        return {"text": final_text, "confidence": avg_confidence}