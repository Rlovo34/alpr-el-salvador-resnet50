# OCR Híbrido para Placas de El Salvador

## 📋 Descripción del Proyecto

Sistema integral de reconocimiento óptico de caracteres (OCR) especializado en la detección y lectura automática de placas vehiculares salvadoreñas. Implementa una arquitectura híbrida que combina **visión por computadora avanzada (OpenCV)** con **aprendizaje profundo (ResNet50)** para lograr resultados robustos bajo diversas condiciones de iluminación y ángulos de captura.

**Características principales:**
- Preprocesamiento inteligente mediante filtros morfológicos Black-Hat y umbralización de Otsu
- Red neuronal convolucional ResNet50 transferida y refinada
- Segmentación de caracteres adaptativos
- Clasificación de 30 clases (dígitos 0-9 y letras A-Z)
- Soporte para dataset sintético binario y datos reales de validación

---

## 🏗️ Arquitectura del Sistema

El sistema opera en tres fases integradas:

### **Fase 1: Preprocesamiento (OpenCV)**
- **Estandarización**: Redimensiona la imagen a 600px de ancho manteniendo relación de aspecto
- **Conversión a escala de grises**: Optimiza el procesamiento
- **Filtro Black-Hat**: Extrae caracteres oscuros sobre fondo claro (crítico para placas reflectantes)
- **Umbralización de Otsu**: Convierte a imagen binaria automáticamente
- **Dilatación**: Suelda trazos rotos causados por reflejos del metal

### **Fase 2: Segmentación de Caracteres**
- Detecta contornos individuales en la imagen binarizada
- Extrae cada carácter como región de interés (ROI)
- Aplica padding dinámico (18% vertical, 40% horizontal)
- Redimensiona cada carácter a 96×96 píxeles normalizados

### **Fase 3: Clasificación con ResNet50 (Aprendizaje Profundo)**

#### **Etapa de Entrenamiento: Warm-Up (10 épocas)**
- Red base ResNet50 congelada
- Entrenamiento exclusivo de capas clasificadoras
- Dense(512) + Dropout(0.5) → Dense(36, softmax)
- Optimizador Adam con learning_rate=0.001

#### **Etapa de Fine-Tuning (N épocas)**
- Descongelamiento de últimas 30 capas de ResNet50
- Aprendizaje diferenciado con learning_rate=0.00001
- Callbacks: ReduceLROnPlateau, EarlyStopping

---

## 📁 Estructura de Directorios

```
PROYECTO PLACAS/
|
├── 📄 README.md                    # Este archivo
├── 📄 requirements.txt             # Dependencias del proyecto
│
├── 📂 data/                        # Datasets organizados
│   ├── raw/                        # Datos crudos de validación
│   │   └── valid/
│   │       ├── 1_EL _SALVADOR/     # Imágenes de El Salvador
│   │       ├── ALABAMA/            # Referencias comparativas
│   │       ├── CALIFORNIA/
│   │       └── ... (otros estados)
│   │
│   └── processed/                  # Datos pre-procesados para entrenamiento
│       └── train/
│           ├── 0 - 9/             # Carpetas para dígitos (250 muestras c/u)
│           └── A - Z/             # Carpetas para letras (250 muestras c/u)
│
├── 📂 models/                      # Modelos entrenados
│   └── trained_model.h5           # Modelo ResNet50 serializado (Keras)
│
├── 📂 output_visuals/              # Resultados visuales de predicción
│   ├── visual_1.jpg
│   ├── visual_2.jpg
│   └── ... (salidas del pipeline)
│
├── 📂 scripts/                     # Scripts de utilidad
│   ├── batch_test.py              # Evaluación masiva de imágenes
│   ├── run_inference.py           # Inferencia en imagen individual
│   └── visualize_sv.py            # Visualización del pipeline completo
│
└── 📂 src/                         # Módulos reutilizables
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── generator.py            # Generador sintético de dataset
    ├── models/
    │   ├── __init__.py
    │   ├── train.py               # Script de entrenamiento
    │   └── evaluate.py            # Evaluación de métricas
    └── inference/
        ├── __init__.py
        └── predict.py             # Motor de predicción e inferencia
```

---

## 🔧 Requisitos Previos e Instalación

### **Requisitos del Sistema**
- Python 3.8 o superior
- CUDA 11.0+ (opcional, acelera GPU)
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio disponible

### **Pasos de Instalación**

#### 1. **Clonar o descargar el proyecto**

"Debido al límite de tamaño de GitHub, las carpetas moldels,data y output_visuals no se encuentra en este repositorio. Puedes descargarlo desde: https://drive.google.com/drive/folders/1I3MJ1wtMx_JNavS3nyOr5-qcG5X42L-w?usp=sharing y colocarlo en la carpeta del proyecto antes de ejecutar la inferencia."

```bash
# Navegar a la carpeta del proyecto
cd "c:\Users\usuario\Deskto\PROYECTO PLACAS"
```

#### 2. **Crear entorno virtual (recomendado)**
```powershell
# Crear entorno virtual
python -m venv parqueo_ocr

# Activar en Windows PowerShell
.\parqueo_ocr\Scripts\Activate.ps1

# Activar en Windows CMD
parqueo_ocr\Scripts\activate.bat

# Activar en macOS/Linux
source parqueo_ocr/bin/activate
```

#### 3. **Instalar dependencias**
```bash
# Instalar paquetes desde requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. **Verificar instalación**
```python
python -c "import tensorflow; import cv2; import numpy; print('✓ Dependencias OK')"
```

### **Dependencias Principales**
- `tensorflow>=2.10.0` - Aprendizaje profundo
- `keras>=2.10.0` - API de modelos
- `opencv-python>=4.6.0` - Visión por computadora
- `numpy>=1.21.0` - Computación numérica
- `matplotlib>=3.5.0` - Visualización

---

## 🚀 Instrucciones de Uso

### **1. Entrenamiento del Modelo (Opcional)**

Si deseas entrenar un nuevo modelo:

```bash
# Primero, generar dataset sintético
python src/data/generator.py

# Luego, entrenar el modelo
python src/models/train.py

# Evaluar rendimiento en validación
python src/models/evaluate.py
```

**Resultado esperado:** Archivo `models/trained_model.h5` actualizado

---

### **2. Inferencia en Imagen Individual**

Procesar una sola imagen de placa:

```bash
python scripts/run_inference.py --input "ruta/a/imagen.jpg"
```

**Ejemplo:**
```bash
python scripts/run_inference.py --input "data/raw/valid/1_EL _SALVADOR/placa_001.jpg"
```

**Salida del programa:**
```
Input: data/raw/valid/1_EL _SALVADOR/placa_001.jpg
Prediction: "SV123456"
Confidence: 0.95
```

**Parámetros:**
- `--input`: Ruta obligatoria de la imagen a procesar

---

### **3. Evaluación Masiva (Batch Test)**

Procesar múltiples imágenes desde estructura de carpetas:

```bash
python scripts/batch_test.py
```

**Comportamiento:**
- Busca recursivamente todas las imágenes (*.jpg) en `data/raw/`
- Procesa hasta 100 imágenes
- Extrae automáticamente el nombre del estado desde la ruta

**Salida de ejemplo:**
```
--- 🚀 Iniciando Evaluación Masiva ---
Imágenes detectadas en subcarpetas: 87

ESTADO/CARPETA       | ARCHIVO              | PREDICCIÓN
--------------------------------------------------------------------
1_EL _SALVADOR       | placa_001.jpg        | SV123456
1_EL _SALVADOR       | placa_002.jpg        | SV654321
ALABAMA              | placa_003.jpg        | AL789012
```

---

### **4. Visualización del Pipeline Completo**

Generar visualizaciones detalladas del proceso de OCR:

```bash
python scripts/visualize_sv.py
```

**Genera:**
- Imágenes preprocesadas (Black-Hat + Otsu)
- Regiones segmentadas de caracteres
- Predicciones individuales con confianza
- Resultado final anotado

**Salida:** Imágenes guardadas en `output_visuals/`

---

## 📊 Estructura de Datos

### **Dataset de Entrenamiento (Sintético)**
```
data/processed/train/
├── 0/ → 250 imágenes (96×96, blanco sobre negro)
├── 1/ → 250 imágenes
├── ...
├── 9/ → 250 imágenes
├── A/ → 250 imágenes
├── B/ → 250 imágenes
└── Z/ → 250 imágenes

Total: 36 clases × 250 muestras = 9,000 imágenes
```

### **Dataset de Validación (Real)**
```
data/raw/valid/
├── 1_EL _SALVADOR/ → Placas salvadoreñas originales
├── ALABAMA/ → Placas de referencia (formato diferente)
└── ... (otros estados para comparación)
```

---

## 🔍 Detalles Técnicos

### **Preprocesamiento de Imagen**
- **Kernel Black-Hat:** 23×23 píxeles (optimizado para placas SV)
- **Kernel Dilatación:** 2×2 píxeles (suelda caracteres)
- **Padding Dinámico:** 18% vertical, 40% horizontal
- **Tamaño normalizado:** 96×96 píxeles (entrada ResNet50)

### **Modelo ResNet50**
- **Pesos iniciales:** ImageNet pre-entrenado
- **Entrada:** 96×96×3 (RGB)
- **Preprocesamiento:** Centrado en cero (modo Caffe de ResNet50)
- **Capas adicionales:**
  - GlobalAveragePooling2D
  - Dense(512, ReLU)
  - Dropout(0.5)
  - Dense(36, Softmax)

### **Hiperparámetros de Entrenamiento**
| Fase | Épocas | Learning Rate | Frozen Layers |
|------|--------|---------------|---------------|
| Warm-Up | 10 | 0.001 | Base (ResNet50) |
| Fine-Tuning | N | 0.00005 | Primeras 30 |

---

## ⚠️ Consideraciones Importantes

1. **Calidad de imágenes:** El sistema funciona mejor con imágenes de 600px+ de ancho
2. **Reflejos metálicos:** El filtro Black-Hat está optimizado para capturar placas reflectantes
3. **Memoria:** Procesamiento de batch requiere ~1GB RAM por 100 imágenes
4. **GPU opcional:** Sin GPU, la inferencia toma ~2-3 segundos por imagen

---

## 📈 Métricas de Desempeño Esperadas

| Métrica | Valor |
|---------|-------|
| **Precisión (Caracter)** | >90% |
| **Precisión (Placa Completa)** | >85% |
| **Tiempo de Inferencia** | 2-3 segundos (CPU) / 200ms (GPU) |
| **Memoria Modelo** | ~95 MB |

---

## 🐛 Troubleshooting

### **Problema:** `ModuleNotFoundError: No module named 'src'`
**Solución:** Asegúrate de ejecutar scripts desde la raíz del proyecto y que PYTHONPATH incluya `src/`

### **Problema:** `Model not found: models/trained_model.h5`
**Solución:** Entrena primero con `python src/models/train.py` o descarga el modelo pre-entrenado

### **Problema:** Imágenes no detectadas en batch_test.py
**Solución:** Verifica que las imágenes estén en `data/raw/valid/` con estructura correcta de carpetas

### **Problema:** Bajo rendimiento en placas reales
**Solución:** Ajusta parámetros de preprocessing (kernel size, padding) en `src/inference/predict.py`

---

## 📞 Soporte y Contribuciones

Para reportar problemas o sugerencias, revisa los logs en el terminal y valida:
- ✓ Ruta del modelo existe
- ✓ Imágenes en formato JPG/PNG válidas
- ✓ Entorno virtual activado
- ✓ TensorFlow 2.10+ instalado

---

## 📝 Licencia

Proyecto desarrollado con fines educativos y de investigación.

---

**Última actualización:** Abril 2026  
**Versión:** 1.0
