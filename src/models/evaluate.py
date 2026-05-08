import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    """Evalúa el modelo entrenado generando métricas y matriz de confusión.
    
    Lee el modelo persistido y el dataset de validación, generando:
    - Reporte de clasificación (precisión, recall, F1-score por clase)
    - Matriz de confusión visual (heatmap) para análisis de errores
    
    La matriz de confusión identifica qué caracteres se confunden frecuentemente,
    permitiendo detectar deficiencias del modelo (ej: "0" vs "O").
    """
    model = tf.keras.models.load_model("models/trained_model.h5")
    
    # Se utiliza shuffle=False para mantener orden consistente entre
    # predicciones reales en la matriz de confusión
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/train", image_size=(96, 96), shuffle=False)
    
    # Extrae todas las etiquetas verdaderas del dataset de validación
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    
    # Genera predicciones en lote para todo el dataset
    preds = model.predict(val_ds)
    # Extrae clase predicha (índice máximo) para cada muestra
    y_pred = np.argmax(preds, axis=1)

    # Reporte textual: Precisión, Recall, F1-score y soporte por clase
    print(classification_report(y_true, y_pred, target_names=val_ds.class_names))
    
    # Matriz de confusión: muestra aciertos en diagonal, errores fuera
    # Permite identificar confusiones específicas (ej: caracteres similares)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=val_ds.class_names, yticklabels=val_ds.class_names)
    plt.title("Matriz de Confusión - OCR Placas")
    plt.show()

if __name__ == "__main__":
    evaluate()