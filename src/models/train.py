import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIR_DATOS = os.path.join(BASE_DIR, 'data', 'processed', 'train')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.h5')

# ImageDataGenerator aplica normalización ResNet50 (centrado en cero, modo Caffe)
# necesaria para transferir conocimiento de pesos preentrenados en ImageNet
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_gen = datagen.flow_from_directory(DIR_DATOS, target_size=(96, 96), batch_size=32, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(DIR_DATOS, target_size=(96, 96), batch_size=32, class_mode='categorical', subset='validation', shuffle=False)

# ResNet50 con pesos preentrenados en ImageNet proporciona extractores de características
# de bajo nivel ya optimizados (bordes, texturas, patrones). Se reutiliza
# este conocimiento mediante transferencia de aprendizaje.
base_model = ResNet50(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

print("\nFASE 1: Entrenamiento de clasificador (ResNet50 congelada)")
print("-" * 70)
print("Se congela la base de ResNet50 para entrenar única y exclusivamente")
print("la capa clasificadora personalizada. Esto es crucial porque:")
print("  1. Los pesos de ImageNet ya son buenos extractores generales")
print("  2. El dataset sintético es pequeño (250 muestras/clase)")
print("  3. Evita sobreajuste fijando características de bajo nivel")
print()
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
# Capa densa de 512 neuronas: ResNet proporciona mapas de características
# complejos que necesitan reducir dimensionalidad antes de clasificación
x = Dense(512, activation='relu')(x)
# Dropout 0.5 reduce sobreajuste en capa clasificadora durante warm-up
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=10, validation_data=val_gen)

print("\nFASE 2: Fine-tuning selectivo de ResNet50")
print("-" * 70)
print("Se descongelan las últimas 30 capas (bloque residual final) para")
print("que se adapten a las características específicas de caracteres en placas.")
print("Se mantiene congelado el resto de la red para preservar")
print("extractores de bajo nivel que ya son óptimos para objetos generales.")
print()
base_model.trainable = True

# Descongelación selectiva: solo las últimas 30 capas
# Permite que ResNet se adapte a características específicas de OCR
# sin perder ventajas de transferencia de ImageNet
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Learning rate significantly reduced (0.00005) porque ahora estamos
# ajustando pesos ya entrenados. Un LR alto rompería el conocimiento.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# ReduceLROnPlateau: Si la pérdida no mejora, reduce LR a la mitad
# Permite ajustes más finos cuando el entrenamiento estabiliza
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=1)

# EarlyStopping: Detiene el entrenamiento cuando accuracy de validación
# deja de mejorar durante 5 épocas. Evita sobreajuste.
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

model.fit(train_gen, epochs=15, validation_data=val_gen, callbacks=[reduce_lr, early_stop])

model.save(MODEL_PATH)
print(f"\nModelo guardado en {MODEL_PATH}")