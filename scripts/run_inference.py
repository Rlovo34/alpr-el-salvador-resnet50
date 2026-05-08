import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference.predict import PlatePredictor

def main():
    """Ejecuta inferencia de OCR sobre una imagen de placa proporcionada.
    
    Lee una imagen desde línea de comandos y retorna el texto extraído
    de la placa vehicular junto con su confianza predicha.
    """
    parser = argparse.ArgumentParser(description="OCR de placas vehiculares")
    parser.add_argument("--input", required=True, help="Ruta de la imagen de entrada")
    args = parser.parse_args()

    predictor = PlatePredictor("models/trained_model.h5")
    res = predictor.run(args.input)
    
    print(f"Input: {args.input}")
    print(f"Prediction: \"{res['text']}\"")
    print(f"Confidence: {res['confidence']:.2f}")

if __name__ == "__main__":
    main()