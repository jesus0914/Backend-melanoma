import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Solo CPU

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo una sola vez
MODEL_PATH = "malgnant-bening-cnn-ad.h5"
model = load_model(MODEL_PATH)
classes = ['Benigno', 'Maligno']

def predict_image_from_file(file_path):
    # Preprocesar imagen
    img = Image.open(file_path).convert("L")  # Escala de grises
    img = img.resize((200, 200))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Canal
    img_array = np.expand_dims(img_array, axis=0)   # Batch

    # Hacer predicción
    prediction = model.predict(img_array, verbose=0)
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    # Generar recomendación
    if predicted_class == "Maligno":
        recomendacion = "Consulta a un dermatólogo lo antes posible."
    else:
        recomendacion = "Parece benigno, pero realiza seguimiento médico."

    return {
        "prediccion": predicted_class,
        "confianza": f"{confidence:.2%}",
        "recomendacion": recomendacion
    }
