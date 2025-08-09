import os

# Desactivar GPU y silenciar logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Solo CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Oculta INFO y WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Evita logs de optimización

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo una sola vez al iniciar el backend
model = load_model("malgnant-bening-cnn-ad.h5")
classes = ['Benigno', 'Maligno']

def predict_image_from_file(file_path):
    # Abrir y procesar la imagen en escala de grises
    img = Image.open(file_path).convert("L")  # Escala de grises
    img = img.resize((200, 200))  # Ajustar al tamaño usado en el entrenamiento
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Añadir canal para grises
    img_array = np.expand_dims(img_array, axis=0)   # Añadir batch

    # Hacer la predicción
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    # Recomendación médica
    if predicted_class == "Maligno":
        recomendacion = "Consulta a un dermatólogo lo antes posible para una evaluación médica especializada."
    else:
        recomendacion = "La lesión parece benigna, pero se recomienda seguimiento médico periódico."

    return {
        "prediccion": predicted_class,
        "confianza": f"{confidence:.2%}",
        "recomendacion": recomendacion
    }
