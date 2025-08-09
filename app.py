import os
# Forzar uso de CPU (evita warnings de cuDNN/cuBLAS)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_image_from_file
import tempfile

app = Flask(__name__)
CORS(app)  # Permitir llamadas desde otros dominios (frontend incluido)

@app.route("/predict", methods=["GET"])
def predict_get():
    return "Usa método POST con un archivo para obtener la predicción.", 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["file"]

    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Hacer predicción
        prediction = predict_image_from_file(tmp_path)

        return jsonify(prediction)

    except Exception as e:
        print("Error en predicción:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        # Asegurar que se borre el archivo temporal
        try:
            os.remove(tmp_path)
        except:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
