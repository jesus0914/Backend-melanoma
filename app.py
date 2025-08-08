from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_image_from_file
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Permite llamadas desde otros orígenes (como localhost:3000)

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

        # Realizar predicción
        prediction = predict_image_from_file(tmp_path)

        # Eliminar archivo temporal
        os.remove(tmp_path)

        return jsonify(prediction)

    except Exception as e:
        print("Error en predicción:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
