from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from inference.predict import MODEL_PATH, predict_image
#python app.py

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    })


@app.post("/api/predict")
def predict_upload():
    image_file = request.files.get("image")

    if image_file is None or image_file.filename == "":
        return jsonify({"error": "Please upload an image file."}), 400

    try:
        image = Image.open(image_file.stream)
        result = predict_image(image)
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    except RuntimeError as error:
        return jsonify({"error": f"Prediction failed: {error}"}), 500

    return jsonify(result)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image is too large. Please upload a file under 8 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
