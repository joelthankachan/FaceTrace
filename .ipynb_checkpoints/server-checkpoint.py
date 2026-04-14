from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("app.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    result = util.classify_image(data["image"])
    return jsonify(result)


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": list(util.class_dict.keys())})


if __name__ == "__main__":
    print("Loading models — please wait...")
    util.load_models()
    print("FaceTrace is running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
