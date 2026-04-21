import cv2
import json
import joblib
import numpy as np
import base64

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D


# ---------------------------------------------------------------------------
# Load everything once when the server starts
# We don't want to reload models on every request — that would be very slow
# ---------------------------------------------------------------------------

# paths to saved artifacts
MODEL_PATH      = r"artifacts/saved_model.pkl"
CLASS_DICT_PATH = r"artifacts/class_dictionary.json"
DNN_PROTO       = r"artifacts/deploy.prototxt"
DNN_WEIGHTS     = r"artifacts/res10_300x300_ssd_iter_140000.caffemodel"

# confidence threshold — predictions below this return "Unknown"
# this value was determined from the FAR/FRR analysis in the notebook
REJECTION_THRESHOLD = 0.829


def load_models():
    """
    Load the face detector, VGG16 embedding model, classifier and class dict.
    Called once at server startup.
    """
    global face_net, embedding_model, classifier, class_dict, idx_to_name

    # OpenCV DNN face detector
    face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_WEIGHTS)

    # VGG16 feature extractor — same setup as the training notebook
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    out  = GlobalAveragePooling2D()(base.output)
    embedding_model = Model(inputs=base.input, outputs=out)
    for layer in embedding_model.layers:
        layer.trainable = False

    # trained ML classifier
    classifier = joblib.load(MODEL_PATH)

    # class dictionary — maps name → index
    with open(CLASS_DICT_PATH) as f:
        class_dict = json.load(f)

    # reverse mapping — index → name (used for predictions)
    idx_to_name = {v: k for k, v in class_dict.items()}

    print("All models loaded successfully")
    print("Classes:", list(class_dict.keys()))


def get_face(img):
    """
    Run the DNN face detector on the image.
    Returns the cropped face region, or the full image if no face found.
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf = 0
    best_box  = None

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > 0.5 and conf > best_conf:
            best_conf = conf
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            best_box = (x1, y1, x2, y2)

    if best_box:
        x1, y1, x2, y2 = best_box
        return img[y1:y2, x1:x2], best_box

    # return full image if no face detected — LFW images are already face-centred
    return img, None


def get_embedding(face_img):
    """
    Preprocess a face image and run it through VGG16 to get a 512-dim embedding.
    """
    face = cv2.resize(face_img, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.expand_dims(face.astype(np.float32), axis=0)
    face = preprocess_input(face)
    embedding = embedding_model.predict(face, verbose=0)
    return embedding  # shape: (1, 512)


def classify_image(image_base64):
    """
    Full pipeline: decode base64 image → detect face → embed → classify.
    
    Returns a dict with:
        - predicted_name: string name or "Unknown"
        - confidence: float (0–1)
        - all_probs: dict of name → probability for all classes
        - face_detected: bool
    """
    # decode base64 image coming from the browser
    # the browser sends: "data:image/jpeg;base64,/9j/4AAQ..."
    # we strip the header and decode the rest
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not read the image"}

    # detect and crop face
    face, box = get_face(img)
    face_detected = box is not None

    # get embedding and predict
    embedding = get_embedding(face)
    probs     = classifier.predict_proba(embedding)[0]
    max_conf  = float(probs.max())
    pred_idx  = int(probs.argmax())
    pred_name = idx_to_name[pred_idx]

    # apply rejection threshold
    if max_conf < REJECTION_THRESHOLD:
        result = "Unknown"
    else:
        result = pred_name

    # build probability dict for all classes (for display in the browser)
    all_probs = {
        idx_to_name[i]: round(float(p) * 100, 1)
        for i, p in enumerate(probs)
    }

    return {
        "predicted_name": result,
        "confidence"    : round(max_conf * 100, 1),
        "all_probs"     : all_probs,
        "face_detected" : face_detected
    }
