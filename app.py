from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the trained model
model = load_model("models/keras_Model.h5", compile=False)

# Load the labels for the model
with open("models/labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Initialize global variables for video file processing
video_stream = None
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera not accessible")

def process_frame(frame):
    """
    Process a single frame for prediction.
    """
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    normalized_frame = (np.array(resized_frame, dtype=np.float32) / 127.5) - 1
    model_input = np.expand_dims(normalized_frame, axis=0)

    prediction = model.predict(model_input, verbose=0)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    return f"{predicted_class.strip()} ({confidence_score * 100:.2f}%)"

def generate_frames(source="live"):
    camera = cv2.VideoCapture(0)
    cap = camera if source == "live" else cv2.VideoCapture(video_stream)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Unable to read from camera")
            break

        prediction_text = process_frame(frame)
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            print("Error: Frame encoding failed")
            continue
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/videofeed/live")
def live_feed():
    return Response(generate_frames(source="live"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/videofeed/uploaded")
def uploaded_feed():
    return Response(generate_frames(source="uploaded"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload", methods=["POST"])
def upload_video():
    global video_stream

    if "video" not in request.files:
        return redirect(url_for("index"))

    video = request.files["video"]
    if video.filename == "":
        return redirect(url_for("index"))

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)
    video_stream = video_path

    return redirect(url_for("uploaded_feed"))

@app.teardown_appcontext
def cleanup(exception=None):
    camera.release()

if __name__ == "__main__":
    app.run(debug=True)