from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load YOLOv11 model
# model = YOLO("../weights/best.pt")
model = YOLO("face-weights/best.pt")
# Create a folder for uploaded and processed images
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "static"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save the uploaded file
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Run YOLOv11 on the image
            image = Image.open(img_path)
            results = model(image)

            # Get processed image with bounding boxes
            output_img = results[0].plot()
            output_img_path = os.path.join(PROCESSED_FOLDER, file.filename)

            # Save output image
            cv2.imwrite(output_img_path, output_img)

            # Redirect to result page
            return redirect(url_for("show_result", filename=file.filename))

    return render_template("upload.html")


@app.route("/result/<filename>")
def show_result(filename):
    processed_img_url = f"/static/{filename}"
    return render_template("result.html", processed_img_url=processed_img_url)


def generate_frames():
    camera = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run YOLOv11 detection
        results = model(frame)
        output_frame = results[0].plot()

        # Convert frame to JPEG format
        _, buffer = cv2.imencode(".jpg", output_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for streaming
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    camera.release()


@app.route("/webcam")
def webcam_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live")
def live_view():
    return render_template("webcam.html")


if __name__ == "__main__":
    app.run(debug=True)
