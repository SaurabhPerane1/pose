from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("yoga_pose_classifier1.keras")
pose_classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

def preprocess_frame(frame):
    img = cv2.resize(frame, (150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = preprocess_frame(frame_rgb)
            prediction = model.predict(input_img, verbose=0)
            predicted_class = pose_classes[np.argmax(prediction)]

            cv2.putText(
                frame,
                f'Pose: {predicted_class}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
