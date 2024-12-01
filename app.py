from flask import Flask, Response, render_template
import cv2
from fer import FER
import os

app = Flask(__name__)
detector = FER(mtcnn=True)  # Emotion detector with MTCNN for face detection

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the webcam for live feed
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect emotions in the frame
        result = detector.detect_emotions(frame)
        if result:
            for face in result:
                emotions = face['emotions']
                top_emotion = max(emotions, key=emotions.get)
                (x, y, w, h) = face["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{top_emotion}: {emotions[top_emotion]:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                if top_emotion in ['angry', 'fear', 'disgust', 'sad']:
                    stress_level = "Stressed"
                else:
                    stress_level = "Not Stressed"
                cv2.putText(frame, stress_level, (x, y + h + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert frame to JPEG format for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for video streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home route."""
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)