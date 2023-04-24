from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np
from keras.models import load_model

import sounddevice as sd

import wavio
import whisper

app = Flask(__name__)

# Load the pre-trained emotion recognition model
model = load_model('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize the camera and the cascade classifier
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to capture video feed and detect emotions in real-time
def gen_frames():
    while True:
        # Capture the video frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the face in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Loop through the detected faces and classify them into one of the possible emotion classes
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                resized_roi = cv2.resize(face_roi, (48, 48))
                normalized_roi = resized_roi / 255.0
                reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
                result = model.predict(reshaped_roi)[0]
                label = emotion_dict[np.argmax(result)]

                # Draw a rectangle around the detected face and display the predicted emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def recordSound():

    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording

    print('start recording')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print('end recording')

    wavio.write("output.wav", myrecording, fs, sampwidth=3) # Save as WAV file 

    transcript = "no text"
    model = whisper.load_model("base")
    result = model.transcribe("output.wav", fp16=False, word_timestamps=True)

    ideal_seconds_per_word = 60 / 115 # 115 wpm

    unclear_words = []
    hesitation_words = []
    pause_timestamps = []
    
    last_word = None
    for segment in result["segments"]:
        for cur_word in segment['words']:
            print(cur_word)
            if cur_word['probability'] < 0.7:
                unclear_words.append((cur_word['word'], cur_word['probability']))
            if cur_word['end'] - cur_word['start'] > ideal_seconds_per_word:
                hesitation_words.append((cur_word['word'], cur_word['end'] - cur_word['start']))
            
            if last_word and cur_word['start'] - last_word['end'] > 0:
                pause_timestamps.append((last_word['end'], cur_word['start']))
        
        last_word = cur_word
    print("Unclear Words:", unclear_words)
    print("Hesitation Words:", hesitation_words)
    print("Pause Timestamps:", pause_timestamps)


    unclear = str(unclear_words).strip('[]')

    hesitation = str(hesitation_words).strip('[]')

    pause =  str(pause_timestamps).strip('[]')

    transcript = result["text"]
    

    return transcript, unclear, hesitation, pause

@app.route('/', methods=["GET", "POST"])
def index():
    transcript = ""
    unclear = ""
    hesitation = ""
    pause = ""
    if request.method == "POST":
        if "transcribe" in request.form:
            print("Form Data Received")

            if "file" not in request.files:
                return redirect(request.url)
            
            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)
            
            if file:
                pass
        elif "record" in request.form:
            transcript, unclear, hesitation, pause = recordSound()
    return render_template('index.html', transcript=transcript, unclear=unclear, hesitation=hesitation, pause=pause)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
