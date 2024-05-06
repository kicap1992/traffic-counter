from flask import Flask, render_template, Response, request,jsonify,send_from_directory
import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os

app = Flask(__name__, static_folder='assets')

video_list = []

color = (0, 255, 0)
color_red = (0, 0, 255)
thickness = 2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5

# Background subtraction menggunakan MOG2
subtracao = cv2.createBackgroundSubtractorMOG2()

jumlah_kenderaan = 0
kenderaan_kiri = 0
kenderaan_kanan = 0



# Define the generate_frames function with parameters for video, threshold, and state
def generate_frames(video, threshold, stat):
    model_path = "models/yolov8n.pt"
    cap = cv2.VideoCapture(video)
    model = YOLO(model_path)

    vehicle_ids = [2, 3, 5, 7]
    track_history = defaultdict(lambda: [])

    up = {}
    down = {}

    global jumlah_kenderaan
    global kenderaan_kiri
    global kenderaan_kanan

    jumlah_kenderaan = 0
    kenderaan_kiri = 0
    kenderaan_kanan = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        try:
            frame = imutils.resize(frame, width=1280, height=720)
            # freame_original = frame.copy()
            frame_color = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = model.track(frame_color, persist=True, verbose=False)[0]
            bboxes = np.array(results.boxes.data.tolist(), dtype="int")

            # Gambar garis pembatas untuk menghitung jumlah kendaraan yang melewati garis
            cv2.line(frame_color, (0, threshold), (1280, threshold), color, thickness)
            text_position = (620, threshold - 5)  # Adjust the Y coordinate to place the text just above the line
            cv2.putText(frame_color, "Pembatas Jalan", text_position, font, 0.7, color_red, thickness)
            

            for box in bboxes:
                x1, y1, x2, y2, track_id, score, class_id = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if class_id in vehicle_ids:
                    class_name = results.names[int(class_id)].upper()

                track = track_history[track_id]
                track.append((cx, cy))
                if len(track) > 20:
                    track.pop(0)

                points = np.hstack(track).astype("int32").reshape(-1, 1, 2)
                cv2.polylines(frame_color, [points], isClosed=False, color=color, thickness=thickness)
                cv2.rectangle(frame_color, (x1, y1), (x2, y2), color, thickness)
                text = "ID: {} {}".format(track_id, class_name)
                cv2.putText(frame_color, text, (x1, y1 - 5), font, font_scale, color, thickness)

                if cy > threshold - 5 and cy < threshold + 5 and cx < 670:
                    down[track_id] = x1, y1, x2, y2

                if cy > threshold - 5 and cy < threshold + 5 and cx > 670:
                    up[track_id] = x1, y1, x2, y2

            up_text = "Kanan:{}".format(len(list(up.keys())))
            down_text = "Kiri:{}".format(len(list(down.keys())))
            kenderaan_kanan = len(list(up.keys()))
            kenderaan_kiri = len(list(down.keys()))
            cv2.putText(frame_color, up_text, (1150, threshold - 5), font, 0.8, color_red, thickness)
            cv2.putText(frame_color, down_text, (0, threshold - 5), font, 0.8, color_red, thickness)

            # Background subtraction dan deteksi kontur
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi frame ke citra grayscale
            blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Reduksi noise menggunakan Gaussian Blur
            img_sub = subtracao.apply(blur)  # Background subtraction
            dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Dilasi untuk meningkatkan ketebalan objek
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel untuk operasi morfologi
            dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Operasi closing untuk mengisi lubang kecil pada objek
            dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  # Operasi closing tambahan
            contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Deteksi kontur objek
            frame_bw = cv2.cvtColor(dilatada, cv2.COLOR_GRAY2BGR)  # Konversi frame grayscale ke BGR

            if stat == 'color':
                frame_to_encode = frame_color
            elif stat == 'grayscale':
                frame_to_encode = frame_gray
            elif stat == 'original':
                frame_to_encode = frame
            else:  # Assuming 'detectar' state
                frame_to_encode = frame_bw

            _, buffer = cv2.imencode('.jpg', frame_to_encode)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print("Terjadi kesalahan:", str(e))
            continue

        jumlah_kenderaan = kenderaan_kiri + kenderaan_kanan
            

    cap.release()

def update_video_list():
    global video_list
    # add "video/" to the video_list and only take video extensions
    video_list = [f"video/{f}" for f in os.listdir("video") if f.endswith(".mp4")]

@app.route('/')
def index():
    update_video_list()
    print("video_list:", video_list)
    video = request.args.get('video', 'video/video.mp4')
    threshold = int(request.args.get('threshold', 450))
    # Pass the video file path and threshold value to the template
    return render_template('index.html', video=video, threshold=threshold, video_list=video_list)

def video_feed():
    # Get the video file path, threshold value, and state from the URL parameters
    video = request.args.get('video')
    threshold = int(request.args.get('threshold', 450))
    stat = request.args.get('stat', 'color')  # Default to 'color' if state is not specified
    # Return the response with the generator function
    print("ini semua variable:", video, threshold, stat)
    return Response(generate_frames(video, threshold, stat), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_list')
def video_list():
    update_video_list()
    return render_template('video_list.html', video_list=video_list)

@app.route('/videos/<path:video>')
def video(video):
    return send_from_directory('', video)

# Add route for the video feed
app.add_url_rule('/video_feed', 'video_feed', video_feed)

@app.route('/check_jumlah_kenderaan', methods=['GET'])
def check_jumlah_kenderaan():
    global jumlah_kenderaan
    global kenderaan_kiri
    global kenderaan_kanan
    return jsonify({'jumlah_kenderaan': jumlah_kenderaan, 'kenderaan_kiri': kenderaan_kiri, 'kenderaan_kanan': kenderaan_kanan})

UPLOAD_FOLDER = 'video'
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': False, 'message': 'No file selected'})
    
    if file:
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({'status': True, 'message': 'File uploaded successfully', 'filename': filename})

if __name__ == "__main__":
    app.run(debug=True)
