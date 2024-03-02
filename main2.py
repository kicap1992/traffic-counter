import cv2  # Import library OpenCV untuk pengolahan citra dan video
import imutils  # Import library imutils untuk mempermudah manipulasi citra
import numpy as np  # Import library numpy untuk operasi numerik
from ultralytics import YOLO  # Import class YOLO dari library ultralytics untuk deteksi objek
from collections import defaultdict  # Import class defaultdict dari library collections untuk struktur data default dictionary

color = (0, 255, 0)  # Warna hijau untuk penggambaran objek dan garis
color_red = (0, 0, 255)  # Warna merah untuk teks dan garis
thickness = 2  # Ketebalan garis untuk penggambaran objek dan garis

font = cv2.FONT_HERSHEY_SIMPLEX  # Jenis font untuk teks
font_scale = 0.5  # Skala font untuk teks

# Path video yang akan diproses
video_path = "video3.mp4"
model_path = "models/yolov8n.pt"

# Buka video
cap = cv2.VideoCapture(video_path)
# Inisialisasi model YOLO dengan file weight yang telah dilatih sebelumnya
model = YOLO(model_path)

# Ukuran frame video
width = 1280
height = 720

# Inisialisasi objek untuk menyimpan video hasil pemrosesan
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))

# Id objek kendaraan yang ingin dilacak berdasarkan kelas di file coco-classes.txt
vehicle_ids = [2, 3, 5, 7]
# Dictionary untuk menyimpan sejarah pergerakan setiap kendaraan yang terdeteksi
track_history = defaultdict(lambda: [])

up = {}  # Dictionary untuk kendaraan yang melewati garis atas
down = {}  # Dictionary untuk kendaraan yang melewati garis bawah
threshold = 400  # Ambang batas garis pemisah kendaraan

# Fungsi untuk mengambil titik tengah dari bounding box objek
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Background subtraction menggunakan MOG2
subtracao = cv2.createBackgroundSubtractorMOG2()

# Loop utama untuk membaca setiap frame dari video
while True:
    ret, frame = cap.read()  # Membaca frame dari video
    if ret == False:  # Keluar dari loop jika tidak ada frame yang dapat dibaca
        break

    try:
        frame = imutils.resize(frame, width = 1280, height = 720)  # Mengubah ukuran frame ke 1280x720
        frame_color = frame.copy()  # Salin frame ke mode warna untuk pengolahan dan penggambaran
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi frame ke citra grayscale
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # Konversi kembali ke citra BGR untuk tampilan grayscale
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi ke citra grayscale untuk mode black and white

        # Deteksi objek menggunakan model YOLO
        results = model.track(frame_color, persist=True, verbose=False)[0]
        bboxes = np.array(results.boxes.data.tolist(), dtype="int")  # Koordinat bounding box objek yang terdeteksi

        # Gambar garis pembatas untuk menghitung jumlah kendaraan yang melewati garis
        cv2.line(frame_color, (0, threshold), (1280, threshold), color, thickness)
        cv2.putText(frame_color, "Pembatas Jalan", (620, 445), font, 0.7, color_red, thickness)

        # Loop untuk setiap objek yang terdeteksi
        for box in bboxes:
            x1, y1, x2, y2, track_id, score, class_id = box  # Ambil koordinat dan informasi lainnya
            cx = int((x1 + x2) / 2)  # Hitung koordinat x pusat objek
            cy = int((y1 + y2) / 2)  # Hitung koordinat y pusat objek
            if class_id in vehicle_ids:  # Periksa apakah objek merupakan kendaraan yang ingin dilacak
                class_name = results.names[int(class_id)].upper()  # Dapatkan nama kelas objek

            track = track_history[track_id]  # Ambil sejarah pergerakan objek berdasarkan ID
            track.append((cx, cy))  # Tambahkan koordinat pusat objek ke dalam sejarah pergerakan
            if len(track) > 20:  # Batasi panjang sejarah pergerakan agar tidak terlalu panjang
                track.pop(0)  # Hapus elemen pertama jika sejarah sudah melebihi batas

            points = np.hstack(track).astype("int32").reshape(-1, 1, 2)  # Konversi sejarah pergerakan ke format yang sesuai untuk penggambaran
            cv2.polylines(frame_color, [points], isClosed=False, color=color, thickness=thickness)  # Gambar garis yang merepresentasikan sejarah pergerakan
            cv2.rectangle(frame_color, (x1, y1), (x2, y2), color, thickness)  # Gambar bounding box objek
            text = "ID: {} {}".format(track_id, class_name)  # Buat teks ID objek dan nama kelasnya
            cv2.putText(frame_color, text, (x1, y1 - 5), font, font_scale, color, thickness)  # Tampilkan teks di atas objek

            if cy > threshold - 5 and cy < threshold + 5 and cx < 670:  # Periksa apakah objek melewati garis atas
                down[track_id] = x1, y1, x2, y2  # Simpan informasi objek yang melewati garis atas

            if cy > threshold - 5 and cy < threshold + 5 and cx > 670:  # Periksa apakah objek melewati garis bawah
                up[track_id] = x1, y1, x2, y2  # Simpan informasi objek yang melewati garis bawah

        up_text = "Kanan:{}".format(len(list(up.keys())))  # Buat teks jumlah kendaraan yang melewati garis atas
        down_text = "Kiri:{}".format(len(list(down.keys())))  # Buat teks jumlah kendaraan yang melewati garis bawah

        cv2.putText(frame_color, up_text, (1150, threshold - 5), font, 0.8, color_red, thickness)  # Tampilkan teks jumlah kendaraan yang melewati garis atas
        cv2.putText(frame_color, down_text, (0, threshold - 5), font, 0.8, color_red, thickness)  # Tampilkan teks jumlah kendaraan yang melewati garis bawah

        # Background subtraction dan deteksi kontur
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi frame ke citra grayscale
        blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Reduksi noise menggunakan Gaussian Blur
        img_sub = subtracao.apply(blur)  # Background subtraction
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Dilasi untuk meningkatkan ketebalan objek
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel untuk operasi morfologi
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Operasi closing untuk mengisi lubang kecil pada objek
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  # Operasi closing tambahan
        contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Deteksi kontur objek

        writer.write(frame_color)  # Menyimpan frame hasil pemrosesan
        # Menampilkan gambar
        cv2.imshow("Warna", frame_color)  # Tampilkan mode warna
        cv2.imshow("Grayscale", frame_gray)  # Tampilkan mode grayscale
        cv2.imshow("Detectar", dilatada)  # Tampilkan mode Detectar dilatada
        if cv2.waitKey(10) & 0xFF == ord("q"):  # Keluar saat tombol q ditekan
            break

    except Exception as e:
        print("Terjadi kesalahan:", str(e))  # Tangkap dan tampilkan kesalahan yang terjadi
        continue  # Lanjutkan ke iterasi berikutnya

cap.release()  # Bebaskan sumber daya setelah selesai pemrosesan video
writer.release()  # Tutup objek writer
cv2.destroyAllWindows()  # Tutup semua jendela yang dibuka oleh OpenCV

print("[INFO]..Video berhasil diproses/disimpan!")  # Tampilkan pesan ketika pemrosesan selesai
