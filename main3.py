import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture('video/video.mp4')
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# membuat data frame pandas dengan jumlah baris sama dengan jumlah frame
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frame" # frame dalam bahasa indonesia

framenumber = 0  # mencatat frame saat ini
carscrossedup = 0  # mencatat mobil yang melintasi atas
carscrosseddown = 0  # mencatat mobil yang melintasi bawah
carids = []  # list kosong untuk menambah id mobil
caridscrossed = []  # list kosong untuk menambah id mobil yang telah melintasi
totalcars = 0  # mencatat total mobil

fgbg = cv2.createBackgroundSubtractorMOG2()  # membuat subtractor latar belakang MOG2

# informasi untuk memulai menyimpan file video
ret, frame = cap.read()  # impor gambar
ratio = .5  # rasio pengubah ukuran
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # ubah ukuran gambar
width2, height2, channels = image.shape
# video = cv2.VideoWriter('penghitung_kendaraan.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

while True:

    ret, frame = cap.read()  # impor gambar

    if ret:  # jika ada frame lanjutkan kode

        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # ubah ukuran gambar

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # konversi gambar ke warna abu-abu

        fgmask = fgbg.apply(gray)  # menggunakan pengurangan latar belakang MOG2

        # menerapkan tingkat kesulitan pada fgmask untuk mencoba mengisolasi mobil
        # perlu mencoba berbagai pengaturan hingga mobil mudah diidentifikasi
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # membuat kernel untuk operasi morfologi
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # menghapus shadow

        # membuat kontur
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # menggunakan konveks hull untuk membuat poligon kait dengan kontur
        hull = [cv2.convexHull(c) for c in contours]

        # menggambar kontur
        cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

        # garis dibuat untuk menghentikan penghitungan kontur, diperlukan karena mobil jauh menjadi kontur satu
        lineypos = 100
        cv2.line(image, (0, lineypos), (width, lineypos), (255, 0, 0), 5)

        # garis y posisi dibuat untuk menghitung kontur
        lineypos2 = 125
        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 255, 0), 5)

        # area minimal untuk kontur agar tidak dihitung sebagai rumit
        minarea = 400

        # area maksimal untuk kontur, dapat cukup besar untuk bus
        maxarea = 40000

        # vektor untuk x dan y lokasi tengah kontur dalam frame saat ini
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):  # melakukan iterasi pada semua kontur dalam frame saat ini

            # menggunakan hierarki untuk hanya menghitung kontur induk (kontur yang tidak berada dalam kontur lain)
            if hierarchy[0, i, 3] == -1:

                area = cv2.contourArea(contours[i])  # menghitung luas kontur

                if minarea < area < maxarea:  # menggunakan area sebagai garis pembatas untuk kontur

                    # menghitung centroid dari kontur
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    if cy > lineypos:  # menghapus kontur yang berada di atas garis (y dimulai dari atas)

                        # mengambil titik koordinat untuk membuat kotak lingkaran
                        x, y, w, h = cv2.boundingRect(cnt)

                        # membuat kotak lingkaran dari kontur
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Menambahkan teks centroid untuk memverifikasi pada tahap selanjutnya
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3, (0, 0, 255), 1)

                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                       line_type=cv2.LINE_AA)

                        # menambahkan centroid yang telah memenuhi kriteria ke dalam list centroid
                        cxx[i] = cx
                        cyy[i] = cy

        # menghapus nol dalam vector centroid yang tidak dihitung (centroid yang tidak dikirim ke dataframe)
        cxx = cxx[cxx != 0]
        cyy = cyy[cyy != 0]

        # list kosong untuk nanti mencatat indeks centroid yang dikirim ke dataframe
        minx_index2 = []
        miny_index2 = []

        # jumlah maksimum yang diizinkan untuk centroid dalam frame saat ini untuk dikaitkan dengan centroid dari frame sebelumnya
        maxrad = 25

        # bagian berikut mengelola centroid dan mengasignasinya ke id mobil lama atau id mobil baru

        # jika terdapat centroid dalam area yang ditentukan
        if len(cxx):  # jika ada centroid dalam area yang ditentukan

            if not carids:  # jika daftar carids kosong

                for i in range(len(cxx)):  # melakukan loop sebanyak centroid yang ada

                    carids.append(i)  # menambahkan id mobil ke dalam daftar kosong
                    df[str(carids[i])] = ""  # menambahkan kolom ke dalam dataframe berdasarkan id mobil

                    # mengisi nilai centroid pada frame saat ini dan id mobil yang sesuai
                    df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                    totalcars = carids[i] + 1  # menambahkan 1 pada jumlah mobil

            else:  # jika sudah ada id mobil

                dx = np.zeros((len(cxx), len(carids)))  # array untuk menghitung deltanya
                dy = np.zeros((len(cyy), len(carids)))  # array untuk menghitung deltanya

                for i in range(len(cxx)):  # melakukan loop sebanyak centroid yang ada

                    for j in range(len(carids)):  # melakukan loop sebanyak id mobil yang ada

                        # mengambil centroid dari frame sebelumnya untuk id mobil tertentu
                        oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                        # mengambil centroid dari frame sekarang yang tidak selalu sesuai dengan centroid dari frame sebelumnya
                        curcxcy = np.array([cxx[i], cyy[i]])

                        if not oldcxcy:  # jika centroid dari frame sebelumnya kosong karena mobil keluar layar

                            continue  # lanjutkan ke id mobil selanjutnya

                        else:  # hitung deltanya untuk dibandingkan dengan centroid dari frame sekarang

                            dx[i, j] = oldcxcy[0] - curcxcy[0]
                            dy[i, j] = oldcxcy[1] - curcxcy[1]

                for j in range(len(carids)):  # melakukan loop sebanyak id mobil yang ada

                    jumlahjumlah = np.abs(dx[:, j]) + np.abs(dy[:, j])  # menghitung jumlah delta wrt id mobil tertentu

                    # mencari indeks id mobil yang memiliki nilai minimum dan ini indeks yang tepat
                    indeksindextrue = np.argmin(np.abs(jumlahjumlah))
                    minx_index = indeksindextrue
                    miny_index = indeksindextrue

                    # mengambil nilai delta untuk id mobil yang dipilih
                    deltadeltadx = dx[minx_index, j]
                    deltadeltady = dy[miny_index, j]

                    if deltadeltadx == 0 and deltadeltady == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                        # periksa apakah nilai minimum adalah 0 dan periksa apakah semua delta adalah nol karena ini adalah kumpulan kosong
                        # delta dapat berupa nol jika centroid tidak berpindah

                        continue  # lanjutkan ke id mobil selanjutnya

                    else:

                        # jika nilai delta kurang dari radius maksimum maka tambahkan centroid ke id mobil yang sesuai
                        if np.abs(deltadeltadx) < maxrad and np.abs(deltadeltady) < maxrad:

                            # menambahkan centroid ke id mobil yang sudah ada
                            df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                            minx_index2.append(minx_index)  # menambahkan indeks centroid yang sudah ditambahkan ke id mobil lain
                            miny_index2.append(miny_index)

                for i in range(len(cxx)):  # melakukan loop sebanyak centroid yang ada

                    # jika centroid tidak ada dalam list minindex maka mobil baru perlu ditambahkan
                    if i not in minx_index2 and miny_index2:

                        df[str(totalcars)] = ""  # membuat kolom baru untuk mobil baru yang tercatat
                        totalcars = totalcars + 1  # menambahkan jumlah mobil yang tercatat
                        t = totalcars - 1  # t adalah placeholder untuk jumlah mobil
                        carids.append(t)  # menambahkan id mobil ke list id mobil
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # menambahkan centroid ke mobil yang sudah ada

                    elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        # jika centroid saat ini ada namun centroid sebelumnya tidak ada
                        # mobil baru perlu ditambahkan jika minindex2 kosong

                        df[str(totalcars)] = ""  # membuat kolom baru untuk mobil baru yang tercatat
                        totalcars = totalcars + 1  # menambahkan jumlah mobil yang tercatat
                        t = totalcars - 1  # t adalah placeholder untuk jumlah mobil
                        carids.append(t)  # menambahkan id mobil ke list id mobil
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # menambahkan centroid ke mobil yang sudah ada

        # Bagian di bawah menglabel centroid yang ada di layar

        currentcars = 0  # mobil yang ada di layar
        currentcarsindex = []  # indeks id mobil yang ada di layar

        for i in range(len(carids)):  # melakukan loops sebanyak jumlah id mobil

            # memeriksa frame saat ini untuk mengetahui id mobil yang sedang aktif
            # dengan memeriksa adanya centroid pada frame saat ini untuk id mobil tertentu
            if df.at[int(framenumber), str(carids[i])] != '':

                currentcars = currentcars + 1  # menambahkan mobil yang ada di layar
                currentcarsindex.append(i)  # menambahkan id mobil yang ada di layar

        for i in range(currentcars):  # melakukan loops sebanyak jumlah mobil yang ada di layar

            # mengambil centroid untuk id mobil tertentu pada frame saat ini
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

            # mengambil centroid untuk id mobil tertentu pada frame sebelumnya
            oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

            if curcent:  # jika ada centroid pada frame saat ini

                # Teks di layar untuk centroid saat ini
                cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                            (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                               thickness=1, line_type=cv2.LINE_AA)

                # Periksa apakah centroid lama ada
                # Tambahkan kotak radius dari centroid lama ke centroid saat ini untuk visualisasi
                if oldcent:
                    xmulai = oldcent[0] - maxrad
                    ymulai = oldcent[1] - maxrad
                    xakhir = oldcent[0] + maxrad
                    yakhir = oldcent[1] + maxrad
                    cv2.rectangle(image, (int(xmulai), int(ymulai)), (int(xakhir), int(yakhir)), (0, 125, 0), 1)

                    # Periksa apakah centroid lama di bawah garis dan centroid baru di atas garis
                    # Untuk menghitung mobil dan memastikan mobil tidak dihitung dua kali
                    if oldcent[1] >= lineypos2 and curcent[1] <= lineypos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrossedup = carscrossedup + 1
                        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 255), 5)
                        caridscrossed.append(
                            currentcarsindex[i])  # Tambahkan id mobil ke daftar mobil yang dihitung untuk mencegah penghitungan dua kali

                    # Periksa apakah centroid lama di atas garis dan centroid baru di bawah garis
                    # Untuk menghitung mobil dan memastikan mobil tidak dihitung dua kali
                    elif oldcent[1] <= lineypos2 and curcent[1] >= lineypos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:

                        carscrosseddown = carscrosseddown + 1
                        cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 125), 5)
                        caridscrossed.append(currentcarsindex[i])

        # menampilkan jumlah mobil yang melintasi atas
        cv2.putText(image, "Mobil yang Melintasi Atas: " + str(carscrossedup), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),
                    1)

        # menampilkan jumlah mobil yang melintasi bawah
        cv2.putText(image, "Mobil yang Melintasi Bawah: " + str(carscrosseddown), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 255, 255), 1)

        # # menampilkan jumlah total mobil yang terdeteksi
        # cv2.putText(image, "Total Mobil yang Terdeteksi: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
        #             (255, 255, 255), 1)

        # menampilkan frame saat ini dan total frame
        cv2.putText(image, "Frame: " + str(framenumber) + ' dari ' + str(frames_count), (0, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 1)

        # menampilkan waktu yang sudah berlalu dan total waktu
        cv2.putText(image, 'Waktu: ' + str(round(framenumber / fps, 2)) + ' detik dari ' + str(round(frames_count / fps, 2))
                    + ' detik', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        # menampilkan images dan transformasi
        cv2.imshow("Output", image)
        cv2.moveWindow("Output", 0, 0)

        cv2.imshow("gray", gray)
        cv2.moveWindow("gray", int(width * ratio), 0)

        cv2.imshow("closing", closing)
        cv2.moveWindow("closing", width, 0)

        # cv2.imshow("opening", opening)
        # cv2.moveWindow("opening", 0, int(height * ratio))

        # cv2.imshow("dilation", dilation)
        # cv2.moveWindow("dilation", int(width * ratio), int(height * ratio))

        # cv2.imshow("binary", bins)
        # cv2.moveWindow("binary", width, int(height * ratio))


        # adds to framecount
        framenumber = framenumber + 1

        # Menunggu key dari user dalam milidetik, fps adalah frame per detik, dan 0xff adalah binary
        # bahasa indonesia: Menunggu key dari user dalam milidetik
        k = cv2.waitKey(int(1000/fps)) & 0xff 
        if k == 27: # bahasa indonesia: Jika key nya adalah 27 (ESC) maka break loop
            break

    else:  # bahasa indonesia: Jika video selesai maka break loop


        break

cap.release()
cv2.destroyAllWindows()

