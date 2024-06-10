from flask import Flask, render_template, Response, request,jsonify,send_from_directory
import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import pandas as pd

app = Flask(__name__, static_folder='assets')

video_list = []

# color = (0, 255, 0)
# color_red = (0, 0, 255)
# thickness = 2

# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 0.5

# # Background subtraction menggunakan MOG2
# subtracao = cv2.createBackgroundSubtractorMOG2()

jumlah_kenderaan = 0
kenderaan_kiri = 0
kenderaan_kanan = 0



# Define the generate_frames function with parameters for video, threshold, and state
# def generate_frames(video, threshold, stat):
#     model_path = "models/yolov8n.pt"
#     cap = cv2.VideoCapture(video)
#     model = YOLO(model_path)

#     vehicle_ids = [2, 3, 5, 7]
#     track_history = defaultdict(lambda: [])

#     up = {}
#     down = {}

#     global jumlah_kenderaan
#     global kenderaan_kiri
#     global kenderaan_kanan

#     jumlah_kenderaan = 0
#     kenderaan_kiri = 0
#     kenderaan_kanan = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        

#         try:
#             frame = imutils.resize(frame, width=1280, height=720)
#             # freame_original = frame.copy()
#             frame_color = frame.copy()
#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
#             frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             results = model.track(frame_color, persist=True, verbose=False)[0]
#             bboxes = np.array(results.boxes.data.tolist(), dtype="int")

#             # Gambar garis pembatas untuk menghitung jumlah kendaraan yang melewati garis
#             cv2.line(frame_color, (0, threshold), (1280, threshold), color, thickness)
#             text_position = (620, threshold - 5)  # Adjust the Y coordinate to place the text just above the line
#             cv2.putText(frame_color, "Pembatas Jalan", text_position, font, 0.7, color_red, thickness)
            

#             for box in bboxes:
#                 x1, y1, x2, y2, track_id, score, class_id = box
#                 cx = int((x1 + x2) / 2)
#                 cy = int((y1 + y2) / 2)
#                 if class_id in vehicle_ids:
#                     class_name = results.names[int(class_id)].upper()

#                 track = track_history[track_id]
#                 track.append((cx, cy))
#                 if len(track) > 20:
#                     track.pop(0)

#                 points = np.hstack(track).astype("int32").reshape(-1, 1, 2)
#                 cv2.polylines(frame_color, [points], isClosed=False, color=color, thickness=thickness)
#                 cv2.rectangle(frame_color, (x1, y1), (x2, y2), color, thickness)
#                 text = "ID: {} {}".format(track_id, class_name)
#                 cv2.putText(frame_color, text, (x1, y1 - 5), font, font_scale, color, thickness)

#                 if cy > threshold - 5 and cy < threshold + 5 and cx < 670:
#                     down[track_id] = x1, y1, x2, y2

#                 if cy > threshold - 5 and cy < threshold + 5 and cx > 670:
#                     up[track_id] = x1, y1, x2, y2

#             up_text = "Kanan:{}".format(len(list(up.keys())))
#             down_text = "Kiri:{}".format(len(list(down.keys())))
#             kenderaan_kanan = len(list(up.keys()))
#             kenderaan_kiri = len(list(down.keys()))
#             cv2.putText(frame_color, up_text, (1150, threshold - 5), font, 0.8, color_red, thickness)
#             cv2.putText(frame_color, down_text, (0, threshold - 5), font, 0.8, color_red, thickness)

#             # Background subtraction dan deteksi kontur
#             grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi frame ke citra grayscale
#             blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Reduksi noise menggunakan Gaussian Blur
#             img_sub = subtracao.apply(blur)  # Background subtraction
#             dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Dilasi untuk meningkatkan ketebalan objek
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel untuk operasi morfologi
#             dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Operasi closing untuk mengisi lubang kecil pada objek
#             dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  # Operasi closing tambahan
#             contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Deteksi kontur objek
#             frame_bw = cv2.cvtColor(dilatada, cv2.COLOR_GRAY2BGR)  # Konversi frame grayscale ke BGR

#             if stat == 'color':
#                 frame_to_encode = frame_color
#             elif stat == 'grayscale':
#                 frame_to_encode = frame_gray
#             elif stat == 'original':
#                 frame_to_encode = frame
#             else:  # Assuming 'detectar' state
#                 frame_to_encode = frame_bw

#             _, buffer = cv2.imencode('.jpg', frame_to_encode)
#             frame_bytes = buffer.tobytes()

#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         except Exception as e:
#             print("Terjadi kesalahan:", str(e))
#             continue

#         jumlah_kenderaan = kenderaan_kiri + kenderaan_kanan
            

#     cap.release()


def generate_frames2(video, threshold,stat):
    global jumlah_kenderaan
    global kenderaan_kiri
    global kenderaan_kanan

    jumlah_kenderaan = 0
    kenderaan_kiri = 0
    kenderaan_kanan = 0

    cap = cv2.VideoCapture(video)
    frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = int(width)
    height = int(height)
    print(frames_count, fps, width, height)

    # creates a pandas data frame with the number of rows the same length as frame count
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = "Frames"

    framenumber = 0  # keeps track of current frame
    carscrossedup = 0  # keeps track of cars that crossed up
    carscrosseddown = 0  # keeps track of cars that crossed down
    carids = []  # blank list to add car ids
    caridscrossed = []  # blank list to add car ids that have crossed
    totalcars = 0  # keeps track of total cars

    fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

    # information to start saving a video file
    ret, frame = cap.read()  # import image
    ratio = .5  # resize ratio
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    width2, height2, channels = image.shape
    # video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

    while True:

        ret, frame = cap.read()  # import image

        if ret:  # if there is a frame continue with code

            image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

            fgmask = fgbg.apply(gray)  # uses the background subtraction

            # applies different thresholds to fgmask to try and isolate cars
            # just have to keep playing around with settings until cars are easily identifiable
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

            # creates contours
            contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            # use convex hull to create polygon around contours
            hull = [cv2.convexHull(c) for c in contours]

            # draw contours
            # cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

            # line created to stop counting contours, needed as cars in distance become one big contour
            lineypos = 125
            # cv2.line(image, (0, lineypos), (width, lineypos), (255, 0, 0), 5)

            # line y position created to count contours
            lineypos2 = 150
            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 255, 0), 5)

            # min area for contours in case a bunch of small noise contours are created
            minarea = 175

            # max area for contours, can be quite large for buses
            maxarea = 50000

            # vectors for the x and y locations of contour centroids in current frame
            cxx = np.zeros(len(contours))
            cyy = np.zeros(len(contours))

            for i in range(len(contours)):  # cycles through all contours in current frame

                if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                    area = cv2.contourArea(contours[i])  # area of contour

                    if minarea < area < maxarea:  # area threshold for contour

                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        if cy > lineypos:  # filters out contours that are above line (y starts at top)

                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3, (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                        line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx[i] = cx
                            cyy[i] = cy

            # eliminates zero entries (centroids that were not added)
            cxx = cxx[cxx != 0]
            cyy = cyy[cyy != 0]

            # empty list to later check which centroid indices were added to dataframe
            minx_index2 = []
            miny_index2 = []

            # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame
            maxrad = 25

            # The section below keeps track of the centroids and assigns them to old carids or new carids

            if len(cxx):  # if there are centroids in the specified area

                if not carids:  # if carids is empty

                    for i in range(len(cxx)):  # loops through all centroids

                        carids.append(i)  # adds a car id to the empty list carids
                        df[str(carids[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                        totalcars = carids[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx), len(carids)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy), len(carids)))  # new arrays to calculate deltas

                    for i in range(len(cxx)):  # loops through all centroids

                        for j in range(len(carids)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx[i], cyy[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                                # adds centroid to corresponding previously existing carid
                                df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                                minx_index2.append(minx_index)  # appends all the indices that were added to previous carids
                                miny_index2.append(miny_index)

                    for i in range(len(cxx)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df[str(totalcars)] = ""  # create another column with total cars
                            totalcars = totalcars + 1  # adds another total car the count
                            t = totalcars - 1  # t is a placeholder to total cars
                            carids.append(t)  # append to list of car ids
                            df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df[str(totalcars)] = ""  # create another column with total cars
                            totalcars = totalcars + 1  # adds another total car the count
                            t = totalcars - 1  # t is a placeholder to total cars
                            carids.append(t)  # append to list of car ids
                            df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

            # The section below labels the centroids on screen

            currentcars = 0  # current cars on screen
            currentcarsindex = []  # current cars on screen carid index

            for i in range(len(carids)):  # loops through all carids

                if df.at[int(framenumber), str(carids[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars = currentcars + 1  # adds another to current cars on screen
                    currentcarsindex.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

                if curcent:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                                (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    # cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                                # cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                                thickness=1, line_type=cv2.LINE_AA)

                    if oldcent:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent[0] - maxrad
                        ystart = oldcent[1] - maxrad
                        xwidth = oldcent[0] + maxrad
                        yheight = oldcent[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent[1] >= lineypos2 and curcent[1] <= lineypos2 and carids[
                            currentcarsindex[i]] not in caridscrossed:

                            carscrossedup = carscrossedup + 1
                            kenderaan_kiri = carscrossedup
                            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 255), 5)
                            caridscrossed.append(
                                currentcarsindex[i])  # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet
                        elif oldcent[1] <= lineypos2 and curcent[1] >= lineypos2 and carids[
                            currentcarsindex[i]] not in caridscrossed:

                            carscrosseddown = carscrosseddown + 1
                            kenderaan_kanan = carscrosseddown
                            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 125), 5)
                            caridscrossed.append(currentcarsindex[i])
                    jumlah_kenderaan = carscrossedup + carscrosseddown

            # Top left hand corner on-screen text
            #cv2.rectangle(image, (0, 0), (250, 100), (255, 0, 0), -1)  # background rectangle for on-screen text

            cv2.putText(image, "Kenderaan Sebelah Kiri: " + str(carscrossedup), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255),
                    4)

            cv2.putText(image, "Kenderaan Sebelah Kanan: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (255,255,255), 4)

            # cv2.putText(image, "Total Cars Detected: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
            #             (255,255,255), 1)

            cv2.putText(image, "Frame: " + str(framenumber) + ' dari ' + str(frames_count), (0, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (255,255,255), 1)

            cv2.putText(image, 'Waktu: ' + str(round(framenumber / fps, 2)) + ' detik dari ' + str(round(frames_count / fps, 2))
                    + ' detik', (0, 75), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            
            
            
            

            # displays images and transformations and resize to 1280x720
            # cv2.imshow("countours", image)
            # cv2.moveWindow("countours", 0, 0)
            if stat == 'color':
                # frame_to_encode = frame
                # resize to 1280x720
                frame_to_encode = cv2.resize(image, (1280, 720))


            # cv2.imshow("fgmask", fgmask)
            # cv2.moveWindow("fgmask", int(width * ratio), 0)
            elif stat == 'grayscale':
                # frame_to_encode = gray
                frame_to_encode = cv2.resize(gray, (1280, 720))

            # cv2.imshow("closing", closing)
            # cv2.moveWindow("closing", width, 0)
            elif stat == 'detectar':
                # frame_to_encode = closing
                frame_to_encode = cv2.resize(bins, (1280, 720))
            else :
                # frame_to_encode = opening
                frame_to_encode = cv2.resize(frame, (1280, 720))

            _, buffer = cv2.imencode('.jpg', frame_to_encode)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # cv2.imshow("opening", opening)
            # cv2.moveWindow("opening", 0, int(height * ratio))

            # cv2.imshow("dilation", dilation)
            # cv2.moveWindow("dilation", int(width * ratio), int(height * ratio))

            # cv2.imshow("binary", bins)
            # cv2.moveWindow("binary", width, int(height * ratio))

            # video.write(image)  # save the current image to video file from earlier

            # adds to framecount
            framenumber = framenumber + 1

            k = cv2.waitKey(int(1000/fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            if k == 27:
                break

        else:  # if video is finished then break loop

            break

    cap.release()
    cv2.destroyAllWindows()



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
    return Response(generate_frames2(video, threshold, stat), mimetype='multipart/x-mixed-replace; boundary=frame')


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
