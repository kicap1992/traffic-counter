from quart import Quart, render_template, Response, request,jsonify,send_from_directory
import cv2
import numpy as np
import aiomysql
from dotenv import load_dotenv
import os
import pandas as pd
import asyncio
import time

app = Quart(__name__, static_folder='assets')

video_list = []

cap = None

videonya = None
jumlah_kenderaan = 0
kenderaan_kiri = 0
kenderaan_kanan = 0
kenderaan_sekarang = 0
selesainya= False
total_kenderaan_sekarang = 0

MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DB = os.getenv('MYSQL_DB')

async def get_db_connection():
    return await aiomysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        loop=asyncio.get_running_loop()
    )



def hitung_kepadatan_kendaraan(jumlah_kendaraan, panjang_jalan):

    if panjang_jalan <= 0:
        raise ValueError("Panjang jalan harus lebih besar dari nol.")
    kepadatan = jumlah_kendaraan / panjang_jalan
    return kepadatan

async def insert_data(nama, waktu,waktu_sekarang ,kenderaan_kiri, kenderaan_kanan,status):
    global total_kenderaan_sekarang
    # get the datetime
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = await get_db_connection()
    async with conn.cursor() as cursor:
        # check if data already exists
        sql = "SELECT * FROM tb_data WHERE nama = %s"
        await cursor.execute(sql, (nama,))
        result = await cursor.fetchone()
        if result:
            # print(waktu_sekarang)
            # update existing data
            # rount the waktu_sekarang
            # rounded_waktu_sekarang = round(float(waktu_sekarang))
            # if (rounded_waktu_sekarang == 0):
            #     rounded_waktu_sekarang = 1
            # jumlah_kenderaan = int(kenderaan_kiri) + int(kenderaan_kanan)
            # jumlah_kenderaan_per_menit = jumlah_kenderaan / rounded_waktu_sekarang * 60
            # kepadatan = ""
            # if(jumlah_kenderaan_per_menit < 20):
            #     kepadatan = "Kepadatan Sepi"
            # elif(jumlah_kenderaan_per_menit < 40 and jumlah_kenderaan_per_menit >= 20):
            #     kepadatan = "Kepadatan Sedang"
            # elif(jumlah_kenderaan_per_menit >= 40):
            #     kepadatan = "Kepadatan Tinggi"

            # kepadatan= "Kepadatan Sepi"

            if total_kenderaan_sekarang <= 2:
                kepadatan = "Kepadatan Sepi"
            elif total_kenderaan_sekarang > 2 and total_kenderaan_sekarang <= 4:
                kepadatan = "Kepadatan Sedang"
            else:
                kepadatan = "Kepadatan Tinggi"


            sql = "UPDATE tb_data SET waktu = %s, waktu_sekarang = %s, kenderaan_kiri = %s, kenderaan_kanan = %s , updated_at = %s , status = %s , kepadatan = %s WHERE nama = %s"
            await cursor.execute(sql, (waktu, waktu_sekarang, kenderaan_kiri, kenderaan_kanan, now, status, kepadatan, nama))
        else:
            # insert new data
            sql = "INSERT INTO tb_data (nama, waktu, waktu_sekarang, kenderaan_kiri, kenderaan_kanan) VALUES (%s,  %s, %s, %s, %s)"
            await cursor.execute(sql, (nama, waktu, waktu_sekarang, kenderaan_kiri, kenderaan_kanan))
        await conn.commit()
    conn.close()




async def generate_frames2(video, threshold,stat):
    global jumlah_kenderaan
    global kenderaan_kiri
    global kenderaan_kanan
    global cap,selesainya, kenderaan_sekarang , total_kenderaan_sekarang

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
            lineypos = 90
            # cv2.line(image, (0, lineypos), (width, lineypos), (255, 0, 0), 5)

            # line y position created to count contours
            lineypos2 = 150
            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 255, 0), 5)

            cv2.line(image, (0, 225), (width, 225), (255, 255, 255), 2)
            cv2.line(image, (0, lineypos), (width, lineypos), (255, 255, 255), 2)

            # min area for contours in case a bunch of small noise contours are created
            minarea = 200

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

            # print("total centroids: " + str(len(cxx)))
            total_kenderaan_sekarang = len(cxx)
            


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

            kenderaan_sekarang = currentcars

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

            # insert data to database here using asyncio
            await insert_data(video, str(round(frames_count / fps, 2)), str(round(framenumber / fps, 2)), kenderaan_kiri, kenderaan_kanan, "Belum Selesai")

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
            
            
            frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
            cv2.imshow("Input", frame)
            cv2.moveWindow("Input", 0, 0)

            cv2.imshow("gray", gray)
            cv2.moveWindow("gray", int(width * ratio), 0)

            cv2.imshow("closing", bins)
            cv2.moveWindow("closing", width, 0)

            cv2.imshow("Output", image)
            cv2.moveWindow("Output", width, int(height * ratio))
            

         
            framenumber = framenumber + 1

            k = cv2.waitKey(int(1000/fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            if k == 27:
                await insert_data(video, str(round(frames_count / fps, 2)), str(round(framenumber / fps, 2)),kenderaan_kiri, kenderaan_kanan, "Belum Selesai")
                cap.release()
                cv2.destroyAllWindows()
                break

        else:  # if video is finished then break loop
            await insert_data(video, str(round(frames_count / fps, 2)),str(round(framenumber / fps, 2)), kenderaan_kiri, kenderaan_kanan, "Selesai")
            selesainya = True
            cap.release()
            cv2.destroyAllWindows()
            break
    
    selesainya = True
    cap.release()
    cv2.destroyAllWindows()



async def update_video_list():
    global video_list
    # add "video/" to the video_list and only take video extensions
    video_list =  [f"video/{f}" for f in os.listdir("video") if f.endswith(".mp4")]

@app.route('/')
async def index():
    global videonya,selesainya
    selesainya = False
    if (cap != None):
        cap.release()
        cv2.destroyAllWindows()
    await update_video_list()
    print("video_list:", video_list)
    video =  request.args.get('video', 'video/video.mp4')
    videonya = video
    the_threshold =  request.args.get('threshold', 450)
    minimal_kepadatan =  request.args.get('minimal_kepadatan', 5)
    threshold =   int(the_threshold)

    try:
        query_select = "SELECT * FROM tb_data WHERE nama = %s "

        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(query_select, (video,))
            result = await cursor.fetchall()
            conn.close()
            
            if len(result) == 0:
                return await render_template('index2.html', video=video, threshold=threshold, video_list=video_list, stat="Belum Ada Data", selesainya=selesainya, minimal_kepadatan=minimal_kepadatan)
            else :
                print(result[0])
                return await render_template('index2.html', video=video, threshold=threshold, video_list=video_list, stat=result[0], selesainya=selesainya, minimal_kepadatan=minimal_kepadatan)

        
    except Exception as e:
        return jsonify({'message': 'Failed to generate frames!', 'error': str(e)}), 500

    # Pass the video file path and threshold value to the template
    

async def video_feed():
    # Get the video file path, threshold value, and state from the URL parameters
    video =  request.args.get('video')
    the_threshold =  request.args.get('threshold', 450)
    threshold = int(the_threshold)
    stat =  request.args.get('stat', 'color')  # Default to 'color' if state is not specified
    # Return the response with the generator function
    print("ini semua variable:", video, threshold, stat)
    await generate_frames2(video, threshold, stat)
    # return  Response( generate_frames2(video, threshold, stat), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_list')
async def video_list():
    await update_video_list()
    return await render_template('video_list.html', video_list=video_list)

@app.route('/videos/<path:video>')
async def video(video):
    return send_from_directory('', video)

# Add route for the video feed
app.add_url_rule('/video_feed', 'video_feed', video_feed)

@app.route('/check_jumlah_kenderaan', methods=['GET'])
async def check_jumlah_kenderaan():
    global jumlah_kenderaan , kenderaan_kiri, kenderaan_kanan ,videonya ,selesainya , kenderaan_sekarang,total_kenderaan_sekarang
    if (videonya != None):
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            sql = "SELECT * FROM tb_data WHERE nama = %s"
            await cursor.execute(sql, (videonya,))
            result = await cursor.fetchall()
            conn.close()
            if len(result) == 0:
                return jsonify({'jumlah_kenderaan': 0, 'kenderaan_kiri': 0, 'kenderaan_kanan': 0})
            else:
                # print(result[0])
                jumlah_kenderaan = result[0][4] + result[0][5]
                kenderaan_kiri = result[0][4]
                kenderaan_kanan = result[0][5]
                waktu_sekarang = result[0][3]
                kepadatan = result[0][7]
                return jsonify({'jumlah_kenderaan': jumlah_kenderaan, 'kenderaan_kiri': kenderaan_kiri, 'kenderaan_kanan': kenderaan_kanan, 'waktu_sekarang':waktu_sekarang , "selesainya": selesainya , "kenderaan_sekarang": kenderaan_sekarang, "kepadatan":kepadatan , "total_kenderaan_sekarang":total_kenderaan_sekarang})
        
    # return jsonify({'jumlah_kenderaan': jumlah_kenderaan, 'kenderaan_kiri': kenderaan_kiri, 'kenderaan_kanan': kenderaan_kanan})

UPLOAD_FOLDER = 'video'
@app.route('/upload', methods=['POST'])
async def upload_file():
    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': False, 'message': 'No file selected'})
    
    if file:
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({'status': True, 'message': 'File uploaded successfully', 'filename': filename})

if __name__ == "__main__":
    app.run(debug=True)
