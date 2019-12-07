# USAGE
# python MusicTherapy_w3.py --ip 127.0.0.1 --port 5000
import os
# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask, request
from flask import render_template
from flask import jsonify
import threading
import argparse
import datetime
import imutils
import time
import cv2
import random
import json
import CheckMood_v2
import logging
import settings

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
settings.logging.debug("initialize the output frame and a lock used to ensure thread-safe")
outputFrame = None
lock = threading.Lock()
settings.logging.debug('exchanges of the output frames useful for multiple browsers/tabs are viewing the stream')

# initialize a flask object
app = Flask(__name__, static_folder="")

# Read application settings json file to configure settings
def readApplicationSettings():
	with open('ApplicationSettings.json', 'r') as myFile:
		data = json.load(myFile)
	return data['numberofsongs_permood'], data['camera_source'], data['emotions']
numberofsongs_permood, camera_source, emotions = readApplicationSettings()

# initialize the video stream and hardcoded the video source to 0
vs = VideoStream(src=int(camera_source)).start()
time.sleep(2.0)

@app.route("/")
@app.route("/index.html")
def index():
	# return the rendered template
	settings.logging.debug("returning rendered template")
	print(request.json)
	fakeUrl = 'myPicture.jpg?' + str(datetime.datetime.now().microsecond)
	feedback = ""
	songName = ""
	songUrl = ""
	comments = ""
	user_data = feedback, songName, songUrl, comments
	return render_template("index.html", user_image=fakeUrl, user_data=user_data)

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot")
def snapshot():
	print('snapshot: myPicture.jpg')
	print(app.instance_path)
	frame = vs.read()
	pictureTaken = "myPicture.jpg"  # picture filename
	cv2.imwrite(pictureTaken, cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE))
	design = cv2.imread(pictureTaken)  # read the picture taken
	settings.logging.debug("picture has been taken by user and is ready to be analyzed")

	# use CascadeClassifier to find face and scale it
	facecapture = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	findface = facecapture.detectMultiScale(design, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30),
											flags=cv2.CASCADE_SCALE_IMAGE)
	for (x, y, w, h) in findface:
		cv2.rectangle(design, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
	# override the original picture taken with the face marked in square
	cv2.imwrite(pictureTaken, cv2.cvtColor(design, cv2.IMREAD_GRAYSCALE))
	settings.logging.debug('overriding the original picture taken with the face marked in square')
	# this will cause the browser not to cache for new picture
	settings.logging.debug('successfully overridden. browser will not cache for new picture')
	fakeUrl = 'myPicture.jpg?' + str(datetime.datetime.now().microsecond)
	feedback = ""
	songName = ""
	songUrl = ""
	comments = ""
	user_data = feedback, songName, songUrl, comments
	return render_template("index.html", user_image=fakeUrl, user_data=user_data)

@app.route("/check_mood")
def check_mood():
	print('Check Mood in progress')
	action = request.args.get('action')
	mood, score = CheckMood_v2.main()
	print('Mood =', mood)
	fakeUrl = 'myPicture.jpg?' + str(datetime.datetime.now().microsecond)
	random_song = random.randint(0, (int(numberofsongs_permood) - 1))
	print("Random Index # ", random_song)
	if mood == "Error":
		settings.logging.debug('Error detecting face')
		feedback = "Sorry, i can not detect your face. Please retake the picture"
		songName = "NA"
		songUrl = "NA"
		comments = "NA"
	else:
		songName, songUrl, comments = readSongConfiguration(mood, random_song)
		settings.logging.debug('Successfully pull song configuration based on mood')
		feedback = 'Your mood is ' + str(mood) + ' with a confidence score of ' + str(score) + '%'
	user_data = feedback, songName, songUrl, comments
	return render_template("index.html", user_image=fakeUrl, user_data=user_data)

@app.route("/displayApplicationSettings")
def displayApplicationSettings():
	print("Display applications settings...")
	with open('ApplicationSettings.json', 'r') as myFile:
		data = json.load(myFile)
	return jsonify(data)

@app.route("/displaySongConfiguration")
def displaySongConfiguration():
	print("Display Song Configuration...")
	with open('SongConfiguration.json', 'r') as myFile:
		settings.logging.debug("Reading configuration json file to look up for song for the input mood")
		data = json.load(myFile)
	return jsonify(data)

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=600)

		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# acquire the lock, set the output frame, and release the lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


# Read configuration json file to look up for song for the input mood
def readSongConfiguration(in_mood, random_song):
    with open('SongConfiguration.json', 'r') as myFile:
        data = json.load(myFile)
    index = 0

    for moodrec in data:
        if moodrec['mood'] == in_mood:      # search for the mood
            settings.logging.debug("successfully determined a mood based on user's face")
            for songrec in moodrec["songs"]:    # loop through all the song selections for the mood
                index += 1
            # return a random song for the mood
            settings.logging.debug("user has received song recommendation")
            return moodrec["songs"][random_song]["name"], moodrec["songs"][random_song]["location"], moodrec["songs"][random_song]["comments"]



# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
