# USAGE
# python MusicTherapy_w1.py --ip 127.0.0.1 --port 5000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import EmotionFinder
import moodFeedBack

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__, static_folder="")

# initialize the video stream and hardcoded the video source to 0
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	fakeUrl = 'myPicture.jpg?' + str(datetime.datetime.now().microsecond)
	print(fakeUrl)
	#return render_template("index.html", user_image='myPicture.jpg')
	return render_template("index.html", user_image=fakeUrl)

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=500)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# gray = cv2.GaussianBlur(gray, (7, 7), 0)

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

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
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

	# use CascadeClassifier to find face and scale it
	facecapture = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	findface = facecapture.detectMultiScale(design, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30),
											flags=cv2.CASCADE_SCALE_IMAGE)
	for (x, y, w, h) in findface:
		cv2.rectangle(design, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
	# cv2.imshow(self.window.title(), design)
	# override the original picture taken with the face marked in square
	cv2.imwrite(pictureTaken, cv2.cvtColor(design, cv2.IMREAD_GRAYSCALE))
	# this will cause the browser not to cache for new picture
	fakeUrl = 'myPicture.jpg?' + str(datetime.datetime.now().microsecond)
	print(fakeUrl)
	return render_template("index.html", user_image=fakeUrl)
@app.route("/check_mood")
def check_mood():
	print('Check Mood in progress')
	mood2 = EmotionFinder.main() #uses EmotionFinder.py func to check mood
	print('mood =', mood2)
	return render_template("index.html") #returns html

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
	app.run(host=args["ip"], port=args["port"], debug=False, threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()