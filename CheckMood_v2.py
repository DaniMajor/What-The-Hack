import cv2
import glob
import numpy as np
import time
import json
import array

# the following are used for detecting and cropping face
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

# read mood configuration to determine which mood datasets to include
def readApplicationSettings():
	with open('ApplicationSettings.json', 'r') as myFile:
		data = json.load(myFile)
	return data['numberofsongs_permood'], data['camera_source'], data['emotions']
numberofsongs_permood, camera_source, emotions = readApplicationSettings()
fishface = cv2.face.FisherFaceRecognizer_create()
inputPictureTaken = "myPicture.jpg"
inputCroppedFile = "myPictureCropped.jpg"

# Define function to get file list, randomly shuffle it and split 80/20
def get_files(emotion):
    datasetFiles = glob.glob("dataset\\%s\\*" % emotion)
    training = datasetFiles[:int(len(datasetFiles) * 1)]  # get 100 percents of the dataset
    return training

# Create different sets for each emotion
def make_sets(emotion):
    training_data = []
    training_labels = []
    training = get_files(emotion)
    for item in training:
        image = cv2.imread(item)  # open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        training_data.append(gray)  # append image array to training data list
        training_labels.append(emotions.index(emotion))
    training = get_files("unknown")
    for item in training:
        image = cv2.imread(item)  # open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        training_data.append(gray)  # append image array to training data list
        training_labels.append(emotions.index("unknown"))
    return training_data, training_labels

# Detect face from picture
def detect_faces():
    inputFile = glob.glob(inputPictureTaken)
    print("@detect face: Input file = ", inputFile)

    for f in inputFile:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures = face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""
            return "Error"

        # Cut and save face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite(inputCroppedFile, out)  # Write file of the cropped image
                pictureTakenHistory = "history//myPicture" + time.strftime("%m%d%Y_%H%M%S") + ".jpg"
                cv2.imwrite(pictureTakenHistory, out)  # Write the cropped image onto history folder
                return "Success"
            except:
                pass  # If error, pass file

# evaluate each emotion individually and displays confidence score (lowest determines the mood)
def run_recognizer(emotion):
    training_data, training_labels = make_sets(emotion)
    print(">>Evaluating emotion:", emotion)
    print("Size of training set is:", len(training_data))
    fishface.train(training_data, np.asarray(training_labels))
    image = cv2.imread(inputCroppedFile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pred, conf = fishface.predict(gray)
    print(">> Prediction confidence score: ", conf)
    return pred, conf

# calculate overall accuracy for each mood by percentage
def calculate_result(emotion_conf_array, total_conf):
    mood_score = 0
    mood_winner = np.argmin(emotion_conf_array) # prediction with lowest confidence level wins
    for emotion in emotions:
        if (emotion != "unknown"):
            mood_score = abs(emotion_conf_array[emotions.index(emotion)]*100/total_conf - 100) # calculate mood score
            print("{}: {:.2f} %" .format(emotion, mood_score))
            if (emotions.index(emotion) == mood_winner):
                score_winner = mood_score
    return mood_winner, score_winner

# Start of program execution
def main():
    result = detect_faces()  # crop and resize input picture
    total_conf = 0
    emotion_conf_array = array.array('f')
    if (result == "Success"):
        print("Using dataset from:", emotions)
        print()
        for emotion in emotions:
            if (emotion != "unknown"):
                training = get_files(emotion)
                mood, conf = run_recognizer(emotion)
                total_conf = total_conf + conf
                emotion_conf_array.append(conf)
                print()

        mood, score_winner = calculate_result(emotion_conf_array, total_conf)
        score_winner_rd = round(score_winner,2)

        # mood and confidence score are displayed in mood display box
        print("\n@CheckMood --> Mood Winner: {}, Confidence Score: {:.2f}%".format(emotions[mood], score_winner_rd))
        return emotions[mood], score_winner_rd
    return "Error"


if __name__ == "__main__":
    x = main()
