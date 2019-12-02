import cv2
import glob
import numpy as np
import time

# the following are used for detecting and cropping face
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

# emotions = ["neutral", "anger", "fear", "happy", "sadness"]
emotions = ["neutral", "anger", "fear", "happy", "sadness", "surprise"]
fishface = cv2.face.FisherFaceRecognizer_create()
inputPictureTaken = "myPicture.jpg"
inputCroppedFile = "myPictureCropped.jpg"


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    datasetFiles = glob.glob("dataset\\%s\\*" % emotion)
    training = datasetFiles[:int(len(datasetFiles) * 1)]  # get 100 percents of the dataset
    return training


def make_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            # print(gray)
            training_labels.append(emotions.index(emotion))
    return training_data, training_labels


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
            # print("face found in file: %s" % f)
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite(inputCroppedFile, out)  # Write file of the cropped image
                pictureTakenHistory = "history//myPicture" + time.strftime("%m%d%Y_%H%M%S") + ".jpg"
                cv2.imwrite(pictureTakenHistory, out)  # Write the cropped image onto history folder
                return "Success"
            except:
                pass  # If error, pass file


def run_recognizer():
    training_data, training_labels = make_sets()
    print("size of training set is:", len(training_data))
    fishface.train(training_data, np.asarray(training_labels))
    image = cv2.imread(inputCroppedFile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pred, conf = fishface.predict(gray)
    print("Prediction confidence=", conf)
    return pred, conf


# Start of program execution
def main():
    result = detect_faces()  # crop and resize input picture
    if (result == "Success"):
        print("Using dataset from:", emotions)
        mood, conf = run_recognizer()
        print(">>Your mood is ", emotions[mood])
        return emotions[mood]
    return "Error"


if __name__ == "__main__":
    x = main()
