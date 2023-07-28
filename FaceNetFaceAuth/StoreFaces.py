import os
# use PIL for image manipulation
from PIL import Image
# import numpy for matrix manipulation
import numpy as np
# import MTCNN for detecting faces from an image
from mtcnn.mtcnn import MTCNN

# Constant that defines where the training and test data can be found
DATADIR = r'Datasets/familyFaceDataset/'

# purpose: Imports the path to a particular image. Using this path it takes an Image object
#          converts it to a numpy array of pixel values. Using the MTCNN() algorithm we are able to detect
#          a face in the image and create a new 160x160 pixel array. This image can then later be used to
#          create an embedding vector for that particular face using the FaceNet model.
def getFaceFromImage(photoPath, detector):

    # use PIL to load an Image object from file path
    image = Image.open(photoPath)

    # use PIL to convert the image to RGB if not coloured
    image = image.convert('RGB')

    # convert the PIL image to an array of pixels using numpy asarray method
    pixels = np.asarray(image)


    # detect the faces in the image
    # result is a list of bounding box, where each bounding box defines a lower-left corner of a bounding box along
    # with the width and height of the box
    # We are assuming that there is only one face in the image so there would only be one bounding box
    results = detector.detect_faces(pixels)

    # if more than one face is detected throw an error message
    if len(results) > 1:
        raise ValueError("For Image: " + photoPath + " More than one face detected")
    # if no face is detected throw an error message
    elif len(results) > 1:
        raise ValueError("For Image: " + photoPath + " No face detected")

    # determine pixel coordinates of bounding box assuming there is only one face in the image.
    leftCoorX, bottomCoorY, boxWidth, boxHeight = results[0]['box']
    # sometimes a negative value is given for the coordinates so find the absolute values
    leftCoorX, bottomCoorY = abs(leftCoorX), abs(bottomCoorY)
    rightCoorX, topCoorY = leftCoorX + boxWidth, bottomCoorY + boxHeight

    # extract the face from the image
    face = pixels[bottomCoorY:topCoorY, leftCoorX:rightCoorX]
    face = np.asarray(face)  # convert to a numpy array

    # the model expects square face images with the shape 160x160 as an input
    # Therefore using PIL we are going to resize the image to 160x160
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)

    # return array containing pixel values of face image
    return face_array

# purpose: imports a path to a directory containing face images and adds these images to a list and exports them
def loadFacesFromDirectory(directoryName):
    faces = list()

    # create Detector using defaults weights for detecting faces from an image
    detector = MTCNN()

    # for each photo in this directory extract face image
    for filename in os.listdir(directoryName):
        path = directoryName + filename  # add filename to the path
        try:
            face = getFaceFromImage(path, detector)
            print(face.shape)  # print the shape of the loaded face image
            faces.append(face)  # add to the list of face images
        except ValueError as error:  # if there is any error when getting face image display error to user
            print("Error: " + repr(error))  # print error message
    return faces


# purpose: imports the path of a directory containing sub directories full of face images
#          it goes through each sub directory and makes a list of faces and a list of labels for each face image
def loadDataSetFromDirectory(trainingDirectoryName):
    # create a list for the training data (daceSet) and its labels (faceLabels)
    faceSet, faceLabels = list(), list()

    # for each family member Alex, Althea and Michelle load the faces images
    for faceSubDir in os.listdir(trainingDirectoryName):
        path = trainingDirectoryName + faceSubDir + '/'

        # skip any files that are in the directory
        if not os.path.isdir(path):
            continue

        faces = loadFacesFromDirectory(path)

        # assign a list the size of how many images there are of the current family member in the directory and
        # for each element assign it the name of that directory
        # essentially creates a list containing the labels for that particular person which is the person's name
        labels = list()

        for faceNum in range(0, len(faces)):  # for each image assign a label with the name of the directory it is in
                                              # which is the name of the person
            labels.append(faceSubDir)  # append the label to the label list

        print(labels)  # print the labels

        # progress summary giving the number of successfully loaded photo's
        print('>loaded %d examples for class: %s' % (len(faces), faceSubDir))

        # store the face images in a list which will contain all face images
        # store the labels for the images in a list containing all labels for all faces
        faceSet.extend(faces)
        faceLabels.extend(labels)
    return np.asarray(faceSet), np.asarray(faceLabels)


# load faces for training and labels for training
trainingFaces, trainingLabels = loadDataSetFromDirectory(DATADIR + "train/")
print(trainingFaces.shape, trainingLabels.shape)

# load faces for testing and labels for testing
testFaces, testLabels = loadDataSetFromDirectory(DATADIR + "test/")
print(testFaces.shape, testLabels.shape)

#  save faces and labels
np.savez_compressed("faceDataset.npz", trainingFaces, trainingLabels, testFaces, testLabels)