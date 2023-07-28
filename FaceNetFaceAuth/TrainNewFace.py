import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import FacenetModules
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

DATADIR = r'Datasets/familyDataset/'
promptMessage = "PRESS 'S' TO TAKE A SNAPSHOT"



def getFaceFromImage(image, detector):

    # convert the PIL image to an array of pixels using numpy
    pixels = np.asarray(image)

    # detect the faces in the image
    # results is a list of bounding box, where each bounding box defines a lower-left corner of a bounding box along
    # with the width and height of the box
    results = detector.detect_faces(pixels)

    # determine pixel coordinates of bounding box assuming there is only one face in the image.

    if len(results) < 1:
        raise ValueError("No face detected")
    elif len(results) > 1:
        raise ValueError("More than one face detected")

    # get the bounding box from the first face
    leftCoorX, bottomCoorY, boxWidth, boxHeight = results[0]['box']
    # sometimes a negative value is given for the coordinates so find the absolute values
    leftCoorX, bottomCoorY = abs(leftCoorX), abs(bottomCoorY)
    rightCoorX, topCoorY = leftCoorX + boxWidth, bottomCoorY + boxHeight

    #extract the face from the image

    face = pixels[bottomCoorY:topCoorY, leftCoorX:rightCoorX]

    # the model expects square face images with the shape 160x160 as an input
    # Therefore using PIL we are going to resize the image to 160x160

    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    return leftCoorX, bottomCoorY, rightCoorX, topCoorY, face_array

faceLabelStrings = list()


def run():
    for faceSubDir in os.listdir(DATADIR + "train/"):
        faceLabelStrings.append(faceSubDir)

    errorExists = False
    cap = cv2.VideoCapture(0)
    detector = MTCNN()

    newFaceEmbeddings = list()
    faceLabels = list()
    numImagesTaken = 0

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            # print('Loading feature extraction model')
            FacenetModules.load_model(r'C:\Users\blue_\PycharmProjects\FaceNetFaceAuth\20180402-114759.pb')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            image = np.zeros((1, 160, 160, 3))

            while(numImagesTaken != 58):
                try:

                    ret, displayedFrame = cap.read()
                    ret, processedFrame = cap.read()

                    photoTakenMessage = "NUMBER OF PHOTO'S TAKEN: " + str(numImagesTaken)

                    cv2.putText(displayedFrame, photoTakenMessage, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                       (0, 0, 255), 2)
                    cv2.putText(displayedFrame, promptMessage, (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if(errorExists == True):
                        cv2.putText(displayedFrame, errorMessage, (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)


                    cv2.imshow('Training New Face', displayedFrame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):

                        leftCoorX, bottomCoorY, rightCoorX, topCoorY, faceImage = getFaceFromImage(processedFrame, detector)
                        # faceEmbedding = embeddingCreation(face)

                        faceImage = faceImage.astype('float32')
                        # standardize pixel values across channels (global)
                        mean = faceImage.mean()
                        standardDev = faceImage.std()
                        faceImage = (faceImage - mean) / standardDev
                        # transform face image into one sample

                        image[0, :, :, :] = faceImage
                        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                        # emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

                        faceEmbedding = sess.run(embeddings, feed_dict=feed_dict)
                        faceEmbedding = np.squeeze(faceEmbedding, axis=0)

                        newFaceEmbeddings.append(faceEmbedding)
                        faceLabels.append("addedPerson")

                        errorExists = False
                        numImagesTaken += 1
                except ValueError as error:
                    errorExists = True
                    errorMessage = "Error: " + repr(error)
                    print("Error: " + repr(error))

            #train new svm
            # load face embedding data set
            faceEmbeddings = np.load('faceEmbeddingsDataset.npz')
            # load the training and test face embeddings along with their corresponding labels
            trainingFaces, trainingLabels, testFaces, testLabels = faceEmbeddings['arr_0'], faceEmbeddings['arr_1'], \
                                                                   faceEmbeddings['arr_2'], faceEmbeddings['arr_3']
            print('Dataset: train=%d, test=%d' % (trainingFaces.shape[0], testFaces.shape[0]))

            #Add new embeddings and the Corresponding labels

            newFaceEmbeddings = np.asarray(newFaceEmbeddings)
            # new face embeddings are added to the numpy array
            trainingFaces = np.append(trainingFaces, newFaceEmbeddings, axis=0)
            # add new face labels to the numpy array of labels
            trainingLabels = np.append(trainingLabels, faceLabels)


            # normalize input vectors so that vector magnitude is 1
            in_encoder = Normalizer(norm='l2')
            trainingFaces = in_encoder.transform(trainingFaces)
            #testFaces = in_encoder.transform(testFaces)

            # convert String target variables for each name to an integer
            out_encoder = LabelEncoder()
            out_encoder.fit(trainingLabels)
            trainingLabels = out_encoder.transform(trainingLabels)
            #testLabels = out_encoder.transform(testLabels)

            # fit a model
            # create the model
            model = SVC(kernel='linear', probability=True)
            # train the model of the face embeddings and face labels
            model.fit(trainingFaces, trainingLabels)

            # make predictions
            # store predictions for each face in an array
            yhat_train = model.predict(trainingFaces)
            yhat_test = model.predict(testFaces)

            # score
            score_train = accuracy_score(trainingLabels, yhat_train)
            score_test = accuracy_score(testLabels, yhat_test)

            # summarize
            print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

            pickle_out = open("classifierModel.pkl", "wb")
            pickle.dump(model, pickle_out)
            pickle_out.close()

    cv2.destroyAllWindows()
    cap.release




