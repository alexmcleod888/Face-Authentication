#  Author: Alex McLeod
#  Purpose: Class for running a video live stream which detects the faces trained by the SVM model
#  Date Modified: 29/10/2019

import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
import tensorflow as tf
import FacenetModules
import pickle
import os
from sklearn.preprocessing import Normalizer

DATADIR = r'Datasets/familyFaceDataset/'  # path to photo dataset
returnMessage = "PRESS 'Q' TO RETURN TO THE MENU"  # return message String


#  Purpose: method that runs the video live stream
def run():
    faceLabelStrings = list()  # list for holding the names of the different people to detect
    detector = MTCNN()  # load the MTCNN Network for capturing a face from an image

    for faceSubDir in os.listdir(DATADIR + "train/"):  # append the different names based on the names of the different
                                                       # directories
        faceLabelStrings.append(faceSubDir)



    #  append "newPerson" to the list of labels. Used when the user chooses to train their own person
    faceLabelStrings.append("newPerson")

    cap = cv2.VideoCapture(0)  # capture video from camera
    #detector = MTCNN()  # load the MTCNN Network for capturing a face from an image
    svmModel = pickle.load(open("classifierModel.pkl", "rb"))  # load the SVM model for classifying faces

    # initiate the tensorflow graph for embedding creation
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load FaceNet model for creating embeddings based on an image
            FacenetModules.load_model('20180402-114759.pb')

            # Set placeholder values for the FaceNet Network
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image = np.zeros((1, 160, 160, 3))  # initialise image with zeros

            while (True):
                errorExists = False  # set error Exists to false as there is currently no error

                ret, frame = cap.read()  # capture frame by frame

                try:
                    # using getFaceFromImage locate the face from an image and return an array of pixel values
                    leftCoorX, bottomCoorY, rightCoorX, topCoorY, faceImage = getFaceFromImage(frame, detector)
                    faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)

                    # convert to float value
                    faceImage = faceImage.astype('float32')
                    # standardize pixel values across channels
                    mean = faceImage.mean()
                    standardDev = faceImage.std()
                    faceImage = (faceImage - mean) / standardDev


                    # transform face image into one sample
                    image[0, :, :, :] = faceImage
                    # input image into the FaceNet model
                    feed_dict = {images_placeholder: image, phase_train_placeholder: False}

                    # get output face embedding
                    faceEmbedding = sess.run(embeddings, feed_dict=feed_dict)
                    # reduce dimensionality of array
                    faceEmbedding = np.squeeze(faceEmbedding, axis=0)

                    # Use L2 Norm regularization method to keep coefficients small, reducing complexity of the model
                    in_encoder = Normalizer(norm='l2')
                    faceEmbedding = in_encoder.transform(faceEmbedding.reshape(1, -1))
                    faceEmbedding = faceEmbedding[0]



                    # use SVM model to predict a face
                    yhat_class = svmModel.predict(faceEmbedding.reshape(1, -1))
                    # get the probability of the predicted face
                    yhat_prob = svmModel.predict_proba(faceEmbedding.reshape(1, -1))
                    # get the corresponding name of the label
                    predictedPerson = faceLabelStrings[yhat_class[0]]
                    # round the probability to 2 decimal places
                    probability = round(yhat_prob[0][yhat_class[0]] * 100, 2)
                    # construct a String with the probability
                    probabilityString = str(probability) + "%"


                except ValueError as error:  # catch any error that occurred while predicting faces
                    errorExists = True  # if an error is caught then set boolean to true so later we can display message
                    errorMessage = "Error: " + repr(error)  # Construct error String based on the error that occured
                    print("Error: " + repr(error))  # print error to terminal

                if errorExists == True:  # if an error occured then display errorMessage on frame
                    cv2.putText(frame, errorMessage, (80, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                elif (probability > 80.0):  # if the probability of the prediction is over 60 than display the prediction
                    # output predicted person to terminal and the probability of each class
                    print('Predicted: %s' % (predictedPerson))
                    print(yhat_prob)
                    authorizedMessage = "AUTHORIZED PERSONNEL"  # construct authorized message
                    text = "{}: {}".format(predictedPerson, probabilityString)  # construct String to hold name of
                                                                                # predicted person and the probability
                                                                                # it is them
                    y = bottomCoorY - 10  # set the y coordinate at to where the text above the rectangle should sit
                    # construct rectangle around the users face
                    cv2.rectangle(frame, (leftCoorX, bottomCoorY), (rightCoorX, topCoorY), (0, 200, 0), 2)
                    # construct text at the top of the rectangle for displaying label and probability
                    cv2.putText(frame, text, (leftCoorX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                    # construct text at the bottom of the frame displaying the authorized message
                    cv2.putText(frame, authorizedMessage, (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                else:  # if the probability is less than or equal to 60 than the person is considered unknown
                    print('Predicted: Unknown')


                    # construct messages
                    unknownMessage = "UNKNOWN"
                    unauthorizedMessage = "UNAUTHORIZED PERSONNEL"
                    print(yhat_prob)  # still print probabilities to terminal
                    y = bottomCoorY - 10  # set y coordinate as to where the text should sit above the rectangle
                    # construct rectangle and texts for the frame
                    cv2.rectangle(frame, (leftCoorX, bottomCoorY), (rightCoorX, topCoorY), (0, 0, 255), 2)
                    cv2.putText(frame, unknownMessage, (leftCoorX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, unauthorizedMessage, (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # set return message to let the user know what to do if they want to exit
                cv2.putText(frame, returnMessage, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                # Display the resulting frame
                cv2.imshow('frame', frame)

                # if user presses q then exit live stream and go back to Menu
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# purpose: method for getting face from an image
def getFaceFromImage(image, detector):

    # convert the PIL image to an array of pixels using numpy
    pixels = np.asarray(image)

    # detect the faces in the image
    # results is a list of bounding box, where each bounding box defines a lower-left corner of a bounding box along
    # with the width and height of the box
    results = detector.detect_faces(pixels)

    # determine pixel coordinates of bounding box assuming there is only one face in the image.
    if len(results) < 1:  # if no box's detected throw error message
        raise ValueError("No face detected")
    elif len(results) > 1:  # if more than one box detected than there are multiple faces in front of the camera
                            # hence throw error message
        raise ValueError("More than one face detected")

    # get the bounding box from the first face
    leftCoorX, bottomCoorY, boxWidth, boxHeight = results[0]['box']
    # sometimes a negative value is given for the coordinates so find the absolute values
    leftCoorX, bottomCoorY = abs(leftCoorX), abs(bottomCoorY)
    rightCoorX, topCoorY = leftCoorX + boxWidth, bottomCoorY + boxHeight

    # extract the face from the image
    face = pixels[bottomCoorY:topCoorY, leftCoorX:rightCoorX]

    # the model expects square face images with the shape 160x160 as an input
    # Therefore using PIL we are going to resize the image to 160x160
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    return leftCoorX, bottomCoorY, rightCoorX, topCoorY, face_array