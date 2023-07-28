import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import pickle
import tensorflow as tf
import FacenetModules
from sklearn.preprocessing import Normalizer
from PIL import Image
import os

DATADIR = r'Datasets/familyDataset/'
promptMessage = "PRESS 'S' TO TAKE A SNAPSHOT"


#  purpose: method for extracting a face from an image
def getFaceFromImage(image, detector):

    # convert the PIL image to an array of pixels using numpy
    pixels = np.asarray(image)

    # detect the faces in the image
    # results is a list of bounding box, where each bounding box defines a lower-left corner of a bounding box along
    # with the width and height of the box
    results = detector.detect_faces(pixels)

    # determine pixel coordinates of bounding box assuming there is only one face in the image.

    if len(results) < 1:  # if there is no face detected then throw an error
        raise ValueError("No face detected")
    elif len(results) > 1:  # if there is more than one face throw an error
        raise ValueError("More than one face detected")

    # get the bounding box from the first face
    leftCoorX, bottomCoorY, boxWidth, boxHeight = results[0]['box']
    # sometimes a negative value is given for the coordinates so find the absolute values
    leftCoorX, bottomCoorY = abs(leftCoorX), abs(bottomCoorY)
    rightCoorX, topCoorY = leftCoorX + boxWidth, bottomCoorY + boxHeight

    #  extract the face from the image
    face = pixels[bottomCoorY:topCoorY, leftCoorX:rightCoorX]

    # the model expects square face images with the shape 160x160 as an input
    # Therefore using PIL we are going to resize the image to 160x160
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    return leftCoorX, bottomCoorY, rightCoorX, topCoorY, face_array

#  create a list to hold the names of each person
faceLabelStrings = list()

# purpose: method for running snapshot face detector
def run():

    for faceSubDir in os.listdir(DATADIR + "train/"):# add the name of each sub directory to the list of label names
        faceLabelStrings.append(faceSubDir)

    #  append "newPerson" to the list of labels. Used when the user chooses to train their own person
    faceLabelStrings.append("newPerson")

    errorExists = False  # assume that there is no error when we first start the program
    cap = cv2.VideoCapture(0)  # capture video from camera
    detector = MTCNN()  # load MTCNN model for getting face image
    svmModel = pickle.load(open("classifierModel.pkl", "rb"))  # load classifier model

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            # print('Loading feature extraction model')
            FacenetModules.load_model(r'C:\Users\blue_\PycharmProjects\FaceNetFaceAuth\20180402-114759.pb')

            # set placeholder values for the FaceNet model
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image = np.zeros((1, 160, 160, 3))  # initialise image with all zero's

            while(True):
                ret, displayedFrame = cap.read()  # get the current frame from the camera and use is for displaying
                                                  # the stream to the user
                ret, processedFrame = cap.read()  # get the current frame and use it for processing the face image

                #  put the prompt message to take a picture on the display frame
                cv2.putText(displayedFrame, promptMessage, (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('img1', displayedFrame)  # show the display frame
                if cv2.waitKey(1) & 0xFF == ord('s'):  # when the user does press 's' process the face image
                    while(True):

                        try:
                            # using the getFaceFromImage method get the face from the image
                            # as well as the coordinates that make up the bounding box of the face
                            leftCoorX, bottomCoorY, rightCoorX, topCoorY, faceImage = getFaceFromImage(processedFrame, detector)

                            # faceEmbedding = embeddingCreation(face)
                            faceImage = cv2.cvtColor(faceImage, cv2.COLOR_RGB2BGR)

                            faceImage = faceImage.astype('float32')
                            # standardize pixel values across channels (global)
                            mean = faceImage.mean()
                            standardDev = faceImage.std()
                            faceImage = (faceImage - mean) / standardDev
                            # transform face image into one sample

                            image[0, :, :, :] = faceImage # transform face image into one sample
                            # input face image into the model
                            feed_dict = {images_placeholder: image, phase_train_placeholder: False}

                            faceEmbedding = sess.run(embeddings, feed_dict=feed_dict) # get output face embedding
                            faceEmbedding = np.squeeze(faceEmbedding, axis=0) # reduce the dimensionality

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

                            if probability > 80.0:# if the probability of the prediction is over 60 than display the prediction
                                # output predicted person to terminal and the probability of each class
                                print('Predicted: %s' % (predictedPerson))
                                print(yhat_prob)
                                authorizedMessage = "AUTHORIZED PERSONNEL"  # construct authorized message
                                text = "{}: {}".format(predictedPerson, probabilityString)  # construct String to hold name of
                                                                                            # predicted person and the probability
                                                                                            # it is them
                                y = bottomCoorY - 10  # set the y coordinate at to where the text above the rectangle should sit
                                # construct rectangle around the users face
                                cv2.rectangle(processedFrame, (leftCoorX, bottomCoorY), (rightCoorX, topCoorY), (0, 200, 0), 2)
                                # construct text at the top of the rectangle for displaying label and probability
                                cv2.putText(processedFrame, text, (leftCoorX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                                # construct text at the bottom of the frame displaying the authorized message
                                cv2.putText(processedFrame, authorizedMessage, (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 200, 0), 2)
                            else:   # if the probability is less than or equal to 60 than the person is considered unknown
                                print('Predicted: Unknown')
                                # construct messages
                                unknownMessage = "UNKNOWN"
                                unauthorizedMessage = "UNAUTHORIZED PERSONNEL"
                                print(yhat_prob)  # still print probabilities to terminal
                                y = bottomCoorY - 10 # set y coordinate as to where the text should sit above the rectangle
                                # construct rectangle and texts for the frame
                                cv2.rectangle(processedFrame, (leftCoorX, bottomCoorY), (rightCoorX, topCoorY), (0, 0, 255), 2)
                                # set unknown person message above the rectangle
                                cv2.putText(processedFrame, unknownMessage, (leftCoorX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 0, 255), 2)
                                # set the unauthorized message at the bottom of the screen
                                cv2.putText(processedFrame, unauthorizedMessage, (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 0, 255), 2)


                        except ValueError as error:  # catch any error that occurred while predicting faces
                            errorExists = True  # if an error is caught then set boolean to true so later we can display message
                            errorMessage = "Error: " + repr(error)  # Construct error String based on the error that occured
                            print("Error: " + repr(error))  # print error to terminal

                        while(True):
                            returnMessage = "PRESS 'Q' TO RETURN TO THE MENU"

                            if(errorExists == True):
                                cv2.putText(processedFrame, errorMessage, (80, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 0, 255), 2)

                            cv2.putText(processedFrame, returnMessage, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 0, 255), 2)

                            cv2.imshow('img1', processedFrame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        break
                    break

    cv2.destroyAllWindows()
    cap.release



