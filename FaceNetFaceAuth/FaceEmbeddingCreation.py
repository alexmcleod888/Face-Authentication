import numpy as np
import tensorflow as tf
import FacenetModules

# purpose: function for creating an embedding based on an imported face image
def embeddingCreation(faceImage):

    # first we need to scale the pixel values
    faceImage = faceImage.astype('float32')
    # standardize pixel values across channels (global)
    mean = faceImage.mean()
    standardDev = faceImage.std()
    faceImage = (faceImage - mean) / standardDev
    # transform face image into one sample

    # initiate tensorflow graph for getting embeddings
    with tf.Graph().as_default():
        with tf.Session() as sess:

            #Load the model
            FacenetModules.load_model(r'C:\Users\blue_\PycharmProjects\FaceNetFaceAuth\20180402-114759.pb')

            # Set placeholder values for the FaceNet Network
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image = np.zeros((1, 160, 160, 3))  # initialise image with 0's

            image[0, :, :, :] = faceImage  # transform face image into one sample
            feed_dict = {images_placeholder: image, phase_train_placeholder: False} # input image into the FaceNet model
            embedding = sess.run(embeddings, feed_dict=feed_dict)  # get output face embedding
            newEmbedding = np.squeeze(embedding, axis=0)  # reduce dimensionality of array

    return newEmbedding  # return embedding


# purpose: function that imports a list of face images and for each face image calls embeddingCreation
# to get an embedding, storing each embedding in a list of embeddings
def createEmbeddingList(faces):
    print("Calculating Embeddings for face images ...")
    faceEmbeddings = list()  # create list to hold embeddings
    for faceImage in faces:  # for each face image get the embedding
        newEmbedding = embeddingCreation(faceImage)
        faceEmbeddings.append(newEmbedding)  # append to list of embeddings
    faceEmbeddings = np.asarray(faceEmbeddings)

    print(faceEmbeddings.shape)
    return faceEmbeddings  # return a list of embeddings


# load face data set
data = np.load('faceDataset.npz')
# get data
trainingFaces, trainingLabels, testingFaces, testingLabels = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('loaded: ', trainingFaces.shape, trainingLabels.shape, testingFaces.shape, testingLabels.shape)
trainingEmbeddings = createEmbeddingList(trainingFaces)  # get embeddings for training faces
testEmbeddings = createEmbeddingList(testingFaces)  # get embeddings for test faces

# save face embeddings
np.savez_compressed('faceEmbeddingsDataset.npz', trainingEmbeddings, trainingLabels, testEmbeddings, testingLabels)