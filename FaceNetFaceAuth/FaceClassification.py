import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

# load face embedding data set
faceEmbeddings = np.load('faceEmbeddingsDataset.npz')
# load the training and test face embeddings along with their corresponding labels
trainingFaces, trainingLabels, testFaces, testLabels = faceEmbeddings['arr_0'], faceEmbeddings['arr_1'], \
                                                       faceEmbeddings['arr_2'], faceEmbeddings['arr_3']
print('Dataset: train=%d, test=%d' % (trainingFaces.shape[0], testFaces.shape[0]))

# normalize input vectors so that vector magnitude is 1
in_encoder = Normalizer(norm='l2')
trainingFaces = in_encoder.transform(trainingFaces)
testFaces = in_encoder.transform(testFaces)

# convert String target variables for each name to an integer
out_encoder = LabelEncoder()
out_encoder.fit(trainingLabels)
trainingLabels = out_encoder.transform(trainingLabels)
testLabels = out_encoder.transform(testLabels)

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
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# save SVM model
pickle_out = open("classifierModel.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()