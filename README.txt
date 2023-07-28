Author: Alex McLeod
Date Modified: 08/11/2019
Purpose: This is the code to run Face Authentication, tested using PyCharm IDE. The program is run via the Run.py class
         where a GUI will be displayed allowing the user to select authorization via live stream or a snapshot. A user
         can also select the register a new face option allowing a new person to be registered.
Requirements:
    - Python 3.5
    - Pillow 6.1.0
    - matplotlib 3.0.3
    - numpy 1.16.1
    - mtcnn 0.0.9
    - opencv-python 4.1.1.26
    - tensorflow 1.7.0
    - scipy 1.0.0

to activate virtual environment Windows:
venv\Scripts\activate
to deactivate virtual environment Windows:
venv\Scripts\deactivate.bat
to activate virtual environment Linux:
source <venv>/bin/activate
to deactivate virtual environment Linux
deactivate


source code:
Creating Default Training Set (run in this order to train model):

    StoreFaces.py: loads images from Datasets directory and extracts faces from each image.
                   Training and testing face images are stored in a list along with lists for their labels.

    FaceEmbeddingCreation.py: creates embeddings for each face image using FaceNet implemented in tensor flow.
                              Embeddings are then stored.

    FaceClassification.py: takes face embeddings and their corresponding labels and trains an SVM which is then saved.

Program Code:

    Run.py: runs the program

    FaceAuthGUI.py: Constructs Face Authentication menu using tkinter allowing the user to choose a particular option.

    LiveStreamFaceAuth.py: Allows user to perform live face authentication via a live stream.

    SnapshotFaceAuth.py: Allows user to perform authentication via a static snapshot image.

    TrainNewFace.py: Allows a new face to be trained and registered

    FacenetModules.py: contains functions used to load FaceNet model. Functions created by David Sandberg:
		       https://github.com/davidsandberg/facenet	
			 