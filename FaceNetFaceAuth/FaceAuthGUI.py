#  Author: Alex McLeod
#  Purpose: Class for creating the applications GUI. Displaying the different Menu options as to what the user
#           can do
#  Date Modified: 29/10/2019

from tkinter import *
import LiveStreamFaceAuth
import SnapshotFaceAuth
import TrainNewFace

#  Purpose: The function where the GUI is run from
def run():
    # create the root of the GUI
    root = Tk()

    # initialise the size of the GUI window
    root.geometry("1200x600")

    # initialise the Title of the GUI: Face Authentication
    title = Label(root, text="Face Authentication", bg="red", fg="black", bd=20)
    title.config(font=("Courier", 30))  # set font
    title.place(relx=0.5, rely=0.1, anchor=CENTER)  # set its position

    # initialise the text for author
    authorText = Label(root, text="By Alex McLeod", fg="black", bd=20)
    authorText.config(font=("Courier", 15))  # set font
    authorText.place(relx=0.5, rely=0.25, anchor=CENTER)  # set its position

    # initialise the buttons for the different options available within the menu
    liveStreamButton = Button(text="Begin Video Stream", bg="grey", fg="white", command=LiveStreamFaceAuth.run)
    liveStreamButton.config(font=("Courier", 20))  # set font
    takePictureButton = Button(text="Take Photo", bg="grey", fg="white", command=SnapshotFaceAuth.run)
    takePictureButton.config(font=("Courier", 20))  # set font
    trainNewFaceButton = Button(text="Train New Face", bg="grey", fg="white", command=TrainNewFace.run)
    trainNewFaceButton.config(font=("Courier", 20))  # set font
    exitButton = Button(text="Exit", bg="grey", fg="white", command=root.quit)
    exitButton.config(font=("Courier", 20))  # set font

    # arrange the different buttons in the middle of the window one above the other
    liveStreamButton.place(relx=0.5, rely=0.45, anchor=CENTER)
    takePictureButton.place(relx=0.5, rely=0.6, anchor=CENTER)
    trainNewFaceButton.place(relx=0.5, rely=0.75, anchor=CENTER)
    exitButton.place(relx=0.5, rely=0.90, anchor=CENTER)

    root.mainloop()