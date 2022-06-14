import threading
from collections import deque

import cv2
import time
import pickle
import os
import platform

class CamRecordingThread(threading.Thread):
    def __init__(
        self, session, camID,camInput, beginTime, parameterDictionary,image_out_queue: deque
    ):
        threading.Thread.__init__(self)
        self.camID = camID
        # self.unix_camID = unix_camID
        self.camInput = camInput
        # self.videoName = videoName
        # self.rawVidPath = rawVidPath
        self.beginTime = beginTime
        self.parameterDictionary = parameterDictionary
        self.session = session
        self.image_out_queue = image_out_queue

    def run(self):
        print("Starting " + self.camID)
        self.timeStamps = CamRecording(
            self.session,
            self.camID,
            self.camInput,
            self.parameterDictionary,
            self.image_out_queue
        )

    def getStamps(self):
        return self.timeStamps


# the recording function that each threaded camera object runs
def CamRecording(
    session, camID, camInput, parameterDictionary, image_out_queue
):
    """
    Runs the recording process for each threaded camera instance. Saves a video to the RawVideos folder.
    Saves a timestamp for each frame into a pickle file. Checks for whether the global variable 'flag' 
    has been set to True to end the recording (the first camera to end sets the flag to True, which
    means the rest of the cameras will quit in short order)
    """
    # the flag is triggered when the user shuts down one webcam to shut down the rest.
    # normally I'd try to avoid global variables, but in this case it's
    # necessary, since each webcam runs as it's own object.
    global flag
    flag = False
    camWindowName = "RECORDING - " + str(camID)+' - Press ESC to exit'
    cv2.namedWindow(camWindowName)  # name the preview window for the camera its showing

    if platform.system() == 'Windows':
        cam = cv2.VideoCapture(camInput, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(camInput, cv2.CAP_ANY)


    # if not cam.isOpened():
    #         raise RuntimeError('No camera found at input '+ str(camID))
    # pulling out all the dictionary paramters
    exposure = parameterDictionary.get("exposure")
    resWidth = parameterDictionary.get("resWidth")
    resHeight = parameterDictionary.get("resHeight")
    framerate = parameterDictionary.get("framerate")
    codec = parameterDictionary.get("codec")


  

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resWidth)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resHeight)
    cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # rawPath = filepath/'RawVideos' #creating a RawVideos folder
    # rawPath.mkdir(parents = True, exist_ok = True)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width:", width, "height:", height)

    if cam.isOpened():
        success, frame = cam.read()
    else:
        success = False

    while (success):  # while the camera is opened, record the data until the escape button is hit
        if flag:  # when the flag is triggered, stop recording and dump the data

            break
        success, frame = cam.read()

        donotrender = True
        if not donotrender:
            cv2.imshow(camWindowName, frame)
        # frame_sized = cv2.resize(frame, (resWidth, resHeight))
        # frame_sized = frame

        #pass into Q
        image_out_queue.append(frame)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            flag = True  # set flag to true to shut down all other webcams
            break
    cv2.destroyWindow(camWindowName)
    return None,None


# this is how we sync our time frames, based on our recorded timestamps
