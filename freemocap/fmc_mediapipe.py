import time
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
from rich.progress import Progress
import cv2
import mediapipe as mp

# from numba import jit

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose= mp.solutions.pose

def setupMediapipe(session, image_streams : List[deque]):
    """obtain on camera image stream width and height."""
    eachCameraResolution = {'Height':[],'Width':[]}
    image_height = image_width = 0
    time2sleep = 0.2

    for camera_int, image_queue in enumerate(image_streams):
        frame_gotten = False
        while frame_gotten is False:
            try:
                print("waiting for frame from camera... " + str(camera_int))
                image = image_queue.pop()  # load first image from video
                frame_gotten = True
            except:
                time.sleep(time2sleep)
                pass

        image_height, image_width, _ = image.shape
        eachCameraResolution["Height"].append(image_height)
        eachCameraResolution["Width"].append(image_width)

        session.eachCameraResolution = eachCameraResolution

def runMediaPipe(image, pose_detector, dummyRun=False):
    """
    Run Mediapipe-pose detector on the camera images
    # Run MediaPipe on synced videos, and save body tracking data to be parsed
    """

    # eachCamerasData = []  # Create an empty list that holds each cameras data
    # mediaPipe_dataList = []  # Create an empty list for mediapipes data
    mediaPipe_data = None
    if image is not None:
        t1 = time.time()
        try:
            mediaPipe_data = pose_detector.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), #Convert the BGR image to RGB before processing
            )  # NOTE: THIS IS WHERE THE MAGIC HAPENS
        except:

            pass
            """ ^ zprevent limbo when:
                RuntimeError: CalculatorGraph::Run() failed in Run: 
                Calculator::Process() for node "poselandmarkbyroicpu__poselandmarksandsegmentat
                ioninverseprojection__InverseMatrixCalculator" failed: ; Inverse matrix cannot 
                be calculated.tors/util/inverse_matrix_calculator.cc:38) 
            """

        t2 = time.time()
        # print("Function=%s, Time=%s" % ("inference call took: ", t2 - t1))
    return mediaPipe_data


def parseMediaPipe(session, mediaPipeData, camIdx):
    """ 
    Parse through saved MediaPipe data, and save out a numpy array of 2D points
        return: numpy array 33*3 (the last column is visiblitiy)
    """
    # numCams = len(session.mediaPipeData)  # Get number of cameras
    # numFrames = len(session.mediaPipeData[0])  # Get number of frames
    # numBodyPoints = len(np.max(session.mediaPipeData[0][:].pose_landmarks.landmark[:]))#Get number of body points
    # numFacePoints = len(np.max(session.mediaPipeData[0][:].face_landmarks.landmark[:]))#Get number of face points
    # numLeftHandPoints = len(np.max(session.mediaPipeData[0][:].left_hand_landmarks.landmark[:]))#Get number of right hand points
    # numRightHandPoints = len(np.max(session.mediaPipeData[0][:].right_hand_landmarks.landmark[:]))#Get number of left hand points
    numBodyPoints = 33
    # numFacePoints = 468
    # numHandPoints = 21

    numTrackedPoints = (
        numBodyPoints
    )  # Get total points

    # Create  array of nans the size of number of cams, frame, points, XYC
    mediaPipeData_nImgPts_XYC = np.empty(
        (int(numTrackedPoints), 3)
    )  # create empty array
    mediaPipeData_nImgPts_XYC[:] = np.NaN  # Fill it with NaNs!

    # for camNum in range(numCams):  # Loop through each camera
        # make empty arrays for thisFrame's data
        # thisFrame_X = np.empty(numTrackedPoints)
        # thisFrame_X[:] = np.nan
        # thisFrame_Y = thisFrame_X.copy()
        # thisFrame_C = thisFrame_X.copy()

    thisFrame_X_body = np.empty(numBodyPoints)
    thisFrame_X_body[:] = np.nan
    thisFrame_Y_body = thisFrame_X_body.copy()
    thisFrame_C_body = thisFrame_X_body.copy()

    fullFrame = True

    try:
        if mediaPipeData is None:
            thisFrame_X_body = np.empty(33)
            thisFrame_Y_body = np.empty(33)
            thisFrame_C_body = np.empty(33)

        # pull out ThisFrame's mediapipe data (`mpData.pose_landmarks.landmark` returns something iterable ¯\_(ツ)_/¯)
        thisFrame_poseDataLandMarks = mediaPipeData\
            .pose_landmarks.landmark  # body ('pose') data
        # stuff body data into pre-allocated nan array
        thisFrame_X_body[:numBodyPoints] = [
            pp.x for pp in thisFrame_poseDataLandMarks
        ]  # PoseX data - Normalized screen coords (in range [0, 1]) - need multiply by image resultion for pixels
        thisFrame_Y_body[:numBodyPoints] = [
            pp.y for pp in thisFrame_poseDataLandMarks
        ]  # PoseY data
        thisFrame_C_body[:numBodyPoints] = [
            pp.visibility for pp in thisFrame_poseDataLandMarks
        ]  #'visibility' is MediaPose's 'confidence' measure in range [0,1]
    except:
        fullFrame = False

    if fullFrame:
        f = 9

    thisFrame_X = thisFrame_X_body
    thisFrame_Y = thisFrame_Y_body
    thisFrame_C = thisFrame_C_body
    # stuff this frame's data into pre-allocated mediaPipeData_.... array
    mediaPipeData_nImgPts_XYC[:, 0] = thisFrame_X
    mediaPipeData_nImgPts_XYC[:, 1] = thisFrame_Y
    mediaPipeData_nImgPts_XYC[:, 2] = thisFrame_C

    # convert from normalized screen coordinates to pixel coordinates
    mediaPipeData_nImgPts_XYC[:, 0] *= session.cgroup.cameras[camIdx].get_size()[0] # get the dimension from the cgroup directly. - that is the one used for triangulation and calculation of reproection error
    mediaPipeData_nImgPts_XYC[:, 1] *= session.cgroup.cameras[camIdx].get_size()[1]

    # mediaPipeData_nCams_nImgPts_XYC[:, 34:, 2] = 1 #sets the non-body point to '1'

    # np.save(session.dataArrayPath / "mediaPipeData_2d.npy", mediaPipeData_nCams_nImgPts_XYC,)

    return mediaPipeData_nImgPts_XYC

# #def parseMediaPipe2(session):

#         #numCams = len(session.mediaPipeData) #Get number of cameras
#         #numFrames = len(session.mediaPipeData[0]) #Get number of frames
#         #numBodyPoints = len(np.max(session.mediaPipeData[0][:].pose_landmarks.landmark[:]))#Get number of body points
#         #numFacePoints = len(np.max(session.mediaPipeData[0][:].face_landmarks.landmark[:]))#Get number of face points        
#         #numLeftHandPoints = len(np.max(session.mediaPipeData[0][:].left_hand_landmarks.landmark[:]))#Get number of right hand points
#         #numRightHandPoints = len(np.max(session.mediaPipeData[0][:].right_hand_landmarks.landmark[:]))#Get number of left hand points
#         numBodyPoints = 33
#         numFacePoints = 468
#         numLeftHandPoints = 21
#         numRightHandPoints = 21

#         numPoints = numBodyPoints+numFacePoints+numLeftHandPoints+numRightHandPoints #Get total points
#         mediaPipe_nCams_nFrames_nImgPts_XYC = np.ndarray((int(session.numCams),int(session.numFrames),int(numPoints),3)) #Create empty array the size of number of cams, frame, points, XYC
#         for nn in range(session.numCams):#Loop through each camera
#             for ii in range(session.numFrames): #Loop through each frame 
#                 for jj in range(numBodyPoints):
#                     if  session.mediaPipeData[0][ii].pose_landmarks is None: #If that point is not detected
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj,0] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj,1] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj,2] = np.nan #Add nan value to that index
#                     else:
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj,0] = session.mediaPipeData[0][ii].pose_landmarks.landmark[jj].x#Take x pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj,1] = session.mediaPipeData[0][ii].pose_landmarks.landmark[jj].y#Take y pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj,2] = session.mediaPipeData[0][ii].pose_landmarks.landmark[jj].visibility#Take visibility(confidence) pos of that point
#                 for jj in range(numFacePoints): 
#                     if  session.mediaPipeData[0][ii].face_landmarks is None: #If that point is not detected
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints,0] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints,1] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints,2] = np.nan #Add nan value to that index
#                     else:
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints,0] = session.mediaPipeData[0][ii].face_landmarks.landmark[jj].x#Take x pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints,1] = session.mediaPipeData[0][ii].face_landmarks.landmark[jj].y#Take y pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints,2] = session.mediaPipeData[0][ii].face_landmarks.landmark[jj].visibility#Take visibility(confidence) pos of that point
#                 for jj in range(numRightHandPoints): 
#                     if  session.mediaPipeData[0][ii].right_hand_landmarks is None: #If that point is not detected
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints,0] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints,1] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints,2] = np.nan #Add nan value to that index
#                     else:
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints,0] = session.mediaPipeData[0][ii].right_hand_landmarks.landmark[jj].x#Take x pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints,1] = session.mediaPipeData[0][ii].right_hand_landmarks.landmark[jj].y#Take y pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints,2] = session.mediaPipeData[0][ii].right_hand_landmarks.landmark[jj].visibility#Take visibility(confidence) pos of that point
#                 for jj in range(numLeftHandPoints): 
#                     if  session.mediaPipeData[0][ii].left_hand_landmarks is None: #If that point is not detected
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints+numRightHandPoints,0] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints+numRightHandPoints,1] = np.nan #Add nan value to that index
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints+numRightHandPoints,2] = np.nan #Add nan value to that index
#                     else:
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints+numRightHandPoints,0] = session.mediaPipeData[0][ii].left_hand_landmarks.landmark[jj].x#Take x pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints+numRightHandPoints,1] = session.mediaPipeData[0][ii].left_hand_landmarks.landmark[jj].y#Take y pos of that point
#                         mediaPipe_nCams_nFrames_nImgPts_XYC[nn,ii,jj+numBodyPoints+numFacePoints+numRightHandPoints,2] = session.mediaPipeData[0][ii].left_hand_landmarks.landmark[jj].visibility#Take visibility(confidence) pos of that point
#             #print(session.mediaPipe_datalist[0].pose_landmarks.landmark[0].x)

#         #path_to_mediapipe_2d = session.dataArrayPath/'mediaPipeData_nCams_nFrames_nImgPts_XY.npy'

#         mediaPipeData_nCams_nFrames_nImgPts_XY =  mediaPipe_nCams_nFrames_nImgPts_XYC[:,:,:,0:2].copy()

#         #mediaPipe_nCams_nFrames_nImgPts_XYC[:, :, :, 0] *= 640
#         #mediaPipe_nCams_nFrames_nImgPts_XYC[:, :, :, 1] *= 480

#         np.save(session.dataArrayPath / "mediaPipe_2d.npy", mediaPipe_nCams_nFrames_nImgPts_XYC,)

#         return mediaPipe_nCams_nFrames_nImgPts_XYC
