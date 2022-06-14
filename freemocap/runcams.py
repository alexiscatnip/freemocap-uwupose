from collections import deque

import numpy

from freemocap import fmc_mediapipe, reconstruct3D, play_skeleton_animation
from freemocap.webcam import startcamrecording, timesync, videotrim

from pathlib import Path
import time
import pickle
import pandas as pd
import numpy as np
from tkinter import Tk

from src import steamVR_thread

def do_triangulation(session):
    session.mediaPipeSkel_fr_mar_xyz, session.mediaPipeSkel_reprojErr = \
        reconstruct3D.reconstruct3D(session,
                                    session.mediaPipeData_nCams_nImgPts_XYC,
                                    0.5)


def setupmediapipe(session, image_streams):
    fmc_mediapipe.setupMediapipe(session, image_streams)


def RecordCams(session, camInputs, parameterDictionary, rotationInputs):
    """ 
Run the live ingerence    """

    numCams = len(camInputs)  # number of cameras
    numCamRange = range(numCams)  # a range for the number of cameras that we have
    vidNames = []
    camIDs = []
    # unix_camIDs = []

    # %% Starting the thread recordings for each camera
    threads = []
    pixel_point_streams = []
    for n in numCamRange:  # starts recording video, opens threads for each camera
        singleCamID = "Cam{}".format(n + 1)
        camIDs.append(
            singleCamID
        )

        pixel_points_out_queue = deque(maxlen=1)
        pixel_point_streams.append(pixel_points_out_queue)
        camRecordings = startcamrecording.CamRecordingThread(
            session,
            camIDs[n],
            # unix_camIDs[n],
            camInputs[n],
            # vidNames[n],
            # session.rawVidPath,
            None,
            parameterDictionary,
            pixel_points_out_queue
        )
        camRecordings.start()

        threads.append(camRecordings)

    # main loop - inference, triangulation, drawing.
    session.mp_poses = []


    # setupmediapipe(session, image_streams)

    pipe_3d_points = deque(maxlen=10)
    steamVR_output_thread = steamVR_thread.SteamVRThread(
        pipe_3d_points,
        session
    )
    steamVR_output_thread.start()

    time.sleep(1)

    # todo.
    # fig = plt.figure(dpi=200)
    # plt.ion()
    #
    # #start animator
    # play_skeleton_animation.PrintSkeletonAnimation(
    #     session,
    #     startFrame=session.startFrame,
    #     azimuth=-90,
    #     elevation=-81,
    #     useOpenPose=False,
    #     useMediaPipe=True,
    #     useDLC=False,
    #     # recordVid=recordVid,
    #     # showAnimation=showAnimation,
    # )
    from timeit import default_timer as timer
    start = timer()

    stop_flag = False
    while not stop_flag:
        not_all_images_ready = False
        #get points and triangulate

        #guard clause
        for stream in pixel_point_streams:
            if not stream: #not empty
                not_all_images_ready = True
                continue

        if not_all_images_ready:
            time.sleep(0.001)
            continue


        mediaPipeData_nCams_nImgPts_XYC = None
        for stream in pixel_point_streams:
            data = stream.pop()
            wrapped_data = np.expand_dims(data, axis=0)
            if mediaPipeData_nCams_nImgPts_XYC is None:
                mediaPipeData_nCams_nImgPts_XYC = wrapped_data
            else:
                mediaPipeData_nCams_nImgPts_XYC = numpy.concatenate((mediaPipeData_nCams_nImgPts_XYC, wrapped_data),axis = 0)
        session.mediaPipeData_nCams_nImgPts_XYC = mediaPipeData_nCams_nImgPts_XYC
        do_triangulation(session)

        # print_timetaken = False
        # if print_timetaken:
        #     print("Function=%s, Time=%s" % (do_inference.__name__, t2 - t1))
        #     print("Function=%s, Time=%s" % (do_triangulation.__name__, t3 - t2))

        pipe_3d_points.append(session.mediaPipeSkel_fr_mar_xyz)

        # #plot it.
        # plt.pause(0.1)
        # plt.draw()

        # time2sleep = 0.2
        # time.sleep(time2sleep)

        end = timer()
        # print("FPS: " + str(1 / (end - start)))
        start = timer()

    for camRecordings in threads:
        camRecordings.join()  # make sure that one thread ending doesn't immediately end all the others (before they can dump data in a pickle file)

    print("finished inference")

    session.numCams = numCams
    session.session_settings['recording_parameters'].update({'numCams': session.numCams})
    session.timeStampData = None
    session.camIDs = camIDs
    session.numCamRange = numCamRange
    session.vidNames = vidNames


def SyncCams(session, timeStampData, numCamRange, vidNames, camIDs):
    """ 
    Runs the time-syncing process. Accesses saved timestamps, runs the time-syncing GUI, and on user-permission, proceeds to create
    synced videos 
    """
    session.syncedVidPath.mkdir(exist_ok=True)

    # start the timesync process
    frameTable, timeTable, unix_synced_timeTable, frameRate, resultsTable, plots = timesync.TimeSync(session,
                                                                                                     timeStampData,
                                                                                                     numCamRange,
                                                                                                     camIDs)

    # this message shows you your percentages and asks if you would like to continue or not. shuts down the program if no
    root = Tk()
    proceed = timesync.proceedGUI(
        root, resultsTable, plots
    )  # create a GUI instance called proceed
    root.mainloop()

    if session.get_synced_unix_timestamps == True:
        unix_synced_timestamps_csvName = 'unix_synced_timestamps.csv'
        unix_synced_timestamps_csvPath = session.sessionPath / unix_synced_timestamps_csvName
        unix_synced_timeTable.to_csv(unix_synced_timestamps_csvPath)

    if proceed.proceed == True:
        print()
        print('Starting editing')
        videotrim.VideoTrim(session, vidNames, frameTable, session.parameterDictionary, session.rotationInputs,
                            numCamRange)
        session.session_settings['recording_parameters'].update({'numFrames': session.numFrames})
        # videotrim.createCalibrationVideos(session,60,parameterDictionary)
        print('all done')
