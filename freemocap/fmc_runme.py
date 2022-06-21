from aniposelib.cameras import CameraGroup

from freemocap.fmc_startup import startup, startupGUI
from freemocap.webcam import camera_settings, timesync

from pathlib import Path
import os
import subprocess
import time
from aniposelib.boards import CharucoBoard

import numpy as np
from scipy.signal import savgol_filter

import cv2

# Rich stuff
from rich import print
from rich.console import Console

console = Console()
from rich.markdown import Markdown
from rich.traceback import install

install(show_locals=False)
from rich import inspect
from rich.padding import Padding

from freemocap import (
    recordingconfig,
    runcams,
    calibrate,
    fmc_mediapipe,
    fmc_openpose,
    fmc_deeplabcut,
    fmc_origin_alignment,
    fmc_mediapipe_annotation,
    reconstruct3D,
    play_skeleton_animation,
    session, parameters,

)

thisStage = 0  # global


# TODO: Replace the below functions with the RunMe options.
def RunMe(sessionID=None,
          stage=1,
          useMediaPipe=True,
          runMediaPipe=True,
          debug=False,
          setDataPath=False,
          userDataPath=None,
          recordVid=True,
          showAnimation=True,
          reconstructionConfidenceThreshold=.5,
          charucoSquareSize=36,
          # mm - ~the size of the squares when printed on 8.5x11" paper based on parameters in ReadMe.md
          calVideoFrameLength=1,
          startFrame=0,
          useBlender=False,
          resetBlenderExe=False,
          get_synced_unix_timestamps=True,
          good_clean_frame_number=0,
          use_saved_calibration=False,
          bundle_adjust_3d_points=False,
          place_skeleton_on_origin=False,
          save_annotated_videos=False,
          ):
    """
    Starts the freemocap pipeline based on either user-input values, or default values. Creates a new session class instance (called sesh)
    based on the specified inputs. Checks for previous user preferences and choices if they exist, or will prompt the user for new choices
    if they don't. Runs the initialization for the system and runs each stage of the pipeline.
    """

    welcome_md = Markdown("""# Welcome to FreeMoCap ✨💀✨ """)
    console.print(welcome_md)

    sesh = session.Session()

    sesh.sessionID = sessionID
    sesh.useMediaPipe = useMediaPipe
    sesh.debug = debug
    sesh.setDataPath = setDataPath
    sesh.userDataPath = userDataPath
    sesh.dataFolderName = recordingconfig.dataFolder
    sesh.startFrame = startFrame
    sesh.get_synced_unix_timestamps = get_synced_unix_timestamps
    sesh.use_saved_calibration = use_saved_calibration

    # %% Startup
    sesh.freemocap_module_path = Path(__file__).parent
    startup.get_user_preferences(sesh)
    sesh.params = parameters.Parameters() #todo: reorg me.

    if stage > 1:
        console.rule()

    # %% Initialization
    if stage == 1:
        camera_settings.initialize(sesh)
    else:
        sesh.initialize(stage)

    # %% Stage Three
    if stage <= 3:
        thisStage = 3
        console.rule(style="color({})".format(thisStage))
        console.rule('Starting Capture Volume Calibration'.upper(), style="color({})".format(thisStage))
        console.rule(style="color({})".format(thisStage))
        console.print(Padding('Loading Anipose-style 6DOF extrinsic parameters. for 3d reconstruction stage', (1, 4)),
                      overflow="fold", justify='center', style="color({})".format(thisStage))
        console.rule('See https://anipose.org for details', style="color({})".format(thisStage))
        console.rule(style="color({})".format(thisStage))

        #the following was used to calibrate based on old stage 2.
        # let's just load the charuco points from our own calibration in anipose.'

        # sesh.cgroup, sesh.mean_charuco_fr_mar_xyz = calibrate.CalibrateCaptureVolume(sesh,board, calVideoFrameLength)
        sesh.cgroup = CameraGroup.load('calibration.toml')
        print('Loading calibration.toml Calibration File Successful!')
    else:
        print('Skipping Calibration')

    # %% Stage Four - uwupose application main loop
    if stage <= 4:
        thisStage = 4
        thisStageColor = 12
        console.rule(style="color({})".format(thisStageColor))
        console.rule('Starting 2D Point Trackers'.upper(), style="color({})".format(thisStageColor))
        stage4_msg = 'This step implements various  computer vision that track the skeleton (and other objects) in the 2d videos, to produce the data that will be combined with the `camera projection matrices` from the calibration stage to produce the estimates of 3d movement. \n \n Each algorithm is different, but most involve using [bold magenta] convolutional neural networks [/bold magenta] trained from labeled videos to produce a 2d probability map of the likelihood that the tracked bodypart/object/feature (e.g. \'LeftElbow\') is in a given location. \n \n The peak of that distrubtion on each frame is recorded as the pixel-location of that item on that frame (e.g. \'LeftElbow(pixel-x, pixel-y, confidence\') where the a confidence value proportional to the underlying probability distribution (i.e. tall peaks in the probablitiy distribution indicate high confidence that the LeftElbow actually is at this pixel-x, pixel-y location) \n \nThis part is crazy future tech sci fi stuff. Seriously unbelievable this kind of thing is possible ✨'
        console.print(Padding(stage4_msg, (1, 4)), overflow="fold", justify='center',
                      style="color({})".format(thisStageColor))
        console.rule(style="color({})".format(thisStageColor))

        if sesh.useMediaPipe:
            console.rule(style="color({})".format(thisStage))
            console.rule('Running MediaPipe skeleton tracker - https://google.github.io/mediapipe',
                         style="color({})".format(thisStage))
            console.rule(style="color({})".format(thisStage))

            # spin up cameras, run inference, open inference UI, spit out
            # data to SteamVR.
            uwu = runcams.UwuRuntime(sesh)
            uwu.InitialiseAndRun()
    else:
        print('Error - wtf? why why why asdfjal111')

    console.rule(style="color({})".format(13))
    console.rule('All Done!'.upper(), style="color({})".format(13))
    console.rule(style="color({})".format(13))
    console.rule('Session Data folder is at: ', style="color({})".format(13))
    console.rule(str(sesh.sessionPath), style="color({})".format(13))
    console.rule(style="color({})".format(13))
    console.rule(style="color({})".format(10))
    console.rule("Thank you for supporting the FreeMoCap Project", style="color({})".format(10))
    console.rule(style="color({})".format(10))
    console.rule(style="color({})".format(13))
    console.print('~✨💀✨~', justify="center")
    console.print('❤️', justify="center")
    console.rule(style="color({})".format(13))
    console.rule(style="color({})".format(14))
