from pathlib import Path
import datetime
import sys
import pandas as pd
import numpy as np

from freemocap.webcam import recordGUI, camsetup
from freemocap import recordingconfig, fmc_anipose
from rich.console import Console

console = Console()

def initialize(session):
    """ 
    Runs the initialization needed to start a new recording from scratch (Stage 1)
    Create a new sessionID and attempt to load previous recording parameters (camera settings/rotations/save locations), 
    and use default values if previous parameters are not found. Ask user to select preferences using a series of GUIs, runs the Setup option if 
    chosen by the user, and ultimately sets a bool that dictates whether we should proceed to start the actual Stage 1 recording. Also saves parameters
    into the session class, which gets saved into the user_preferences yaml

    """ 
    
    console.rule(style="color({})".format(13))    
    console.rule('Finding available webcams',style="color({})".format(13))
    console.rule(style="color({})".format(13)) 

    # if stage == 1:
    filepath = Path.cwd()

    # %% Stage One Initialization
    session.sessionID = datetime.datetime.now().strftime("sesh_%Y-%m-%d_%H_%M_%S")

    proceedToRecording = False #create this boolean, set it to false, and if the user wants to record
                                #later in the pipeline, it will be set to true

    # run the GUI to get the tasks, the cams chosen, the camera settings, and the session ID
    cam_inputs, task = recordGUI.RunChoiceGUI()
    restartSetup = True

    while restartSetup == True:
        
        try:
            rotation_entry = session.preferences["saved"]["rotations"]
            parameter_entry = session.preferences["saved"]["parameters"]
        except:
            print("Could not load saved parameters, using default parameters")
            rotation_entry = session.preferences['default']['rotations']
            parameter_entry = session.preferences['default']['parameters']
                
        rotDict, paramDict, mediaPipeOverlay = \
            recordGUI.RunParametersGUI(rotation_entry, parameter_entry,
                                       cam_inputs, task)
    
        session.preferences['saved']['rotations'] = rotDict
        session.preferences['saved']['parameters'] = paramDict

        session.save_user_preferences(session.preferences)

        #create a list from the rotation dictionary to be used in running webcams
        rotation_input = list(rotDict.values())

        if task == "setup":
            # run setup processes, and then check if th user wants to proceed to recording
            camsetup.RunSetup(cam_inputs, rotation_input, paramDict,mediaPipeOverlay)
            proceedToRecording, restartSetup = recordGUI.RunProceedtoRecordGUI(
            )
            session.save_user_preferences(session.preferences)
        elif task == "record":
            proceedToRecording = True
            restartSetup = False

    if proceedToRecording:
        # create these session properties to be used later in the pipeline
        session.cam_inputs = cam_inputs
        session.parameterDictionary = paramDict
        session.rotationInputs = rotation_input

        #create a config yaml and text file for this session
        session.start_session(session.parameterDictionary,session.rotationInputs)
        session.session_settings['recording_parameters']['RotationInputs'] = rotDict
        session.session_settings['recording_parameters']['ParameterDict'] = paramDict

        print('Proceeding to Stage One - inference')
    else:
        sys.exit('Recording Canceled')
