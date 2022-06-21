from collections import deque
from pathlib import Path
from ruamel.yaml import YAML

from freemocap import recordingconfig
from freemocap.webcam import timesync

import cv2


class Session:  # self like "recording self"
    def __init__(self):
        self.numCams = None
        self.numFrames = None
        self.numTrackedPoints = None

        self.basePath = Path.cwd()

        self.cam_inputs = None
        self.rotationInputs = None

        self.cgroup = None

        self.skel_fr_mar_xyz = None  # skeleton points xyz.

        self.useMediaPipe = None
        self.params = None

        self.do_calibration_flag = False

        self.pose_buffer = deque(maxlen=20) # outputted points to steamvr
        self.previous_pose3d = None # known pose3d. Beware that this is raw
        # in camera frame - there is no rotation or tranform applied yet!
        self.mediaPipeSkel_reprojErr = None
        self.mediaPipeSkel_fr_mar_xyz = None

        self.offset  = [0,0,0] # bring the 2 coordinate frames together

        self.calibration_scale_factor = 1000


    def start_session(self, paramDict, rotDict):
        """ 
        When starting a session from stage 1, create all the file paths necessary, and create a session dictionary to save settings.
        Calls the create_session_paths and create_session_dictionary function
        """
        # dataFolderPath = self.basePath/self.dataFolderName

        if self.basePath.stem == self.dataFolderName:  # don't recursively craete 'FreeMoCap_Data' folders!
            dataFolderPath = self.basePath
        else:
            dataFolderPath = self.basePath / self.dataFolderName

        dataFolderPath.mkdir(exist_ok=True)

        self.sessionPath = dataFolderPath / self.sessionID
        self.sessionPath.mkdir(exist_ok=True)

        # create all the session filepaths and settings - create and load them into a dictionary
        self.pathList = self.create_session_paths()
        self.session_settings = self.create_session_dictionary()
        self.create_session_txt(paramDict, rotDict)

        self.save_session()

    def create_session_paths(self):
        """ 
        Creates Path objects to each folder created in a recording self and adds them all to the pathList variable
        folders for these paths are created at the start of each stage where they are needed 
        """
        pathList = []

        self.mediaPipeDataPath = self.sessionPath / 'MediaPipeData'
        pathList.append(self.mediaPipeDataPath)

        self.mediaPipeDataPath = self.sessionPath / 'SteamVRLog'
        pathList.append(self.mediaPipeDataPath)

        self.imOutPath = self.sessionPath / 'imOut'
        pathList.append(self.imOutPath)

        return pathList

    def create_session_dictionary(self):
        """ 
        Creates a dictionary of settings that should be saved into a
        YAML through each session
        """

        recording_parameters = {'numCams': self.numCams,
                                'numFrames': self.numFrames,
                                'numTrackedPoints': self.numTrackedPoints}

        # create a dictionary with the folder names (taken from the
        # recordingconfig.py file) and the folder paths from pathList NOTE
        # AC- 6/30/21 - as of right now, the order of folder names in the
        # recordingconfig.folder_setup list needs to match the the order of
        # folder paths in the pathList
        session_paths = {}

        for foldername, folderpath in zip(recordingconfig.folder_setup,
                                          self.pathList):
            session_paths[foldername] = folderpath

        session_settings_dictionary = {
            'recording_parameters': recording_parameters,
            'session_paths': session_paths}

        return session_settings_dictionary

    def create_session_txt(self, paramDict, rotDict):
        """ 
        Create a text file listing recording parameters
        """
        parameter_text = self.sessionPath / 'sessionSettings.txt'
        text = open(parameter_text, 'w')
        text.write("Session ID = %s\n" % (self.sessionID))
        text.write("%s = %s\n" % ("Parameters", paramDict))
        text.write("%s = %s\n" % ("Rotations", rotDict))
        text.close()

    def save_session(self):
        """ 
        Save the settings needed to rerun the session at a later point into a dictionary, and then save it into a YAML
        """

        session_dictionary_to_save = self.session_settings

        for key, value in session_dictionary_to_save['session_paths'].items():
            session_dictionary_to_save['session_paths'][key] = str(value)

        self.session_yaml_path = self.sessionPath / '{}_config.yaml'.format(
            self.sessionID)

        session_yaml = YAML()

        with open(self.session_yaml_path, 'w') as outfile:
            session_yaml.dump(session_dictionary_to_save, outfile)

    def initialize(self, stage):
        """ 
        When starting a session from Stage 3 onwards, modify the current session class instance with all necessary 
        attributes needed to run the rest of the pipeline 
        """
        # load all session settings back into the session class for this run-through of the code

        self.sessionPath = self.dataFolderPath / self.sessionID
        self.sessionPath.mkdir(exist_ok=True)

        self.session_yaml_path = self.sessionPath / '{}_config.yaml'.format(
            self.sessionID)



        if self.session_yaml_path.is_file():
            self.session_settings = self.load_session()  # if a yaml exists, load it in (this is the case for a webcam recording, or an external recording that's been processed already)
        else:  # if a session yaml doesn't exist, as is the case of an external recording
            self.start_session({}, {})
            # run a check to make sure all the frame numbers are the same
            a_sync_vid_path = list(self.syncedVidPath.glob('*.mp4'))
            frames_per_cam = {}

            for vid in a_sync_vid_path:
                temp_cap = cv2.VideoCapture(str(vid))
                thisCamFrames = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frames_per_cam[str(vid)] = thisCamFrames
                temp_cap.release()
            frame_check = len(list(set(list(
                frames_per_cam.values())))) == 1  # using set() to remove duplicates and check for values count

            if frame_check:
                self.numFrames = int(thisCamFrames)
                self.session_settings['recording_parameters'].update(
                    {'numFrames': self.numFrames})
                self.numCams = len(a_sync_vid_path)
                self.session_settings['recording_parameters'].update(
                    {'numCams': self.numCams})
                self.save_session()
            else:
                raise ValueError(
                    'The number of frames in each video are not equal. Frame counts per video are are: ' + str(
                        frames_per_cam))

    def load_session(self):
        """ 
        Called by the self.initialize function - reads the session yaml and loads the number of cameras and frames, and then 
        creates all necessary session paths
        """

        # session_yaml_path = self.sessionPath/'{}_config.yaml'.format(self.sessionID)
        session_yaml = YAML(typ='safe', pure=True)

        with open(self.session_yaml_path, 'r') as fp:
            session_dictionary_to_load = session_yaml.load(fp)

        session_settings_dictionary = session_dictionary_to_load  # create a copy of the loaded dictionary

        for key, value in session_settings_dictionary['session_paths'].items():
            session_settings_dictionary['session_paths'][key] = Path(value)

        # self.load_session_paths(session_settings_dictionary)
        self.create_session_paths()
        self.numCams = session_settings_dictionary['recording_parameters'][
            'numCams']
        self.numFrames = session_settings_dictionary['recording_parameters'][
            'numFrames']
        self.numTrackedPoints = \
        session_settings_dictionary['recording_parameters']['numTrackedPoints']

        return session_settings_dictionary
        f = 2

    # def load_session_paths(self, session_settings_dict):

    #     session_paths = session_settings_dict['session_paths']

    #     self.rawVidPath = session_paths['RawVideos']
    #     self.syncedVidPath = session_paths['SyncedVideos']
    #     self.calVidPath = session_paths['CalVideos']
    #     self.mediaPipeDataPath = session_paths['MediaPipeData']
    #     self.openPoseDataPath = session_paths['OpenPoseData']
    #     self.dlcDataPath = session_paths['DLCData']
    #     self.imOutPath = session_paths['imOut']
    #     self.dataArrayPath = session_paths['DataArrays']

    def save_user_preferences(self, preferences):
        preferences_yaml = YAML()
        preferences_yaml.dump(preferences, self.preferences_path)

        f = 2
