import threading
from collections import deque
from typing import Deque, List

import numpy
import winsound

from freemocap import fmc_mediapipe, reconstruct3D, play_skeleton_animation, \
    inference_gui
from freemocap.parameters import Parameters
from freemocap.session import Session
from freemocap.webcam import startcamrecording, timesync, videotrim

from pathlib import Path
import time
import pickle
import pandas as pd
import numpy as np
from tkinter import Tk

from src import steamVR_thread
from src.helpers import mediapipe33To3dpose, get_rot


def do_triangulation(session : Session, visibility_thresh):
    """
    given C * K * 2D points (obtained from keypoint detection in 2D image),
    perform triangulation to get K*3D points (in extrinsic calibration space.)

    If C (cameracount) == 2,
    there is only one way to do triangulation.

    If C > 2,
    then, there are multiple possible solutions to the triangulation.

    EG. if C == 3,
    then, there are these possible solutions to triangulation:
    1. triangulation(C1,C2,C3)
    2. triangulation(C1,C2)
    3. triangulation(C2,C3)
    4. triangulation(C1,C3)
    thus, we will choose the smallest reprojection error of these 4
    possible solutions.

    """
    session.mediaPipeSkel_fr_mar_xyz, session.mediaPipeSkel_reprojErr, \
    session.mediaPipeSkel_Reprojerr_C_N_2 = \
        reconstruct3D.reconstruct3D(session,
                                    session.mediaPipeData_nCams_nImgPts_XYC,
                                    visibility_thresh)
    assert (session.mediaPipeSkel_fr_mar_xyz is not None), "calibration.toml " \
                                                           "not provided"
    assert (session.mediaPipeSkel_reprojErr is not None), "calibration.toml " \
                                                          "not provided"


def setupmediapipe(session, image_streams):
    fmc_mediapipe.setupMediapipe(session, image_streams)


def combine_pose3d(old_pose3d, new_pose3d):
    """
    add new data gatheres in this frame into the pose from last frame.
    """
    for idx in range(len(new_pose3d)):
        if np.isnan(new_pose3d[idx]).any():
            continue
        else:
            old_pose3d[idx] = new_pose3d[idx]
    return old_pose3d


class UwuRuntime:
    def __init__(self, session):
        self.session = session
        self.params: Parameters = session.params
        self.parameterDictionary = session.parameterDictionary

        self.numCams = len(session.cam_inputs)  # number of cameras
        self.numCamRange = range(
            self.numCams)  # a range for the number of cameras that we have
        self.vidNames = []
        self.camIDs = []

    def InitialiseAndRun(self):
        """
        Run the live inference,
        triangulare 2D -> 3D,
        and send sensor data to the apriltag driver for steamVR
        """
        camInputs = self.session.cam_inputs
        session = self.session
        rotationInputs = self.session.rotationInputs
        parameterDictionary = self.parameterDictionary
        camIDs = self.camIDs

        # %% Starting the capture-and-inference thread for each camera
        threads = []
        pixel_point_streams: List[
            Deque] = []  # shape C*33*(u,v) output of 2D pose estimation from the camera threads
        for n in self.numCamRange:  # starts recording video, opens threads
            # for each camera
            singleCamID = "Cam{}".format(n + 1)
            camIDs.append(
                singleCamID
            )

            pixel_points_stream = deque(maxlen=1)
            pixel_point_streams.append(pixel_points_stream)
            camRecordings = startcamrecording.CamRecordingThread(
                session,
                camIDs[n],
                camInputs[n],
                None,
                parameterDictionary,
                pixel_points_stream
            )
            camRecordings.start()
            threads.append(camRecordings)

        # start steamVR output thread.
        pipe_3d_points_in = deque(maxlen=10)
        pipe_HMD_out = deque(maxlen=1)
        steamVR_output_thread = steamVR_thread.SteamVRThread(
            pipe_3d_points_in,
            pipe_HMD_out,
            session
        )
        steamVR_output_thread.start()

        # start UI thread
        gui_thread = threading.Thread(target=inference_gui.make_inference_gui,
                                      args=(session,),
                                      daemon=True)
        gui_thread.start()

        # wait 10 seconds for camera-inference to become stable.
        # At the same time, the user will hold still to do calibration.
        # winsound.PlaySound('calib_prompter.mp3',
        #                    winsound.SND_FILENAME | winsound.SND_ASYNC)
        # time.sleep(0.3)
        # do_calibration(pipe_3d_points_in, pixel_point_streams, session)

        # from timeit import default_timer as timer
        # start = timer()
        # winsound.PlaySound('calib_complete.mp3',
        #                    winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.run_work_loop(pipe_3d_points_in, pixel_point_streams, session,
                           pipe_HMD_out)

        for camRecordings in threads:
            camRecordings.join()  # make sure that one thread ending doesn't immediately end all the others (before they can dump data in a pickle file)

        print("finished working. will close now.")

    def get_average_from_pose3d_buffer_at_idx(self, pose_buffer: deque, index:
    int):
        """
        get average position of the keypoint 'index' inside of pose_buffer
        todo: find average rotation too.
        """

        position = None
        for pose3d in pose_buffer:
            pose3d_pos = pose3d[0][index]
            if position is None:
                position = pose3d_pos
            else:
                position += pose3d_pos
        # divide by len
        position = position / len(pose_buffer)

        return position

    def do_calib_using_common_child_frame(self, hmd_neck, camera_neck):
        """
        we have an object, whose pos and rot are expressed in 2 coordinate frames.
        we wish to find the transformation between the 2 coordinate frames

        in particular, find the transform that maps points expressed in
        camera-frame,
        into points expressed in hmd-frame.
        """

        # in to-space
        to_pos = hmd_neck[0]
        to_rot = hmd_neck[1]
        # in from-space
        from_pos = camera_neck[0]
        from_rot = camera_neck[1]

        print(str(to_pos))
        print(str(to_rot))
        print(str(from_pos))
        print(str(from_rot))

        return

    def run_work_loop(self, pipe_3d_points_in, pixel_point_streams, session:
    Session, pipe_HMD_out):
        """main triangulation loop. the main 'while-loop' """
        stop_flag = session.params.exit_ready
        while not stop_flag:
            not_all_images_ready = False

            # guard clause - wait until all streams have data.
            for stream in pixel_point_streams:
                if not stream:  # if empty
                    not_all_images_ready = True
                    continue
            if not_all_images_ready:
                time.sleep(0.001)
                continue

            # parse camera-inference output data.
            this_frame_mediaPipeData_nCams_nImgPts_XYC = None
            for stream in pixel_point_streams:
                data = stream.pop()
                mediaPipe_Data = np.expand_dims(data,
                                                axis=0)  # hack: to concatatenate into lists of added outer dimension C, where C is num cameras.
                if this_frame_mediaPipeData_nCams_nImgPts_XYC is None:
                    this_frame_mediaPipeData_nCams_nImgPts_XYC = mediaPipe_Data
                else:
                    this_frame_mediaPipeData_nCams_nImgPts_XYC = numpy.concatenate(
                        (this_frame_mediaPipeData_nCams_nImgPts_XYC,
                         mediaPipe_Data),
                        axis=0)
            session.mediaPipeData_nCams_nImgPts_XYC = \
                this_frame_mediaPipeData_nCams_nImgPts_XYC

            # triangulate (with remove occluded points.)
            visibility_thresh = 0.8
            do_triangulation(session, visibility_thresh)

            # do something if one camera has high repro error, while the
            # other 2 are ok.
            # session.mediaPipeSkel_Reprojerr_C_N_2

            # if reprojection error too high, drop the points.
            repro_error_thresh = 400
            for idx, repro_error in enumerate(session.mediaPipeSkel_reprojErr):
                if repro_error > repro_error_thresh:
                    session.mediaPipeSkel_fr_mar_xyz[idx, :] = np.NaN

            # parse points into *steamVR convention*
            mp_points_3D = session.mediaPipeSkel_fr_mar_xyz
            if mp_points_3D is not None:
                pose3d = mediapipe33To3dpose(mp_points_3D)
                pose3d /= session.calibration_scale_factor

                # flip the points into steamvr coordinate system
                pose3d[:, 0] = -pose3d[:, 0]
                pose3d[:, 1] = -pose3d[:, 1]
            else:
                pose3d = None

            # update previous_pose3d on the new points data.
            if session.previous_pose3d is None:
                session.previous_pose3d = pose3d
            else:
                session.previous_pose3d = combine_pose3d(
                    session.previous_pose3d, pose3d)

            # add rotation and transform offsets to pose3d to send out.
            pose3d = session.previous_pose3d.copy()
            # apply calibration rotations to get into *steamVR coordinateframe*
            # 1. rotate camera-frame to intermediate-frame
            # 2. translate intermediate-frame to VR frame
            for j in range(pose3d.shape[0]):  # apply rotations from sliders
                pose3d[j] = self.params.global_rot_z.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_x.apply(pose3d[j])
                pose3d[j] = self.params.global_rot_y.apply(pose3d[j])
            # apply positional offset to get into *steamVR coordinateframe*
            for j in range(pose3d.shape[0]):
                pose3d[j] = pose3d[j] + self.session.offset

            # send!
            pose3d_rot = get_rot(pose3d)
            pipe_3d_points_in.append([pose3d, pose3d_rot,
                                      session.mediaPipeSkel_reprojErr])

            # push points into past buffer.
            session.pose_buffer.append([pose3d, pose3d_rot])

            if session.do_calibration_flag is True:
                session.do_calibration_flag = False
                print("calibrating....\n")
                # get the HMD pose - lets not bother making a buffer for this
                hmdpos, hmdrot = pipe_HMD_out.popleft()
                # get the pose3d at neck
                camerasFrame_neckpos = \
                    self.get_average_from_pose3d_buffer_at_idx(
                        session.pose_buffer, 7)  # btwn shoulders
                # sync 2 coordinate frames by neck-to-neck.
                neckoffset = hmdrot.apply(
                    session.params.hmd_to_neck_offset)
                hmdpos = hmdpos + neckoffset

                camerasFrame_neckrot = session.pose_buffer[0][1]
                # self.do_calib_using_common_child_frame([hmdpos, hmdrot],
                #                                        [camerasFrame_neckpos,
                #                                         camerasFrame_neckrot])
                self.do_calib_using_feetsies_and_neck([hmdpos, hmdrot])

                session.previous_pose3d  = None

        print("camera thread killing.")

    def do_calib_using_feetsies_and_neck(self, param):
        """
        calibration from original repo.
        assumptions:
        - user is standing up straight.
        - the feetsies are directly below the hips.
             ___
            /o o\
            \_O_/   <--- this is u
            | | |
              |
              |   <--- use hip as the alignment point.
             /\
            /  \   <--- use between-feet as the 2nd alignment point
            they should be vertical (x and z are zero)

        logical sequence:
        - get (z) roll values by taking the vector from left foot to right
        foot,
        and take rotate along the roll axis until the vector is flat
        relative to x-z plane. *The assumption is that the x-z plane of
        to-space is also flat on the ground. *a secondary assumption is that this vector from left foot to right foot is not parallel to either x/z axis
        - repeat above for (x) tilt.
        - using pose3d and hmd rotation, get their relative
        yaw-difference.
        - do scaling using extreme pose3d values.
        - do translation of coordinate frames using 'offset' calculated
        using neck postiions.

        """
        neckhmdpos, headsetrot = param

        ## roll calibaration
        hip_from_origin = self.session.previous_pose3d[6]
        feet_middle_from_origin = (self.session.previous_pose3d[0] +
                                   self.session.previous_pose3d[5]) / 2

        feet_middle_from_hip = feet_middle_from_origin - hip_from_origin
        print(feet_middle_from_hip)
        value = np.arctan2(feet_middle_from_hip[0],
                           -feet_middle_from_hip[1]) * 57.295779513
        print("Precalib z angle: ", value)
        self.params.rot_change_z(-value + 180)
        for j in range(self.session.previous_pose3d.shape[0]):
            self.session.previous_pose3d[j] = self.params.global_rot_z.apply(
                self.session.previous_pose3d[j])

        feet_middle_from_origin = (self.session.previous_pose3d[0] +
                                   self.session.previous_pose3d[5]) / 2
        feet_middle_from_hip = feet_middle_from_origin - hip_from_origin
        value = np.arctan2(feet_middle_from_hip[0],
                           -feet_middle_from_hip[1]) * 57.295779513
        print("Postcalib z angle: ", value)

        ##tilt calibration
        value = np.arctan2(feet_middle_from_hip[2],
                           -feet_middle_from_hip[1]) * 57.295779513
        print("Precalib x angle: ", value)
        self.params.rot_change_x(value + 90)
        for j in range(self.session.previous_pose3d.shape[0]):
            self.session.previous_pose3d[j] = self.params.global_rot_x.apply(
                self.session.previous_pose3d[j])
        feet_middle_from_origin = (self.session.previous_pose3d[0] +
                                   self.session.previous_pose3d[5]) / 2
        feet_middle_from_hip = feet_middle_from_origin - hip_from_origin
        value = np.arctan2(feet_middle_from_hip[2],
                           -feet_middle_from_hip[1]) * 57.295779513
        print("Postcalib x angle: ", value)

        # yaw (y axis) calibration
        feet_rot = self.session.previous_pose3d[0] - self.session.previous_pose3d[5]
        value = np.arctan2(feet_rot[0], feet_rot[2])
        value_hmd = np.arctan2(headsetrot.as_matrix()[0][0],
                               headsetrot.as_matrix()[2][0])
        print("Precalib y value: ", value * 57.295779513)
        print("hmd y value: ", value_hmd * 57.295779513)

        value = value - value_hmd
        value = -value
        print("Calibrate to value:", value * 57.295779513)
        self.params.rot_change_y(value * 57.295779513)

        for j in range(self.session.previous_pose3d.shape[0]):
            self.session.previous_pose3d[j] = self.params.global_rot_y.apply(
                self.session.previous_pose3d[j])

        feet_rot = self.session.previous_pose3d[0] - self.session.previous_pose3d[5]
        value = np.arctan2(feet_rot[0], feet_rot[2])
        print("Postcalib y value: ", value * 57.295779513)

        skelSize = np.max(self.session.previous_pose3d, axis=0) - np.min(
            self.session.previous_pose3d, axis=0)
        print("the camerase estimated you this tall : ", skelSize[1])

        # the vector that you apply(add) after the rotation, in order to
        # change points-in-intermediateRotatedFrame to points-in-vrFrame
        self.session.offset = neckhmdpos - self.session.previous_pose3d[7]
        print("estimated offset : ", self.session.offset)
        print("because neck hmd pos : ", neckhmdpos)
        print("because neck pos in cameraframe : ", self.session.previous_pose3d[7])


def SyncCams(session, timeStampData, numCamRange, vidNames, camIDs):
    """ 
    Runs the time-syncing process. Accesses saved timestamps, runs the time-syncing GUI, and on user-permission, proceeds to create
    synced videos 
    """
    session.syncedVidPath.mkdir(exist_ok=True)

    # start the timesync process
    frameTable, timeTable, unix_synced_timeTable, frameRate, resultsTable, plots = timesync.TimeSync(
        session,
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
        videotrim.VideoTrim(session, vidNames, frameTable,
                            session.parameterDictionary,
                            session.rotationInputs,
                            numCamRange)
        session.session_settings['recording_parameters'].update(
            {'numFrames': session.numFrames})
        # videotrim.createCalibrationVideos(session,60,parameterDictionary)
        print('all done')
