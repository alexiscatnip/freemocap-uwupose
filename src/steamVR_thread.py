# this thread outputs the data (tracker poses) to steamVR
import collections
import threading
from collections import deque, namedtuple

import numpy

from src.helpers import sendToSteamVR, mediapipe33To3dpose, get_rot, get_rot_mediapipe, get_rot_hands

import time


def pose_ok(pose3d):
    hip_left = 2
    hip_right = 3
    hip_up = 16

    knee_left = 1
    knee_right = 4

    ankle_left = 0
    ankle_right = 5

    is_ok = not numpy.isnan(pose3d[ankle_left]).any() and \
            not numpy.isnan(pose3d[ankle_right]).any() and \
            not numpy.isnan(pose3d[knee_left]).any() and \
            not numpy.isnan(pose3d[knee_right]).any() and \
            not numpy.isnan(pose3d[hip_left]).any() and \
            not numpy.isnan(pose3d[hip_right]).any() and \
            not numpy.isnan(pose3d[hip_up]).any()
    return is_ok


def set_basestations(session):
    calib_units_to_meters = 1000 # use 1000 if you calibrated in mm
    for idx, camera in enumerate(session.cgroup.cameras):
        Tvec = camera.get_translation() # numpy size 3.  in mm? since we calibed in mm.

        # pose3d[:, 0] = -pose3d[:, 0]  # flip the points a bit since steamvrs coordinate system is a bit diffrent
        # pose3d[:, 1] = -pose3d[:, 1]

        res = sendToSteamVR("addstation")
        print(res)
        tosend = f"updatestation {idx} {-Tvec[0]/calib_units_to_meters} {-Tvec[1]/calib_units_to_meters} {Tvec[2]/calib_units_to_meters} 1 0 0 0"
        res = sendToSteamVR(tosend) # right ('east'), elevation (up), backward (south)
        print(res)

class SteamVRThread(threading.Thread):
    def __init__(self, queue_3d_points : deque, session):
        threading.Thread.__init__(self)
        self.queue_3d_points = queue_3d_points
        self.session = session

        # 1. initialise steamVR connection - keep looping until ok.
        self.steamVR_found = False
        while not self.steamVR_found:
            self.connect_to_steamVR()
            time.sleep(1)

        # set cameras (for debug purposes only -- since we are not calibrating ourselves with respect to the steamVR frame, the cameras postions will be very offset.)
        # set_basestations(session)

        session.use_hands = True
        if session.use_hands:
            self.total_trackers = 5
        else:
            self.total_trackers = 3

        # give  other trackers 'None' as the role.
        # roles = []
        # for i in range(self.other_trackers):
        #     roles.append("None")

        roles = ["TrackerRole_Waist", "TrackerRole_RightFoot", "TrackerRole_LeftFoot", None, None]
        # 2. initialise our trackers in steamVR
        if True:
            for i in range(self.num_trackers, self.total_trackers):
                # adding a tracker into VR.
                resp = sendToSteamVR(f"addtracker MediaPipeTracker{i} {roles[i]}")
                while "error" in resp:
                    resp = sendToSteamVR(f"addtracker MediaPipeTracker{i} {roles[i]}")
                    print("error adding tracker")
                    print(resp)
                    time.sleep(0.2)
                time.sleep(0.2)

            self.params = {}
            self.params['smoothing'] = 0.1
            self.params['additional_smoothing'] = 0.1
            self.camera_latency = 0.0 #300ms
            resp = sendToSteamVR(f"settings 50 {self.params['smoothing']} {self.params['additional_smoothing']}")
            # print("settings returned this: ")
            # print(resp)
            # while "error" in resp:
            #     resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
            #     print(resp)
            #     time.sleep(1)

    def run(self):
        """run the thread."""
        # 3. send 3D pose data stream to vr:
        lastSentTime = time.time()
        while True:
            try:
                points_3D = self.queue_3d_points.pop() # 33*3
                #wrap the points back into mediapipe-pose data 'pose_landmarks"' structure

            except:
                points_3D = None

            calibration_scale_factor = 1000 # since we calibrated in units of mm, we should divide by 1000 to obtain meters, which is the units used in steamVR.
            if points_3D is not None:
                pose3d = mediapipe33To3dpose(points_3D)
                pose3d /= calibration_scale_factor

                """wtf is this?"""
                pose3d[:, 0] = -pose3d[:, 0]  # flip the points a bit since steamvrs coordinate system is a bit diffrent
                pose3d[:, 1] = -pose3d[:, 1]

                pose3d_og = pose3d.copy()
                # params.pose3d_og = pose3d_og

                # for j in range(pose3d.shape[0]):  # apply the rotations from the sliders
                #     pose3d[j] = params.global_rot_z.apply(pose3d[j])
                #     pose3d[j] = params.global_rot_x.apply(pose3d[j])
                #     pose3d[j] = params.global_rot_y.apply(pose3d[j])

                if not pose_ok(pose3d):
                    print("failed to triangulate lower body for this frame.")
                    continue

                self.params['feet_rotation'] = False
                if not self.params['feet_rotation']:
                    rots = get_rot(pose3d)  # get rotation data of feet and hips from the position-only skeleton data
                else:
                    rots = get_rot_mediapipe(pose3d)

                # send feet and hip
                frameTime = time.time() - lastSentTime
                lastSentTime = time.time()
                for i in [(0, 1), (5, 2), (6, 0)]:
                    joint = pose3d[i[0]]
                    # words = f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {-frameTime - self.camera_latency} 0.0"
                    words = f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {0} 0.0"
                    res = sendToSteamVR(words)
                    print(words)

                if self.session.use_hands:
                    hand_rots = get_rot_hands(pose3d)
                    for i in [(10, 0), (15, 1)]:
                        joint = pose3d[i[0]] # for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(
                            f"updatepose {i[1] + 3} {joint[0]} {joint[1]} {joint[2]} {hand_rots[i[1]][3]} {hand_rots[i[1]][0]} {hand_rots[i[1]][1]} {hand_rots[i[1]][2]} {0} 0.6")




            time.sleep(0.001)

    def connect_to_steamVR(self):
        use_steamvr = True
        if use_steamvr:
            print("Connecting to SteamVR...")

            # ask the driver, how many devices are connected to ensure we dont add additional trackers
            # if not, we try again
            self.num_trackers = sendToSteamVR("numtrackers")
            for i in range(10):
                if "error" in self.num_trackers:
                    print("Error in SteamVR connection. Retrying...")
                    time.sleep(1)
                    self.num_trackers = sendToSteamVR("numtrackers")
                else:
                    break

            if "error" in self.num_trackers:
                print("Could not connect to SteamVR after 10 retries!")
                print("Will sleep for 10s before trying to contact SteamVR again")
                time.sleep(10)

            self.num_trackers = int(self.num_trackers[2])
            self.steamVR_found = True
            print("connected to SteamVR")
