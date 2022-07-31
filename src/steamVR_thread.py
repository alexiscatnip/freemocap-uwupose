# this thread outputs the data (tracker poses) to steamVR
import collections
import threading
from collections import deque, namedtuple
from statistics import mean

import numpy
from scipy.signal import savgol_filter

from freemocap import parameters
from src.helpers import sendToSteamVR, mediapipe33To3dpose, get_rot, \
    get_rot_mediapipe, get_rot_hands
from scipy.spatial.transform import Rotation as R
import time
import winsound


def pose_ok(pose3d):
    """
    return true only if important joints are OK.

    happens when one of the filtering step has removed it due to
    reprojection error.
    """
    hip_left = 2
    hip_right = 3
    hip_up = 16

    knee_left = 1
    knee_right = 4

    ankle_left = 0
    ankle_right = 5

    if numpy.isnan(pose3d[ankle_left]).any():
        print("ankle_left is NaN")
    if numpy.isnan(pose3d[ankle_right]).any():
        print("ankle_right is NaN")
    if numpy.isnan(pose3d[knee_left]).any():
        print("knee_left is NaN")
    if numpy.isnan(pose3d[knee_right]).any():
        print("knee_right is NaN")
    if numpy.isnan(pose3d[hip_up]).any():
        print("hip_up is NaN")
    if numpy.isnan(pose3d[hip_left]).any():
        print("hip_left is NaN")
    if numpy.isnan(pose3d[hip_right]).any():
        print("hip_right is NaN")

    is_ok = not numpy.isnan(pose3d[ankle_left]).any() and \
            not numpy.isnan(pose3d[ankle_right]).any() and \
            not numpy.isnan(pose3d[knee_left]).any() and \
            not numpy.isnan(pose3d[knee_right]).any() and \
            not numpy.isnan(pose3d[hip_left]).any() and \
            not numpy.isnan(pose3d[hip_right]).any() and \
            not numpy.isnan(pose3d[hip_up]).any()
    return is_ok


def writeHeaders(writer):
    the_row = ["counter"]
    for idx in range(29):
        the_row.extend([f"Pose{idx},x",
                        f"Pose{idx},y",
                        f"Pose{idx},z"])
    for idx in range(29):
        the_row.extend([f"MeanReprojectionError{idx}"])

    the_row.extend(["frametime"])
    writer.writerow(the_row)


class SteamVRThread(threading.Thread):
    """
    thread managing in/output to SteamVR driver
    my inputs: points to send to VR
    my outputs: the HMD pose from the user's headset.
    """

    def __init__(self,
                 queue_3d_points_to_SteamVR: deque,
                 queue_3d_poses_from_SteamVR: deque,
                 session):
        threading.Thread.__init__(self)
        self.queue_3d_points_to_SteamVR = queue_3d_points_to_SteamVR
        self.queue_3d_poses_from_SteamVR = queue_3d_poses_from_SteamVR
        self.session = session
        self.params = session.params

        # 1. initialise steamVR connection - keep looping until ok.
        self.steamVR_found = False
        while not self.steamVR_found:
            self.connect_to_steamVR()
            time.sleep(1)

        self.session.use_hands = False
        if self.session.use_hands:
            self.total_trackers = 5
        else:
            self.total_trackers = 3

        roles = ["TrackerRole_Waist", "TrackerRole_RightFoot",
                 "TrackerRole_LeftFoot"]
        # 2. initialise our trackers in steamVR
        if True:
            for i in range(self.num_trackers, self.total_trackers):
                # adding a tracker into VR.
                resp = sendToSteamVR(
                    f"addtracker MediaPipeTracker{i} {roles[i]}")
                while "error" in resp:
                    resp = sendToSteamVR(
                        f"addtracker MediaPipeTracker{i} {roles[i]}")
                    print("error adding tracker")
                    print(resp)
                    time.sleep(0.2)
                time.sleep(0.2)

            self.params.smoothing = 0
            self.params.additional_smoothing = 0
            self.camera_latency = 0.0  # 300ms?
            resp = sendToSteamVR(f"settings 50 "
                                 f"{self.params.smoothing} "
                                 f"{self.params.additional_smoothing}")

        self.is_beeping = False

    def run(self):
        # filter_memory = deque(maxlen = 5)
        last_frame_times = deque(maxlen=20)

        """run the thread."""
        # 3. send 3D pose data stream to vr:
        lastSentTime = time.time()

        writeToCsv = False
        if (writeToCsv):
            f = open('steamVROutput_log.csv', 'w')
            import csv
            writer = csv.writer(f)
            writeHeaders(writer)
        frame_count = 0
        try:
            stop_flag = False
            while not stop_flag:
                stop_flag = self.session.params.exit_ready

                try:
                    pose3d, rots, reprojectionerror = \
                        self.queue_3d_points_to_SteamVR.pop()  # 33*3

                except:
                    pose3d = None
                    reprojectionerror = None
                    rots = None

                # to steamVR
                if (pose3d is not None ) and (pose_ok(pose3d)): # dont send
                    # if we are fucked!

                    # if not pose_ok(pose3d):
                    #     # print("failed to triangulate lower body for this frame.")
                    #     self.is_beeping = True
                    #     winsound.PlaySound('sound.wav',
                    #                        winsound.SND_FILENAME | winsound.SND_ASYNC)
                    #     continue

                    # filter_memory
                    # # smooth it.
                    # smoothWinLength = 5
                    # smoothOrder = 3
                    # for dim in range(pose3d.shape[1]):
                    #     for mm in range(pose3d.shape[0]):
                    #         pose3d[mm, dim] = savgol_filter(
                    #             pose3d[mm, dim], smoothWinLength, smoothOrder)

                    # send feet and hip
                    frameTime = time.time() - lastSentTime
                    last_frame_times.append(frameTime)
                    if (writeToCsv):
                        line = [frame_count]
                        frame_count += 1
                        flat = pose3d.flatten()
                        flat = flat.tolist()
                        line = line + flat

                        line = line + reprojectionerror.tolist()

                        line = line + [frameTime]
                        writer.writerow(line)

                    lastSentTime = time.time()
                    print("FPS : " + str(1 / mean(last_frame_times)))
                    for i in [(0, 1), (5, 2), (6, 0)]:
                    # for i in [(6, 0)]:
                        joint = pose3d[i[0]]
                        # words = f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {-frameTime - self.camera_latency} 0.0"
                        words = f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {0} 0"
                        res = sendToSteamVR(words)
                        print(words)

                    if self.session.use_hands:
                         hand_rots = get_rot_hands(pose3d)
                         for i in [(10, 0), (15, 1)]:
                         # for i in [(10, 0)]:
                             joint = pose3d[i[
                                 0]]  # for each foot and hips, offset it by skeleton position and send to steamvr
                             handmsg = f"updatepose {i[1] + 1} {joint[0]} {joint[1]} {joint[2]} {hand_rots[i[1]][3]} {hand_rots[i[1]][0]} {hand_rots[i[1]][1]} {hand_rots[i[1]][2]} {0} 0"
                             sendToSteamVR(handmsg)
                             # print(handmsg)

                # from steamvr
                array = sendToSteamVR("getdevicepose 0")
                if "error" in array:
                    pass
                else:

                    headsetpos = [float(array[3]), float(array[4]),
                                  float(array[5])]
                    headsetrot = R.from_quat(
                        [float(array[7]), float(array[8]), float(array[9]),
                         float(array[6])])

                    neckoffset = headsetrot.apply(
                        [0, -0.2,
                         0.1])  # the neck position seems to be the best point to allign to, as its well defined on
                    # the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.
                    self.queue_3d_poses_from_SteamVR.append([headsetpos,
                                                             headsetrot])

                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            if writeToCsv:
                f.close()

    def connect_to_steamVR(self):
        use_steamvr = True
        if use_steamvr:
            while True:
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
                    print(
                        "Will sleep for 10s before trying to contact SteamVR again")
                    time.sleep(10)
                else:
                    break  # exit of this infinite loop.

            self.num_trackers = int(self.num_trackers[2])
            self.steamVR_found = True
            print("connected to SteamVR")
