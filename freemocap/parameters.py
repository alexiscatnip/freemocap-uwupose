import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import json

class Parameters():
    def __init__(self) -> None:
        """
        get parameters from saved_params.json file
        if this file not exists, print error message and use generic values.
        """

        param = {}

        # cameras.
        param["camid"] = ['http://192.168.1.103:8080/video']
        param["camera_width"] = [640]
        param["camera_height"] = [360]
        param["camera_settings"] = False # camera settings popup window.


        # mediapipe model.
        param["model_complexity"] = 1
        param["min_tracking_confidence"] = 0.5
        param["static_image"] = False

        # calibration of the two coordinate frames
        param["neckoffset"] = [0.0, 0.0, 0.0]
        param["rotateclock"] = False
        param["rotatecclock"] = False
        param["rotate"] = None

        param["calib_scale"] = True
        param["calib_tilt"] = True
        param["calib_rot"] = True
        param["use_hands"] = False
        param["ignore_hip"] = False

        # smoothing/filtering of final points.
        param["feetrot"] = False


        # steamVR API
        param["camlatency"] = 0.05
        param["smooth"] = 0.5


        #is this needed? / what in the heavens is this
        param["prevskel"] = False
        param["advanced"] = True
        param["imgsize"] = 640
        param["waithmd"] = False


        self.advanced = param["advanced"]
        self.model = param["model_complexity"]
        self.min_tracking_confidence = param["min_tracking_confidence"]
        self.static_image = param["static_image"]

        # PARAMETERS:
        self.maximgsize = param[
            "imgsize"]  # to prevent working with huge images, images that have one axis larger than this value will be downscaled.
        self.cameraid = param[
            "camid"]  # to use with an usb or virtual webcam. If 0 doesnt work/opens wrong camera, try numbers 1-5 or so
        # cameraid = "http://192.168.1.102:8080/video"   #to use ip webcam, uncomment this line and change to your ip
        self.hmd_to_neck_offset = [0, -0.2,
                                   0.1]  # offset of your hmd to the base of your neck, to ensure the tracking is stable even if you look around. Default is 20cm down, 10cm back.
        self.preview_skeleton = param[
            "prevskel"]  # if True, whole skeleton will appear in vr 2 meters in front of you. Good to visualize if everything is working
        self.dont_wait_hmd = param[
            "waithmd"]  # dont wait for movement from hmd, start inference immediately.
        # self.camera_latency = param["camlatency"]
        self.smoothing = param["smooth"]
        self.camera_latency = 0.1
        self.smoothing_1 = 0.5
        self.additional_smoothing_1 = 0
        self.smoothing_2 = 0.5
        self.additional_smoothing_2 = 0
        self.feet_rotation = param["feetrot"]
        self.use_hands = param["use_hands"]
        self.ignore_hip = param["ignore_hip"]

        self.camera_settings = param["camera_settings"]
        self.camera_width = param["camera_width"]
        self.camera_height = param["camera_height"]

        self.render_cameras = True

        self.calib_rot = True
        self.calib_tilt = True
        self.calib_scale = True

        self.recalibrate = False

        # rotations in degrees!
        self.euler_rot_y = 0
        self.euler_rot_x = 0
        self.euler_rot_z = 0

        self.offset = np.asarray([0,0,0])

        self.posescale = 1

        self.exit_ready = False


        self.img_rot_dict_rev = {None: 0, cv2.ROTATE_90_CLOCKWISE: 1,
                                 cv2.ROTATE_180: 2,
                                 cv2.ROTATE_90_COUNTERCLOCKWISE: 3}

        self.paused = False

        self.flip = False

        self.log_frametime = False

        self.global_rot_y = R.from_euler('y', self.euler_rot_y,
                                         degrees=True)  # default rotations, for 0 degrees around y and x
        self.global_rot_x = R.from_euler('x', self.euler_rot_x - 90,
                                         degrees=True)
        self.global_rot_z = R.from_euler('z', self.euler_rot_z - 180,
                                         degrees=True)

        self.smoothing = self.smoothing_1
        self.additional_smoothing = self.additional_smoothing_1

        # load from filesystem
        self.load_params()


    def change_recalibrate(self):
        self.recalibrate = True

    def rot_change_y(self,
                     value):  # callback functions. Whenever the value on sliders are changed, they are called
        print(f"Changed y rotation value to {value}")
        self.euler_rot_y = value
        self.global_rot_y = R.from_euler('y', value,
                                         degrees=True)  # and the rotation is updated with the new value.

    def rot_change_x(self, value):
        print(f"Changed x rotation value to {value}")
        self.euler_rot_x = value
        self.global_rot_x = R.from_euler('x', value - 90, degrees=True)

    def rot_change_z(self, value):
        print(f"Changed z rotation value to {value}")
        self.euler_rot_z = value
        self.global_rot_z = R.from_euler('z', value - 180, degrees=True)

    def change_scale(self, value):
        print(f"Changed scale value to {value}")
        # posescale = value/50 + 0.5
        self.posescale = value


    def change_smoothing(self, val, paramid=0):
        print(f"Changed smoothing value to {val}")
        self.smoothing = val

        if paramid == 1:
            self.smoothing_1 = val
        if paramid == 2:
            self.smoothing_2 = val

    def change_additional_smoothing(self, val, paramid=0):
        print(f"Changed additional smoothing value to {val}")
        self.additional_smoothing = val

        if paramid == 1:
            self.additional_smoothing_1 = val
        if paramid == 2:
            self.additional_smoothing_2 = val

    def change_camera_latency(self, val):
        print(f"Changed camera latency to {val}")
        self.camera_latency = val

    def change_neck_offset(self, x, y, z):
        print(f"Hmd to neck offset changed to: [{x},{y},{z}]")
        self.hmd_to_neck_offset = [x, y, z]


    def ready2exit(self):
        self.exit_ready = True

    def save_params(self):
        param = {}
        param["smooth1"] = self.smoothing_1
        param["smooth2"] = self.smoothing_2

        param["camlatency"] = self.camera_latency
        param["addsmooth1"] = self.additional_smoothing_1
        param["addsmooth2"] = self.additional_smoothing_2

        # if self.flip:
        param["roty"] = self.euler_rot_y
        param["rotx"] = self.euler_rot_x
        param["rotz"] = self.euler_rot_z
        param["scale"] = self.posescale

        param["calibrot"] = self.calib_rot
        param["calibtilt"] = self.calib_tilt
        param["calibscale"] = self.calib_scale

        param["render_cameras"] = self.render_cameras

        param["flip"] = self.flip

        param["hmd_to_neck_offset"] = self.hmd_to_neck_offset

        # print(param["roty"])

        with open("saved_params.json", "w") as f:
            json.dump(param, f, indent=4)

    def load_params(self):
        try:
            with open("saved_params.json", "r") as f:
                param = json.load(f)

            # print(param["roty"])

            self.smoothing_1 = param["smooth1"]
            self.smoothing_2 = param["smooth2"]
            self.camera_latency = param["camlatency"]
            self.additional_smoothing_1 = param["addsmooth1"]
            self.additional_smoothing_2 = param["addsmooth2"]

            self.euler_rot_y = param["roty"]
            self.euler_rot_x = param["rotx"]
            self.euler_rot_z = param["rotz"]
            self.posescale = param["scale"]

            self.calib_rot = param["calibrot"]
            self.calib_tilt = param["calibtilt"]
            self.calib_scale = param["calibscale"]

            self.render_cameras = param["render_cameras"]

            if self.advanced:
                self.hmd_to_neck_offset = param["hmd_to_neck_offset"]

            self.flip = param["flip"]
        except:
            print(
                "Save file not found, will use simple defaults.")


if __name__ == "__main__":
    print("hehe")
