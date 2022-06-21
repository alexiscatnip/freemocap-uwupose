import time
import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.spatial.transform import Rotation as R
from src import helpers

use_steamvr = True


class InferenceWindow(tk.Frame):
    def __init__(self, root, session, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)

        self.params = session.params
        self.root = root
        self.session = session

        # calibrate rotation
        self.calib_rot_var = tk.BooleanVar(value=self.params.calib_rot)
        self.calib_flip_var = tk.BooleanVar(value=self.params.flip)
        self.rot_y_var = tk.DoubleVar(value=self.params.euler_rot_y)

        frame1 = tk.Frame(self.root)
        frame1.pack()
        self.calibrate_rotation_frame(frame1)

        self.put_separator()

        # calibrate tilt
        self.calib_tilt_var = tk.BooleanVar(value=self.params.calib_tilt)
        self.rot_x_var = tk.DoubleVar(value=self.params.euler_rot_x)
        self.rot_z_var = tk.DoubleVar(value=self.params.euler_rot_z)

        frame2 = tk.Frame(self.root)
        frame2.pack()
        self.calibrate_tilt_frame(frame2)

        self.put_separator()

        # calibrate scale
        self.calib_scale_var = tk.BooleanVar(value=self.params.calib_scale)
        self.scale_var = tk.DoubleVar(value=self.params.posescale)

        frame3 = tk.Frame(self.root)
        frame3.pack()
        self.calibrate_scale_frame(frame3)

        self.put_separator()

        # recalibrate
        tk.Button(self.root,
                  text='Recalibrate (automatically recalibrates checked values above)',
                  command=self.autocalibrate).pack()

        # pause tracking
        tk.Button(self.root, text='Pause/Unpause tracking',
                  command=self.pause_tracking).pack()

        # show the Profile 1 profile 2 text:
        if self.params.advanced:
            frame_profile = tk.Frame(self.root)
            frame_profile.pack()
            tk.Label(frame_profile, text=" ", width=20).pack(side='left')
            tk.Label(frame_profile, text="Profile 1", width=10).pack(
                side='left')
            tk.Label(frame_profile, text=" ", width=5).pack(side='left')
            tk.Label(frame_profile, text="Profile 2", width=10).pack(
                side='left')
            tk.Label(frame_profile, text=" ", width=5).pack(side='left')
            tk.Label(frame_profile, text=" ", width=5).pack(side='left')

        # smoothing
        frame4 = tk.Frame(self.root)
        frame4.pack()
        self.change_smooothing_frame(frame4)

        # smoothing
        frame4_2 = tk.Frame(self.root)
        frame4_2.pack()
        self.change_add_smoothing_frame(frame4_2)

        # smoothing
        frame4_1 = tk.Frame(self.root)
        frame4_1.pack()
        self.change_cam_lat_frame(frame4_1)

        # neck offset
        if self.params.advanced:
            frame6 = tk.Frame(self.root)
            frame6.pack()
            self.change_neck_offset_frame(frame6)

        # frametime log
        # self.log_frametime_var = tk.BooleanVar(value=self.params.log_frametime)
        # log_frametime_check = tk.Checkbutton(self.root, text="Log frametimes to console", variable=self.log_frametime_var, command=self.change_log_frametime)
        # log_frametime_check.pack()

        # exit
        tk.Button(self.root, text='Press to exit',
                  command=self.params.ready2exit).pack()

        # self.root.after(0, self.set_rot_y_var)
        # self.root.after(0, self.set_rot_x_var)

    def change_neck_offset_frame(self, frame):
        tk.Label(frame, text="HMD to neck offset:", width=20).pack(side='left')

        text1 = tk.Entry(frame, width=5)
        text1.pack(side='left')
        text1.insert(0, self.params.hmd_to_neck_offset[0])

        text2 = tk.Entry(frame, width=5)
        text2.pack(side='left')
        text2.insert(0, self.params.hmd_to_neck_offset[1])

        text3 = tk.Entry(frame, width=5)
        text3.pack(side='left')
        text3.insert(0, self.params.hmd_to_neck_offset[2])

        tk.Button(frame, text='Update',
                  command=lambda *args: self.params.change_neck_offset(
                      float(text1.get()), float(text2.get()),
                      float(text3.get()))).pack(side='left')

    def change_log_frametime(self):
        self.params.log_frametime = self.log_frametime_var.get()
        if self.params.log_frametime:
            print("Enabled frametime logging")
        else:
            print("Disabled frametime logging")

    def set_rot_y_var(self):
        angle = self.params.euler_rot_y

        if self.params.flip:
            angle += 180

        # print("calculated angle from rot is:",angle)
        if angle >= 360:
            angle -= 360
        elif angle < 0:
            angle += 360
        # print("calculated angle final is:",angle)
        self.rot_y_var.set(angle)
        # self.root.after(0, self.set_rot_y_var)

    def set_rot_z_var(self):
        self.rot_z_var.set(self.params.euler_rot_z)

    def set_rot_x_var(self):
        self.rot_x_var.set(self.params.euler_rot_x)
        # self.root.after(0, self.set_rot_x_var)


    def change_rot_auto(self):
        self.params.calib_rot = self.calib_rot_var.get()
        print(
            f"Mark rotation to{'' if self.params.calib_rot else ' NOT'} be automatically calibrated")

    def change_rot_flip(self):
        self.params.flip = self.calib_flip_var.get()
        print("changed flip to: ", self.params.flip)

    def calibrate_rotation_frame(self, frame):
        rot_check = tk.Checkbutton(frame,
                                   text="Enable automatic rotation calibration",
                                   variable=self.calib_rot_var,
                                   command=self.change_rot_auto)  # , command=lambda *args: show_hide(varrot, [rot_y_frame]))
        flip_check = tk.Checkbutton(frame, text="Flip calibration",
                                    variable=self.calib_flip_var,
                                    command=self.change_rot_flip)
        rot_y_frame = tk.Frame(frame)

        rot_check.pack()
        flip_check.pack()
        rot_y_frame.pack()

        rot_y = tk.Scale(rot_y_frame, label="Roation y:", from_=-40, to=400,
                         # command=lambda *args: self.params.rot_change_y(self.rot_y_var.get()),
                         orient=tk.HORIZONTAL, length=400, showvalue=1,
                         tickinterval=60, variable=self.rot_y_var)
        rot_y.pack(expand=True, fill='both', side='left')

        self.rot_y_var.trace_add('write', callback=lambda var, index,
                                                          mode: self.params.rot_change_y(
            self.rot_y_var.get()))

        tk.Button(rot_y_frame, text="<",
                  command=lambda *args: self.rot_y_var.set(
                      self.rot_y_var.get() - 1), width=10).pack(expand=True,
                                                                fill='both',
                                                                side='left')
        tk.Button(rot_y_frame, text=">",
                  command=lambda *args: self.rot_y_var.set(
                      self.rot_y_var.get() + 1), width=10).pack(expand=True,
                                                                fill='both',
                                                                side='left')

    def change_tilt_auto(self):
        self.params.calib_tilt = self.calib_tilt_var.get()
        print(
            f"Mark tilt to{'' if self.params.calib_tilt else ' NOT'} be automatically calibrated")

    def calibrate_tilt_frame(self, frame):

        tilt_check = tk.Checkbutton(frame,
                                    text="Enable automatic tilt calibration",
                                    variable=self.calib_tilt_var,
                                    command=self.change_tilt_auto)  # , command=lambda *args: show_hide(vartilt, [rot_z_frame, rot_x_frame]))
        tilt_check.pack()

        rot_x_frame = tk.Frame(frame)
        rot_x_frame.pack()
        rot_z_frame = tk.Frame(frame)
        rot_z_frame.pack()

        rot_x = tk.Scale(rot_x_frame, label="Roation x:", from_=0, to=180,
                         # command=lambda *args: self.params.rot_change_x(self.rot_x_var.get()),
                         orient=tk.HORIZONTAL, length=400, showvalue=1,
                         tickinterval=15, variable=self.rot_x_var)
        rot_x.pack(expand=True, fill='both', side='left')
        self.rot_x_var.trace_add('write', callback=lambda var, index,
                                                          mode: self.params.rot_change_x(
            self.rot_x_var.get()))

        tk.Button(rot_x_frame, text="<",
                  command=lambda *args: self.rot_x_var.set(
                      self.rot_x_var.get() - 1), width=10).pack(expand=True,
                                                                fill='both',
                                                                side='left')
        tk.Button(rot_x_frame, text=">",
                  command=lambda *args: self.rot_x_var.set(
                      self.rot_x_var.get() + 1), width=10).pack(expand=True,
                                                                fill='both',
                                                                side='left')

        rot_z = tk.Scale(rot_z_frame, label="Roation z:", from_=90, to=270,
                         # command=lambda *args: self.params.rot_change_z(self.rot_z_var.get()),
                         orient=tk.HORIZONTAL, length=400, showvalue=1,
                         tickinterval=30, variable=self.rot_z_var)
        rot_z.pack(expand=True, fill='both', side='left')
        self.rot_z_var.trace_add('write', callback=lambda var, index,
                                                          mode: self.params.rot_change_z(
            self.rot_z_var.get()))

        tk.Button(rot_z_frame, text="<",
                  command=lambda *args: self.rot_z_var.set(
                      self.rot_z_var.get() - 1), width=10).pack(expand=True,
                                                                fill='both',
                                                                side='left')
        tk.Button(rot_z_frame, text=">",
                  command=lambda *args: self.rot_z_var.set(
                      self.rot_z_var.get() + 1), width=10).pack(expand=True,
                                                                fill='both',
                                                                side='left')

    def change_scale_auto(self):
        self.params.calib_scale = self.calib_scale_var.get()
        print(
            f"Mark scale to{'' if self.params.calib_scale else ' NOT'} be automatically calibrated")

    def calibrate_scale_frame(self, frame):

        scale_check = tk.Checkbutton(frame,
                                     text="Enable automatic scale calibration",
                                     variable=self.calib_scale_var,
                                     command=self.change_scale_auto)  # , command=lambda *args: show_hide(varrot, [rot_y_frame]))
        scale_frame = tk.Frame(frame)

        scale_check.pack()
        scale_frame.pack()

        # command=lambda *args: self.params.change_scale(self.scale_var.get()),
        scale = tk.Scale(scale_frame, label="Scale:", from_=0.5, to=2.0,
                         orient=tk.HORIZONTAL, length=400, showvalue=1,
                         tickinterval=0.1, variable=self.scale_var,
                         resolution=0.01)
        scale.pack(expand=True, fill='both', side='left')
        self.scale_var.trace_add(
            'write', callback=
            lambda var, index, mode: self.params.change_scale(
                self.scale_var.get()))

        tk.Button(scale_frame, text="<",
                  command=lambda *args: self.scale_var.set(
                      self.scale_var.get() - 0.01), width=10).pack(expand=True,
                                                                   fill='both',
                                                                   side='left')
        tk.Button(scale_frame, text=">",
                  command=lambda *args: self.scale_var.set(
                      self.scale_var.get() + 0.01), width=10).pack(expand=True,
                                                                   fill='both',
                                                                   side='left')

    def change_smooothing_frame(self, frame):

        tk.Label(frame, text="Smoothing window:", width=20).pack(side='left')
        smoothingtext1 = tk.Entry(frame, width=10)
        smoothingtext1.pack(side='left')
        smoothingtext1.insert(0, self.params.smoothing_1)

        tk.Button(frame, text='Update',
                  command=lambda *args: self.params.change_smoothing(
                      float(smoothingtext1.get()), 1)).pack(side='left')

        if self.params.advanced:
            smoothingtext2 = tk.Entry(frame, width=10)
            smoothingtext2.pack(side='left')
            smoothingtext2.insert(0, self.params.smoothing_2)

            tk.Button(frame, text='Update',
                      command=lambda *args: self.params.change_smoothing(
                          float(smoothingtext2.get()), 2)).pack(side='left')

        tk.Button(frame, text='Disable',
                  command=lambda *args: self.params.change_smoothing(
                      0.0)).pack(side='left')

    def change_cam_lat_frame(self, frame):

        tk.Label(frame, text="Camera latency:", width=20).pack(side='left')
        lat = tk.Entry(frame, width=10)
        lat.pack(side='left')
        lat.insert(0, self.params.camera_latency)

        tk.Button(frame, text='Update',
                  command=lambda *args: self.params.change_camera_latency(
                      float(lat.get()))).pack(side='left')

    def change_add_smoothing_frame(self, frame):

        tk.Label(frame, text="Additional smoothing:", width=20).pack(
            side='left')
        lat1 = tk.Entry(frame, width=10)
        lat1.pack(side='left')
        lat1.insert(0, self.params.additional_smoothing_1)

        tk.Button(frame, text='Update', command=lambda
            *args: self.params.change_additional_smoothing(float(lat1.get()),
                                                           1)).pack(
            side='left')
        if self.params.advanced:
            lat2 = tk.Entry(frame, width=10)
            lat2.pack(side='left')
            lat2.insert(0, self.params.additional_smoothing_2)

            tk.Button(frame, text='Update', command=lambda
                *args: self.params.change_additional_smoothing(
                float(lat2.get()), 2)).pack(side='left')

        tk.Button(frame, text='Disable', command=lambda
            *args: self.params.change_additional_smoothing(0.0)).pack(
            side='left')

    def autocalibrate(self):
        self.session.do_calibration_flag = True
        self.params.recalibrate = False # this is the old mediappipe one/// ?
        time.sleep(1)
        #strange way to update the UI without callback...
        self.set_rot_z_var()
        self.set_rot_x_var()
        self.set_rot_y_var()

    def put_separator(self):
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x')

    def pause_tracking(self):
        self.params.paused = not self.params.paused


def make_inference_gui(session):
    root = tk.Tk()
    InferenceWindow(root, session).pack(side="top", fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    # make_inference_gui()
    print("hehe")
