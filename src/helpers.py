import cv2
import numpy
import numpy as np
from scipy.spatial.transform import Rotation as R

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]

skeleton3d = ((0,1),(1,2),(5,4),(4,3),(2,6),(3,6),(6,16),(16,7),(7,8),(8,9),(7,12),(7,13),(10,11),(11,12),(15,14),(14,13)) #head is 9, one hand is 10, other is 15


def draw_pose(frame,pose,size):
    pose = pose*size
    for sk in EDGES:
        cv2.line(frame,(int(pose[sk[0],1]),int(pose[sk[0],0])),(int(pose[sk[1],1]),int(pose[sk[1],0])),(0,255,0),3)


def midpoint(p1, p2):
    return [p1[0]/2+p2[0]/2,p1[1]/2+p2[1]/2,p1[2]/2+p2[2]/2]


def mediapipe33To3dpose(lms):
    
    #convert 33*3 to the skeleton he used.
    # lms means nothing at this poiont.
    
    pose = np.zeros((29,3))

    # ctrl-r.... wtf. omg. hahahahahahaah
    # .x = [0]
    # .y = [1]
    # .z = [2]

    pose[0]=lms[28] #R_ankle
    pose[1]=lms[26] #R_knee
    pose[2]=lms[24] #R_hip
    pose[3]=lms[23] #_hip
    pose[4]=lms[25] #L_knee
    pose[5]=lms[27] #L_ankle

    pose[6]=midpoint(lms[24], lms[23])  # the hip.

    #some keypoints in mediapipe are missing, so we calculate them as avarage of two keypoints
    pose[7]=midpoint(lms[12], lms[11]) # betwn left and right shoulder
    pose[8]=midpoint(lms[10], lms[9]) #mid mouth.

    pose[9]=lms[0] #nose

    pose[10]=lms[15] # right wrist
    pose[11]=lms[13]
    pose[12]=lms[11]

    pose[13]=lms[12]
    pose[14]=lms[14]
    pose[15]=lms[16]# left wrist

    pose[16]= midpoint(pose[7], pose[6]) # btwn mid-shoulder and mid-hip. -- torso?

    #right foot
    pose[17] = lms[31] #forward
    pose[18] = lms[29]  #back
    pose[19] = lms[25] #up
    
    #left foot
    pose[20] = lms[32] #forward
    pose[21] = lms[30] #back
    pose[22] = lms[26]  #up
    
    #right hand
    pose[23] = lms[17]  #forward
    pose[24] = lms[15]  #back
    pose[25] = lms[19] #up
    
    #left hand
    pose[26] = lms[18] #forward
    pose[27] = lms[16]  #back
    pose[28] = lms[20]  #up

    return pose

def keypoints_to_original(scale,center,points):
    scores = points[:,2]
    points -= 0.5
    #print(scale,center)
    #print(points)
    points *= scale
    #print(points)
    points[:,0] += center[0]
    points[:,1] += center[1]
    #print(points)
    
    points[:,2] = scores
    
    return points

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def get_rot_hands(pose3d):

    hand_r_f = pose3d[26]
    hand_r_b = pose3d[27]
    hand_r_u = pose3d[28]
    
    hand_l_f = pose3d[23]
    hand_l_b = pose3d[24]
    hand_l_u = pose3d[25]
    
    # left hand
    
    x = hand_l_f - hand_l_b
    w = hand_l_u - hand_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    l_hand_rot = np.vstack((z, y, -x)).T
    
    # right hand
    
    x = hand_r_f - hand_r_b
    w = hand_r_u - hand_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_hand_rot = np.vstack((z, y, -x)).T

    r_hand_rot = R.from_matrix(r_hand_rot).as_quat()
    l_hand_rot = R.from_matrix(l_hand_rot).as_quat()
    
    return l_hand_rot, r_hand_rot

def get_rot_mediapipe(pose3d):
    hip_left = pose3d[2]
    hip_right = pose3d[3]
    hip_up = pose3d[16]
    
    foot_l_f = pose3d[20]
    foot_l_b = pose3d[21]
    foot_l_u = pose3d[22]
    
    foot_r_f = pose3d[17]
    foot_r_b = pose3d[18]
    foot_r_u = pose3d[19]
    
    # hip
    
    x = hip_right - hip_left
    w = hip_up - hip_left
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T
    
    # left foot
    
    x = foot_l_f - foot_l_b
    w = foot_l_u - foot_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    l_foot_rot = np.vstack((x, y, z)).T
    
    # right foot
    
    x = foot_r_f - foot_r_b
    w = foot_r_u - foot_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_foot_rot = np.vstack((x, y, z)).T
    
    hip_rot = R.from_matrix(hip_rot).as_quat()
    r_foot_rot = R.from_matrix(r_foot_rot).as_quat()
    l_foot_rot = R.from_matrix(l_foot_rot).as_quat()
    
    return hip_rot, r_foot_rot, l_foot_rot

    

def get_rot(pose3d):

    ## guesses
    hip_left = 2
    hip_right = 3
    hip_up = 16
    
    knee_left = 1
    knee_right = 4
    
    ankle_left = 0
    ankle_right = 5
    
    # hip
    
    x = pose3d[hip_right] - pose3d[hip_left]
    w = pose3d[hip_up] - pose3d[hip_left]
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T

    # right leg
    
    y = pose3d[knee_right] - pose3d[ankle_right]
    w = pose3d[hip_right] - pose3d[ankle_right]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_left] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_r_rot = np.vstack((x, y, z)).T

    # left leg
    
    y = pose3d[knee_left] - pose3d[ankle_left]
    w = pose3d[hip_left] - pose3d[ankle_left]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_right] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_l_rot = np.vstack((x, y, z)).T

    rot_hip = R.from_matrix(hip_rot).as_quat()
    rot_leg_r = R.from_matrix(leg_r_rot).as_quat()
    rot_leg_l = R.from_matrix(leg_l_rot).as_quat()

    # to remove after fixed the rotataion.
    if numpy.isnan(rot_hip).any():
        rot_hip.fill(0)
        rot_hip[3] = 1
    if numpy.isnan(rot_leg_r).any():
        rot_leg_r.fill(0)
        rot_leg_r[3] = 1
    if numpy.isnan(rot_leg_l).any():
        rot_leg_l.fill(0)
        rot_leg_l[3] = 1

    return rot_hip, rot_leg_l, rot_leg_r


def sendToSteamVR(text):
    #Function to send a string to my steamvr driver through a named pipe.
    #open pipe -> send string -> read string -> close pipe
    #sometimes, something along that pipeline fails for no reason, which is why the try catch is needed.
    #returns an array containing the values returned by the driver.
    try:
        pipe = open(r'\\.\pipe\ApriltagPipeIn', 'rb+', buffering=0)
        some_data = str.encode(text)
        pipe.write(some_data)
        resp = pipe.read(1024)
    except:
        return ["error"]
    string = resp.decode("utf-8")
    array = string.split(" ")
    pipe.close()
    
    return array