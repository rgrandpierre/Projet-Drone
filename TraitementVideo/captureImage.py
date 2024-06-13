from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time

isAlive = False

class UserVision:
    def __init__(self, vision):
        self.index = 874
        self.vision = vision

    def save_pictures(self, args):
        print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            #filename = "./3/image_3_%05d.png" % self.index
            #cv2.imwrite(filename, img)
            self.index +=1


# make my bebop object
bebop = Bebop(drone_type="Bebop2", ip_address="192.168.42.1")

# connect to the bebop
success = bebop.connect(5)

if (success):
    # start up the video
    bebopVision = DroneVision(bebop, is_bebop=True)

    userVision = UserVision(bebopVision)
    bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = bebopVision.open_video()

    if (success):
        print("Vision successfully started!")
        print("Fly me around by hand!")
        bebop.smart_sleep(5)

        print("Moving the camera using velocity")
        #bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)
        bebop.smart_sleep(25)
        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    bebop.disconnect()
else:
    print("Error connecting to bebop.  Retry")