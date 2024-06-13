#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:25:13 2024

@author: christophemura
"""

from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import cv2
import time

def main():
    # Create a Bebop object
    bebop = Bebop(drone_type="Bebop2", ip_address="192.168.42.1")
    

    # Connect to the drone
    print("Connecting to the Bebop 2 drone...")
    success = bebop.connect(10)
    if success:
        print("Successfully connected to Bebop 2 drone!")
        
        bebopVision = DroneVision(bebop, is_bebop=True)
        
        bebopVision.set_user_callback_function(None, user_callback_args=None)

        # Start video stream
        print("Starting video stream...")
        #bebop.start_video_stream()
        bebopVision.open_video()
        time.sleep(2)

        # OpenCV window to display the video stream
        cv2.namedWindow("Bebop 2 Video Stream", cv2.WINDOW_NORMAL)
        
        

        while True:
            # Get the frame from the video stream
            frame = bebopVision.get_latest_valid_picture()

            if frame is not None:
                # Convert the frame to a format suitable for OpenCV
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the OpenCV window
                cv2.imshow("Bebop 2 Video Stream", frame)

            # Check if the 'q' key is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop the video stream
        print("Stopping video stream...")
        #bebop.stop_video_stream()
        bebopVision.close_video()

        # Disconnect from the drone
        print("Disconnecting from the Bebop 2 drone...")
        bebop.disconnect()
        print("Disconnected successfully!")

        # Close all OpenCV windows
        cv2.destroyAllWindows()
    else:
        print("Failed to connect to Bebop 2 drone.")

if __name__ == "__main__":
    main()