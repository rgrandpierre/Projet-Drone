#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:56:52 2024

@author: christophemura
"""

from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import cv2
import time
import numpy as np
from threading import Thread
from queue import Queue

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


batch_size = 32
img_width, img_height = (128, 128)

# Prédictions du modèle
model = tf.keras.models.load_model('./model_IA_NewImages.h5', compile=False)

optimizer = Adam()

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def processIA(frame):
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resized_frame = cv2.resize(rgb_frame, (img_width, img_height))
    
    processed_frame = resized_frame / 255.0
    
    input_image = np.expand_dims(processed_frame, axis=0)
    
    
    try:
        prediction = model.predict(input_image)
        y_pred = np.argmax(prediction, axis=1)

        return y_pred[0]
    
    except Exception as e:
        print(f"Erreur: {e}")
        return None

def cropProcessFrame(frame):
    taille = 64

    lo = np.array([0, 85, 85])
    hi = np.array([7, 255, 255])
    color_infos = (0, 255, 255)
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output_image = np.zeros((128, 128), dtype=np.uint8)
    
    mask = cv2.inRange(image, lo, hi)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(elements) > 0:
        c = max(elements, key=cv2.contourArea)
        ((x, y), rayon) = cv2.minEnclosingCircle(c)
        
        if rayon > 30:
            top_left = (int(x) - taille, int(y) - taille)
            bottom_right = (int(x) + taille, int(y) + taille)
            
            # Gardes-fous pour éviter que le rectangle ne sorte de l'image
            top_left = (max(top_left[0], 0), max(top_left[1], 0))
            bottom_right = (min(bottom_right[0], image.shape[1]), min(bottom_right[1], image.shape[0]))
            
            # Vérifier que les coordonnées sont valides
            if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
                # Dessiner un rectangle autour de la zone détectée (facultatif)
                cv2.rectangle(image, top_left, bottom_right, color_infos, 1)
                
                # Extraire et retourner la région d'intérêt (ROI)
                output_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                
                output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2BGR)
    
    # Vérifier si output_image est vide
    if output_image.shape[0] > 0 and output_image.shape[1] > 0:
        return output_image
    else:
        return None
    

def process_frame(bebopVision, output_queue):
    print("Lancement du Thread process_frame")
    previous_prediction = None
    counter = 0
    required_count = 3
    
    while True:
        frame = bebopVision.get_latest_valid_picture()
        if frame is not None:
            
            processed_frame = cropProcessFrame(frame)
            
            y_pred = processIA(processed_frame)
            print("Prédiction : ", y_pred)
            
            """
            Vérifier s'il y a de bonne détection
            Mettre un compteur jusqu'à 3 pour voir s'il y a une bonne détection
            """
            
            if y_pred is not None:
                if y_pred == previous_prediction:
                    counter += 1
                else:
                    counter = 1
                    previous_prediction = y_pred
                
                if counter == required_count:
                    print(f"Bonne détection confirmée : Classe {y_pred}")
                    counter = 0
            
            output_queue.put(processed_frame)
            time.sleep(0.25)


def main():
    
    # Create a Bebop object
    bebop = Bebop(ip_address="192.168.42.1")
    
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
        time.sleep(1)

        # OpenCV window to display the video stream
        cv2.namedWindow("Bebop 2 Video Stream", cv2.WINDOW_NORMAL)
        
        # Queue pour communiquer entre les threads
        output_queue = Queue()
        
        processing_thread = Thread(target=process_frame, args=(bebopVision, output_queue))
        processing_thread.start()
        
        while True:
            # Get the frame from the video stream
            frame = bebopVision.get_latest_valid_picture()

            if frame is not None:

                # Display the frame in the OpenCV window
                cv2.imshow("Bebop 2 Video Stream", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('a'):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("Bebop 2 Niveau de gris", frame)

                # Check if the 'q' key is pressed to exit
                elif cv2.waitKey(1) & 0xFF == ord('q'):
                    processing_thread.join()
                    break
        
            if not output_queue.empty():
                    processed_frame = output_queue.get()
                    cv2.imshow("Processed Bebop 2 Video Stream", processed_frame)
                    output_queue.task_done()
        
        # Stop the video stream
        print("Stopping video stream...")
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



