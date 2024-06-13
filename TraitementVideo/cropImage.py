#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:01:18 2024

@author: christophemura
"""


import numpy as np

import os
import cv2

taille = 64

lo = np.array([0, 85, 85])
hi = np.array([7, 255, 255])
color_infos = (0, 255, 255)


# Définir les chemins des dossiers
input_folder = './3'
output_folder = './3_Crop'
i = 0

# Vérifier si le dossier de sortie existe, sinon le créer
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Fonction pour traiter une image (par exemple, conversion en niveaux de gris)
def process_image(image_path, output_path):
    # Lire l'image
    img = cv2.imread(image_path)
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lo, hi)
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    image2 = cv2.bitwise_and(img, img, mask=mask)
    
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(elements) > 0:
        c = max(elements, key=cv2.contourArea)
        ((x, y), rayon) = cv2.minEnclosingCircle(c)
        
        if rayon > 30:
            cv2.rectangle(img, (int(x)-taille, int(y)-taille), (int(x)+taille, int(y)+taille), color_infos, 1)
            crop_img = img[int(y)-taille:int(y)+taille, int(x)-taille:int(x)+taille]
            # Enregistrer l'image traitée
            
            cv2.imwrite(output_path, crop_img)

# Parcourir toutes les images du dossier d'entrée
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path+"image_1_%05d.png"%i)
        i+=1

print("Traitement terminé!")

cv2.destroyAllWindows()