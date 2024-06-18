# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:09:07 2024

@author: chris
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Chargement des datasets de validation
batch_size = 32
img_width, img_height = (128, 128)
validation_dir = './DataTest'


# Prédictions du modèle
model = tf.keras.models.load_model('./model_IA_NewImages.h5', compile=False)

optimizer = Adam()

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

""" --------------------------- Préparer les fichers et les labels pour evaluer le modèle -------------------------- """
def testIA(validationDirectory):
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validationDirectory,
        labels='inferred',
        label_mode='int',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = validation_ds.class_names
    print("Class Names:", class_names)

    labels_list = []
    images_list = []
    
    for images, labels in validation_ds:
        labels_list.extend(labels.numpy().flatten())
        images_list.extend(images.numpy())

    print("Labels:", labels_list)
    
    labels_tensor = tf.convert_to_tensor(labels_list)
    images_tensor = tf.convert_to_tensor(images_list)
    
    # Extraction des étiquettes de validation
    class_names = validation_ds.class_names
    y_true = np.concatenate([y for x, y in validation_ds], axis=0)
    
    return images_tensor, labels_tensor, y_true, class_names
""" ---------------------------------------------------------------------------------------------------------------- """

""" ------------------------------- Afficher les informations du modèle préentrainné ------------------------------- """
def showInfoModel(model):
    model.summary()

""" ---------------------------------------------------------------------------------------------------------------- """

""" --------------------------------------------- Evaluer le modèle IA --------------------------------------------- """
def evaluation(evalutionDS, confidence_threshold=0.5):
    resultat = model.evaluate(evalutionDS)
    print("Evaluation Results:", resultat)
    
    prediction = model.predict(evalutionDS)
    #print("Raw Predictions: \n", prediction[:20])
    
    prediction_probabilities = tf.nn.softmax(prediction).numpy()
    
    
    y_pred = np.argmax(prediction_probabilities, axis=1)
    confidence_scores = np.max(prediction_probabilities, axis=1)
    
    """for i in range(len(y_pred)):
        if confidence_scores[i] >= confidence_threshold:
            print(f"Classe prédite : {y_pred[i]}, Probabilité : {confidence_scores[i]}")
            print("Probabilités pour toutes les classes :", prediction_probabilities[i])
            print()  # Ajouter une ligne vide pour séparer les résultats
"""
    
    y_pred_filtered = np.where(confidence_scores >= confidence_threshold, y_pred, -1)
    y_pred_filtered = y_pred_filtered[y_pred_filtered != -1]
    print("Classes prédites avec seuil de confiance:", y_pred_filtered)
    print("Classe prédite : \n",y_pred)
    
    return y_pred_filtered#y_pred
""" ---------------------------------------------------------------------------------------------------------------- """

""" ---------------------------- Calculer et afficher la matrice de confusion du modèle ---------------------------- """
def calculMatriceDeConfusion(y_true, y_pred, classe):
    # Calcul de la matrice de confusion
    matriceDeConfusion = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriceDeConfusion, display_labels=classe)
    
    # Affichage de la matrice de confusion
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
""" ---------------------------------------------------------------------------------------------------------------- """

x_validation, y_validation, y_true, classe = testIA(validationDirectory=validation_dir)

evaluation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
evaluation_dataset = evaluation_dataset.batch(batch_size)

y_pred = evaluation(evaluation_dataset)

calculMatriceDeConfusion(y_true, y_pred, classe)




