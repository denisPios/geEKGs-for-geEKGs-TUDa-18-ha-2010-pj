# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: geEKGs for geEKGs 
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
import joblib
import neurokit2 as nk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#for the signal denoising
import scipy.io.wavfile
import scipy.signal
import joblib
import feature

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : 
str='model.obj',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:

    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

# Euer Code ab hier  
    with open(model_name, 'rb') as f:  
        loaded_model = joblib.load(f)         # Lade Model
        
    predictions = list()
    df = feature.get_feature(ecg_leads, fs)
    narray_x = df.to_numpy()
    pred = loaded_model.predict(narray_x)
    
    for name, p in zip(ecg_names, pred):
        predictions.append((name, p))
#---------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
