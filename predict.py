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
from process_data import  generate_data_set,filter_ecg,decode_predictions
from keras.models import model_from_json


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='Abgabe3',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
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

#------------------------------------------------------------------------------
# Euer Code ab hier  
    #load CNN
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    cnn_model = model_from_json(loaded_model_json)
    # load weights into new model
    cnn_model.load_weights(model_name+'.h5')
    #compile the model
    cnn_model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    #get the number of Beat segmentation and the optimal length form the model
    num_HB,opt_len=np.load(model_name+'.npy')
    
    
    #filter the data
    for idx, ecg_lead in enumerate(ecg_leads):
        ecg_leads[idx]=filter_ecg(ecg_leads[idx])
        ecg_leads[idx]=ecg_leads[idx]/np.amax(ecg_leads[idx])
    
    #generate the data and decoder in a segmentation form for the CNN
    test=generate_data_set(num_HB,ecg_leads,ecg_names,fs,smote=False,minL=opt_len)
    x_test=test['input']
    decoder=test['decoder']
    
    #make the predictions and return them
    predictions=decode_predictions(cnn_model,x_test,decoder)
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
