# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references
import feature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

def train(ecg_leads, ecg_labels, fs, ecg_names): 
    df = feature.get_feature(ecg_leads, fs)
    #dataframe to numpy_array
    narray_x = df.to_numpy()
    list_x = narray_x.tolist()
    list_y = ecg_labels

    model = RandomForestClassifier(n_estimators=160, min_samples_split=2, min_samples_leaf=4, max_depth=20)
    model.fit(list_x, list_y)

    if os.path.exists("Abgabe_2.obj"):
        os.remove("Abgabe_2.obj")
    with open('Abgabe_2.obj', 'wb') as f:
        joblib.dump(model, f)
