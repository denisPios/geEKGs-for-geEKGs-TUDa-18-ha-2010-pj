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

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.obj',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
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
    file_type=model_name.split(".")
    file_type=file_type[1]
    if(file_type=="npy"):
        model_name="model.obj"
    with open(model_name, 'rb') as f:  
        loaded_model = joblib.load(f)         # Lade Model
        #analog to train part
        #actual dataframe
        #width of the QRS Complex
        q2sMeanT=()
        q2sVarT=()
        #distance between the P peak and the R peak
        p2rMeanT=()
        p2rVarT=()
        #number of detected P waves compared to the detected R waves
        p2NumT=()
        #distance between the R R peaks
        r2rVarT=()
        errT=()

        #for each ecg find the peaks of each wave component
        for idx, ecg_lead in enumerate(ecg_leads):
            #converts from an nth dimension array to a series
            ecg_lead=pd.Series(ecg_lead)
            #ecg_lead= scipy.signal.lfilter(b, a, ecg_lead)
            ecg_lead=ecg_lead/np.amax(ecg_lead)
            try:
                _, rpeaks = nk.ecg_peaks(ecg_lead,sampling_rate=fs)
                #find the rest wave peaks
                # Delineate the ECG signal
                _, waves_peak = nk.ecg_delineate(ecg_lead, rpeaks, sampling_rate=fs, method="peak")
                q=pd.Series(waves_peak['ECG_Q_Peaks'])
                s=pd.Series(waves_peak['ECG_S_Peaks'])
                p=pd.Series(waves_peak['ECG_P_Peaks'])
            except:
                er=1
                #error might be that we have not many r peaks, still found somehow a lot of the other peaks
                r=s
                r.loc[:]=np.nan
            else:
                #save the locations, rP contains R peak locations for all ecg samples, can be accesed with r[index]
                #all the locations of the q, s, r,p waves peaks
                r=pd.Series(rpeaks['ECG_R_Peaks'])
                er=0
            #get qrs width
            qs=s-q
            #only take into the account the ones you can actually detect
            qs=qs.dropna()
            #get time from frequency to get generalization of parameters
            qs=qs/fs*1000
            qsMean=np.mean(qs)
            qsVar=np.std(qs)
            #get distance between p and r
            pr=r-p
            pr=pr/fs*1000
            prTotal=len(pr)
            pr=pr.dropna()
            prMean=np.mean(pr)
            prVar=np.std(pr)
            prNonNaN=len(pr)
            #to get how many p waves couldn be detected compared to the number of R peaks that got detected
            pNum=(prTotal-prNonNaN)/prTotal
            #difference between the R to R distance
            rrVar=np.std(np.diff(r)/fs*1000)
            #sometimes when the signal is noisy or idk it gets a few nan over all, maybe need to filter signals or something
            if pd.isna(qsMean) or pd.isna(qsVar) or pd.isna(prMean) or pd.isna(prVar) or pd.isna(pNum) or pd.isna(rrVar):
                qsMean=-1
                qsVar=-1
                prMean=-1
                prVar=-1
                pNum=-1
                rrVar=-1
                er=1
                print(idx)
            #save the extracted parameters of all ecg signals
            q2sMeanT=q2sMeanT+(qsMean,)
            q2sVarT=q2sVarT+(qsVar,)
            #distance between the P peak and the R peak
            p2rMeanT=p2rMeanT+(prMean,)
            p2rVarT=p2rVarT+(prVar,)
            #number of detected P waves compared to the detected R waves
            p2NumT=p2NumT+(pNum,)
            #distance between the R R peaks
            r2rVarT=r2rVarT+(rrVar,)
            errT=errT+(er,)
            #just to show how many 
            if (idx % 100)==0:
                 print(str(idx) + "\t EKG Signale wurden verarbeitet.")
        #analog to the train part
        #generate the data frame 
        #del(inputTest)
        inputTest = pd.DataFrame(data=dict(variance_of_RR=r2rVarT))
        inputTest['Variance RR']= r2rVarT
        #just to make it look better xd
        inputTest = inputTest.drop(columns="variance_of_RR")
        #keep addint the extracted parameters
        inputTest['Mean QS width']=q2sMeanT
        inputTest['Variance QS width']=q2sVarT
        inputTest['Mean PR width']=p2rMeanT
        inputTest['Variance PR width']=p2rVarT
        inputTest['Relative number of P waves detected']=p2NumT
        inputTest['Could extract parameters']=errT
        #
        predictions=list()
        pred_rf = loaded_model.predict(inputTest)

    for idx,ecg_lead in enumerate(ecg_leads):
        predictions.append((ecg_names[idx], pred_rf[idx]))
#---------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
