# -*- coding: utf-8 -*-
"""
Train code 

"""


import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references
import neurokit2 as nk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#for the signal denoising
import scipy.io.wavfile
import scipy.signal
import joblib

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # 
#Sampling-Frequenz 300 Hz
b, a = scipy.signal.butter(3, [.01, .05], 'band')
#extract first the desired parameters from each ecg signal
#actual paramters
#width of the QRS Complex
q2sMean=()
q2sVar=()
#distance between the P peak and the R peak
p2rMean=()
p2rVar=()
#number of detected P waves compared to the detected R waves
p2Num=()
#distance between the R R peaks
r2rVar=()
#to get if it could extract the desired parameters, 1 if yes, otherwise no
err=()

#for each ecg find the peaks of each wave component
for idx, ecg_lead in enumerate(ecg_leads):
    #converts to a series
    ecg_lead=pd.Series(ecg_lead)
    #ecg_lead= scipy.signal.lfilter(b, a, ecg_lead)
    ecg_lead=ecg_lead/np.amax(ecg_lead)
    #find the R peaks
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
    #qrs complex width
    q2sMean=q2sMean+(qsMean,)
    q2sVar=q2sVar+(qsVar,)
    #distance between the P peak and the R peak
    p2rMean=p2rMean+(prMean,)
    p2rVar=p2rVar+(prVar,)
    #number of detected P waves compared to the detected R waves
    p2Num=p2Num+(pNum,)
    #distance between the R R peaks
    r2rVar=r2rVar+(rrVar,)
    err=err+(er,)
    #just to show how many 
    if (idx % 100)==0:
         print(str(idx) + "\t EKG Signale wurden verarbeitet.")

#now save all the parameters in a single data frame to train
inputTrain = pd.DataFrame(data=dict(variance_of_RR=r2rVar))
inputTrain['Variance RR']= r2rVar
#just to make it look better xd
inputTrain = inputTrain.drop(columns="variance_of_RR")
#keep the extracted parameters
inputTrain['Mean QS width']=q2sMean
inputTrain['Variance QS width']=q2sVar
inputTrain['Mean PR width']=p2rMean
inputTrain['Variance PR width']=p2rVar
inputTrain['Relative number of P waves detected']=p2Num
inputTrain['Could extract parameters']=err

#labels in another data frame separatly
labelTrain=pd.DataFrame(data=dict(Label=ecg_labels))
model_rf = RandomForestClassifier(n_estimators=20,#___________, # Set the number of trees to 20
                                 random_state=123)
labelTrain2=np.ravel(labelTrain)
# Fit the model to the training set
model_rf.fit(inputTrain, labelTrain2)


if os.path.exists("model.obj"):
    os.remove("model.obj")
with open('model.obj', 'wb') as f:
    joblib.dump(model_rf, f)
