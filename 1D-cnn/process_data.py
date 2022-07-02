#finds the optimal value for the signal segmentation length from the traing dataset
import statistics
from statistics import mode
import neurokit2 as nk
import pandas as pd
from ecgdetectors import Detectors
from imblearn.over_sampling import SMOTE 
from sklearn.decomposition import PCA
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import padasip as pa
import scipy.io
import scipy.fftpack
import scipy.signal as signal



#since our model has predictions for parts of the whole signal, we want to assign a final prediction of a signal
#based on the predictions from the segmentation of the signal
def decode_predictions(model,data,decoder):

    final_predictions=list()
    #whole predictions from all segmentation signals
    predictions_raw=model.predict(data)
    #since predictions are in % for each label, the one with the higher value is our prediction
    predictions=list()
    for idx, pred_label in enumerate(predictions_raw):
        predictions.append(np.argmax(pred_label))
    #this dictionary will save all the predictions in groups
    label_decoder={}
    for idx, label in enumerate(predictions):
        #if we have saved other predictions from the same whole signal
        if decoder[idx] in label_decoder:
            label_decoder[decoder[idx]].append(label)
        else:
            #generate a new list to save them
            label_decoder[decoder[idx]]=list()
            label_decoder[decoder[idx]].append(label)    
    
    for idx,keys in enumerate(label_decoder):
        #get the most frequent prediction
        frequent_label=mode(label_decoder[keys])
        #convert it into a letter
        labels2give="N"
        if int(frequent_label)==0:
            labels2give="N"
        elif int(frequent_label)==1:
            labels2give="A"
        elif int(frequent_label)==2:
            labels2give="O"
        else:
            labels2give="~"
        #save the final predictred labels
        final_predictions.append((keys,labels2give))
    return final_predictions

#overall generats the data in a format for the CNN to be used
#numHB is an integer that gives the number of beats each signals will have from the original signal
def generate_data_set(numHB,ecg,names,freq,labels=None,smote:bool=False,minL:int=0,pca_prev=None,pca:bool=False):
    #data will be returen as a dictionary with the desired variables 
    data={}
    #first get the ecg split into hear beats
    hb,hb_labels,decoder=split_HB(numHB,ecg,names,freq,labels)
    #find the optimal length if minL was not given
    if minL<=0:
        minL=int(find_optimal_length(hb))
        
     #transforms it into a list of np array
    temp=list()
    for idx,ecg_lead in enumerate(hb): 
        temp.append(ecg_lead.to_numpy())
    #make all heart beat signals the same length, for larger signal cut them, for shorter signals just add 0s until it matches the desired length
  
    #length normalization
    new_ecg=list()
    new_labels=list()
    #instead of removing them, we want to do something different, add them in another list
    for idx, ecg_lead in enumerate(temp):
        if len(ecg_lead)>=minL:
            new_ecg.append(ecg_lead[:minL])
            if len(hb_labels)>0:
                new_labels.append(hb_labels[idx])
        else:
            toAdd=minL-len(ecg_lead)
            toAdd=np.zeros(toAdd)
            #merges the signal that was too short with signal full of 0 to achive the desired lenght
            new_ecg.append(np.concatenate((ecg_lead, toAdd)))
            if len(hb_labels)>0:
                new_labels.append(hb_labels[idx])
    #use the smote alg to generate more training data for unbalanced data sets
    if smote and len(new_labels)>0:
        sm = SMOTE()
        new_ecg, new_labels = sm.fit_resample(new_ecg, new_labels)

    
    #only use pca if we say so
    if pca:
        #if we dont send any older pca, then create its own
        if pca_prev is None:
            pca_prev=PCA(.99)
            pca_prev.fit(new_ecg)
        #use the pca to reduce the dimension of the data
        new_ecg=pca_prev.transform(new_ecg)
        data['pca']=pca_prev

 #convert the data into working dataset for the cnn model
    x=new_ecg
    x=np.array(x)
    x=x.reshape((x.shape[0],x.shape[1],1))
    if len(hb_labels)>0:
        y=np.array(new_labels)
        data['labels']=y
    #return the data in a dictionary way, to simplily later use of the desired variables
   
    data['input']=x
    
    data['decoder']=decoder
    data['optimal_length']=minL
    
    return data

#segment the data after a specific amount of R-peaks
def split_HB(num_HB, ecg,names,fs,labels=None):

    hb=list()
    hb_labels=list()
    #saves information about which segmented signal is part of which whole signal
    decoder=list()
    #start detectors as alternative
    detectors=Detectors(fs)
    for idx, ecg_lead in enumerate(ecg):
         #converts to a series
        ecg_lead=pd.Series(ecg_lead)
        #find the R peaks
        try:
            _, rpeaks = nk.ecg_peaks(ecg_lead,sampling_rate=fs)
            rpeaks=rpeaks['ECG_R_Peaks']
            prev_slice=rpeaks[0]

        #if we cant find the r peaks because sadge, then use other detector
        except:
            rpeaks = detectors.pan_tompkins_detector(ecg_lead)
            prev_slice=rpeaks[0]
            #this is the splitting of the signal
        #if the previous signal didnt find that many r peaks use alternative detection
        if len(rpeaks)<2*num_HB:
            rpeaks = detectors.pan_tompkins_detector(ecg_lead)
            prev_slice=rpeaks[0]
        for count, posR in enumerate(rpeaks):
            #only extract from the n_th peak
            if (count%num_HB)==0 and len(ecg_lead[prev_slice:posR])>0:
                hb.append(ecg_lead[prev_slice:posR])
                if not labels is None:
                    hb_labels.append(labels[idx])
                decoder.append(names[idx])
                prev_slice=posR
    return hb, hb_labels,decoder



#find the optimal length of a segmented signal from the overall training data set
def find_optimal_length(hb):
    opt_len=list()
    for idx, ecg in enumerate(hb):
        opt_len.append(len(ecg))
    opt_len=statistics.median(opt_len)
    #opt_len=mode(opt_len)
    return opt_len

#to filter the data
def filter_ecg(y):
    # Number of samplepoints
    N = len(y)
    # sample spacing
    Fs = 300
    T = 1.0 / Fs
    #Compute x-axis
    x = np.linspace(0.0, N*T, N)

    #Compute FFT
    yf = scipy.fftpack.fft(y)
    #Compute frequency x-axis
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))


                                        ###Compute filtering co-efficients to eliminate 50hz brum noise###
    #band_filt = np.array([45, 55])
    #b, a = signal.butter(2, band_filt/(Fs/2), 'bandstop', analog=False)
    b, a = signal.butter(4, 50/(Fs/2), 'low')

    ###ax3.plot(w, 20 * np.log10(abs(h)))
    #Compute filtered signal
    tempf = signal.filtfilt(b,a, y)
    #b, a = signal.butter(1, band_filt/(Fs/2), 'bandstop')
    tempf = signal.filtfilt(b,a, y)
    yff = scipy.fftpack.fft(tempf)

                                        ### Compute Kaiser window co-effs to eliminate baseline drift noise ###
    nyq_rate = Fs/ 2.0
    # The desired width of the transition from pass to stop.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    O, beta = signal.kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    cutoff_hz = 4.0
                                        ###Use firwin with a Kaiser window to create a lowpass FIR filter.###
    taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
    # Use lfilter to filter x with the FIR filter.
    y_filt = signal.lfilter(taps, 1.0, tempf)
    yff = scipy.fftpack.fft(y_filt)

    return y_filt