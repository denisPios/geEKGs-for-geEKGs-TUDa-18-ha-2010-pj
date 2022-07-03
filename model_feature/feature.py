import numpy as np
import pandas as pd
from ecgdetectors import Detectors
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_csi_cvi_features
import neurokit2 as nk

def get_feature(ecg_leads, fs):
    
    mean_nni,sdnn,sdsd,nni_50,pnni_50,nni_20,pnni_20,rmssd,median_nni,range_nni,cvsd,cvnni,mean_hr,max_hr,min_hr,std_hr,cvi,csi=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for idx, ecg_lead in enumerate(ecg_leads):
        ecg_lead=pd.Series(ecg_lead)
        ecg_lead=ecg_lead/np.amax(ecg_lead)
        try:
            cleaned = nk.ecg_clean(ecg_lead, sampling_rate=fs)
            _, r_dict = nk.ecg_peaks(cleaned, fs)
            r_peaks = r_dict['ECG_R_Peaks'].tolist()
        except:
            #does nothing
            e=0
        if(len(r_peaks)==0):
            print('we got a none r_peak signal!!!')
            mean_nni.append(-1)
            sdnn.append(-1)
            sdsd.append(-1)
            nni_50.append(-1)
            pnni_50.append(-1)
            nni_20.append(-1)
            pnni_20.append(-1)
            rmssd.append(-1)
            median_nni.append(-1)
            range_nni.append(-1)
            cvsd.append(-1)
            cvnni.append(-1)
            mean_hr.append(-1)
            max_hr.append(-1)
            min_hr.append(-1)
            std_hr.append(-1)
            csi.append(-1)
            cvi.append(-1)
        else:
            if(len(r_peaks) == 1):               
                r_peaks.append(r_peaks[0])
            #R peak locations in ms
            r_peaks = np.array(r_peaks)/fs*1000
            time_domain_features = get_time_domain_features(r_peaks)
            #mena of RR intervalls
            mean_nni.append(time_domain_features['mean_nni'])
            #variance of RR intervalls
            sdnn.append(time_domain_features['sdnn'])
            #variance of difference between RR intervalls
            sdsd.append(time_domain_features['sdsd'])
            #number of successive RR difference greater than 50ms
            nni_50.append(time_domain_features['nni_50'])
            #nni_50 as fraction with all RR
            pnni_50.append(time_domain_features['pnni_50'])
            #same but for 20ms
            nni_20.append(time_domain_features['nni_20'])
            pnni_20.append(time_domain_features['pnni_20'])
            #square root of the mean between adjacent RR intervals 
            rmssd.append(time_domain_features['rmssd'])
            #median of difference of successive RR
            median_nni.append(time_domain_features['median_nni'])
            #difference between max and min RR interval
            range_nni.append(time_domain_features['range_nni'])
            #rmssd/mean_nni
            cvsd.append(time_domain_features['cvsd'])
            #sdnn/mean_nni
            cvnni.append(time_domain_features['cvnni'])
            #mean Heart rate
            mean_hr.append(time_domain_features['mean_hr'])
            #max Heart rate
            max_hr.append(time_domain_features['max_hr'])
            #min Heart rate
            min_hr.append(time_domain_features['min_hr'])
            std_hr.append(time_domain_features['std_hr'])
            #get csi features, should be something like non linear domain features
            csi_features=get_csi_cvi_features(nn_intervals=r_peaks)
            #cardiac sympathetic index
            csi.append(csi_features['csi'])
            #cardiac vagal index
            cvi.append(csi_features['cvi'])
            

        
    data = {'mean_nni': mean_nni, 'sdnn': sdnn, 'sdsd': sdsd, 'nni_50': nni_50, 'pnni_50': pnni_50, 'nni_20': nni_20, 'pnni_20': pnni_20, 'rmssd':  rmssd, 'median_nni': median_nni, 'range_nni': range_nni, 'cvsd': cvsd, 'cvnni': cvnni, 'mean_hr': mean_hr, 'max_hr': max_hr, 
 'min_hr': min_hr, 'std_hr': std_hr, 'csi':csi,'cvi':cvi}
    df = pd.DataFrame(data)
    df = df.replace(np.nan,-1)

    return df
