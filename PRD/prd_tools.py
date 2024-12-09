import pandas as pd
import numpy as np
import itertools
from itertools import groupby
from itertools import chain
import pywt
import math
from scipy.fftpack import fft
import heapq

def wavelet(sig,threshold = 3, level=3, wavelet='db8'):
    sigma = sig.std()
    dwtmatr = pywt.wavedec(data=sig, wavelet=wavelet, level=level)
    denoised = dwtmatr[:]
    denoised[1:] = [pywt.threshold(i, value=threshold*sigma, mode='soft') for i in dwtmatr[1:]]
    smoothed_sig = pywt.waverec(denoised, wavelet, mode='sp1')[:sig.size]
    noises = sig - smoothed_sig
    return smoothed_sig,noises

def distance_c(bandpass,min_int):
    res=np.diff(bandpass)
    res=np.insert(res,0,1)
    breaks=np.array(bandpass)[np.array(res)>min_int]
    bandpass=list(bandpass)
    [bandpass.insert(list(bandpass).index(i),'div') for i in breaks]
    result=[list(g) for k,g in groupby(bandpass,lambda x:x=='div') if not k]
    return result

def find_wide(cluster,distance=25):
    wide_rfi_index=[]
    cluster_list = np.arange(len(cluster))
    cluster_length_list = [len(cluster[i]) for i in cluster_list]
    wide_band_index =np.where(np.array(cluster_length_list)>distance)[0]
    for k in wide_band_index:
        wide_rfi_index.append(np.arange(min(cluster[k]),max(cluster[k])+1).tolist())
    idxbad_wideband = np.array(list(chain(*wide_rfi_index)))
    return idxbad_wideband

def detect_peaks(x, mph,n_peaks):
    x = np.atleast_1d(x).astype('float64')
    peaks_index=list(map(list(x).index, heapq.nlargest(n_peaks, x)))
    if len(peaks_index)<=0:
        print('no peaks found')
        return list(1)
    if min(peaks_index)<=mph or max(peaks_index)>=len(x)-50:
       peaks_index.remove(min(peaks_index))
       return peaks_index
    else:
        peaks_index=list(np.delete(peaks_index,np.where(np.array(peaks_index)<mph)[0]))
        mph_arr=np.arange(-mph,mph+1)
        mph_arr=np.delete(mph_arr,mph)
        no_peaks=[]
        for i in peaks_index:
            dx=[x[i]-x[i+j] for j in mph_arr]
            if (np.array(dx)>0).all()==False:
               no_peaks.append(i)
            else:
               pass
        for i in no_peaks:
            peaks_index.remove(i)
        return peaks_index

def judge_continue(array):
    diff = array[1:]-array[:-1]
    #right_edge = np.where(diff!=1)[0]
    right_edge = np.where(diff>50)[0]
    array_split = np.split(np.array(array),np.array(right_edge)+1)
    #out_put = [int(np.median(array_split[i])) for i in range(len(array_split))]
    out_put = [list(array_split[i]) for i in range(len(array_split))]
    return out_put

def judge_in_out(array0,array1):
    array0 = list(array0)
    array1 = list(array1)
    array0 = list(itertools.chain.from_iterable(array0))
    #array1 = list(itertools.chain.from_iterable(array1))
    array0_ = np.array([np.arange(i-20,i+20) for i in array0]).reshape(-1)
    list0 = list(array0_)
    #print('array0',array0_)
    #print('array1',array1)
    num = len(set(list0) & set(array1))
    return num

def norm(data):
    return (data-min(data))/(max(data)-min(data))

def classify_rfi(data):
    ###detree1###
    data = (data-max(data))/(max(data)-min(data))
    std = np.std(data)
    mean = np.mean(data)
    median = np.median(data)
    data_fft = abs(fft(data))[1:len(data)//2]
    detree1=data[(data>=9*std+median)].size
    detree11 = max(data)/np.median(data)
    if detree1>=1:
       rfi_type='Impulsive'
       return rfi_type,data_fft
    else:
       ###detree2###
       peaks_index=list(map(list(data_fft).index, heapq.nlargest(15, data_fft)))
       detree2 = max(peaks_index)
       if detree2 <=30:
          rfi_type='Sporadic'
          return rfi_type,data_fft
       else:
             ###detree3###
             peak_x = detect_peaks(data_fft,mph=50,n_peaks=15)
             mean50 = np.mean(data_fft[:50])
             mean_peaks = np.mean(data_fft[peak_x])
             peak_x_mean = np.mean(peak_x)
             detree3_1 = mean_peaks/np.mean(data_fft[50:])
             detree3_2 = mean_peaks/mean50
             detree3_3 = max(data_fft[50:])/mean50
             detree3 = detree3_1+detree3_2+detree3_3
             if detree3 <= 10:
                rfi_type = 'Sporadic'
                return rfi_type,data_fft
             else:
                if peak_x_mean<30:
                   rfi_type = 'Sporadic'
                   #return rfi_type,data_fft
                else:
                   std1 = np.std(data_fft[30:])
                   if std1*detree3<10:
                      rfi_type = 'Sporadic'
                   else:
                      rfi_type = 'Periodic'
                return rfi_type,data_fft

def running_filter(data,window,step_size,thre=False):
    data1=norm(data)
    test = pd.Series(data1).rolling(window=step_size).mean()
    #test = test[step_size-1:]
    test1 = pd.Series(test).rolling(window=window).std()
    mean1=np.nanmean(test1)
    std1=np.nanstd(test1)
    if thre:
        jj1=(test1>mean1+thre*std1)
        jj0=(test1<=mean1+thre*std1) 
        test1[jj1]=1
        test1[jj0]=0
    test1[np.isnan(test1)]=0
    return np.array(test1)

def get_new_timeseries(data,ch_d,i):
        time_series = data[:,i]
      #  add_h=list(np.zeros(ch_d[0]-ch_d[i])+round(np.mean(data[:,i])))
        add_h=list(np.zeros(ch_d[0]-ch_d[i]))
        add_t=list(np.zeros(ch_d[i]))
        #add_t=list(np.zeros(ch_d[i])+round(np.mean(data[:,i])))
        new_time_series = add_h+list(time_series)+add_t
        return new_time_series    
