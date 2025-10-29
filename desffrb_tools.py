import numpy as np
import psrchive
import pywt
from itertools import groupby
from itertools import chain
import itertools
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy import stats

def readfile(filename):
    basename = filename.replace('.ar','')
    arch = psrchive.Archive_load(filename)
    ####info####
    subint=arch.get_Integration(0)
    obs_t = subint.get_duration()
    nchan = subint.get_nchan()
    mjd = subint.get_start_time().in_days()
    freq_lo = arch.get_centre_frequency() - arch.get_bandwidth()/2.0
    freq_hi = arch.get_centre_frequency() + arch.get_bandwidth()/2.0
    ####info####
    arch.dedisperse()
    arch.remove_baseline()
    arch.convert_state('Stokes')
    data = arch.get_data()
    #### only the I ####
    data = data[0,0,:,:]
    return data,obs_t,mjd,freq_lo,freq_hi

def smooth(sig,level=5,threshold=1, wavelet='db8'):
    sig = np.array(sig)
    sigma = sig.std()
    dwtmatr = pywt.wavedec(sig, wavelet=wavelet, level=level)
    denoised = dwtmatr[:]
    denoised[1:] = [pywt.threshold(i, value=threshold*sigma, mode='soft') for i in dwtmatr[1:]]
    smoothed_sig = pywt.waverec(denoised, wavelet, mode='sp1')[:sig.size]
    noises = sig - smoothed_sig
    return smoothed_sig, noises

def judge_sig(arr):
    idx=np.arange(len(arr))
    sig_id=idx[arr>np.mean(arr)+3*np.std(arr)]
    if len(sig_id)>20 and np.std(sig_id)<np.mean(sig_id):
        return 1
    else:
        return 0

def distance_c(arr,min_int=5):
    bandpass = np.where(arr>3*np.std(arr)+np.mean(arr))[0]
    res=np.diff(bandpass)
    res=np.insert(res,0,1)
#     print(len(arr),res)
    if max(res)<min_int:
        return [[0]]
    else:    
        breaks=np.array(bandpass)[np.array(res)>min_int]
        bandpass=list(bandpass)
        [bandpass.insert(list(bandpass).index(i),'div') for i in breaks]
        result=[list(g) for k,g in groupby(bandpass,lambda x:x=='div') if not k]
        return result

def find_wide(cluster,distance=3):
    wide_rfi_index=[]
    cluster_list = np.arange(len(cluster))
    cluster_pulse_width_list = [len(cluster[i]) for i in cluster_list]
    if len(cluster_list)<1:
        idxbad_wideband=[0]
    else:    
        cluster_pulse_width_list = [len(cluster[i]) for i in cluster_list]
#         print(cluster_pulse_width_list,cluster)
        wide_band_index =np.where(np.array(cluster_pulse_width_list)>distance)[0]
        for k in wide_band_index:
            wide_rfi_index.append(np.arange(min(cluster[k]),max(cluster[k])+1).tolist())
        idxbad_wideband = np.array(list(chain(*wide_rfi_index)))
    return idxbad_wideband

def running_smooth(arr,window=64,full='empty'):
    df = pd.DataFrame(arr)
    smooth_arr=df.rolling(window=64).mean()
    smooth_arr=np.array(smooth_arr)[:,0]
    smooth_arr=np.hstack((np.array(smooth_arr[window//2:]),np.array(smooth_arr[:window//2])))
    if full=='empty':
        #smooth_arr=np.hstack((np.full(int(window//4),np.nan),smooth_arr,(np.full(int(window-window//4-window//2),np.nan))))
       pass
    else:
        full=np.float(full)
#        smooth_arr=np.hstack((np.full(int(window//4),full),smooth_arr,(np.full(int(window-window//4-window//2),full))))
        np.nan_to_num(smooth_arr,nan=full)
    return smooth_arr


def dbscan(smooth_data,threshold,pulse_frac,eps, dens,zscore):
    prd_iddata=np.zeros((smooth_data.shape))
    bandw,width = smooth_data.shape
    t = np.percentile(smooth_data,threshold)
    fit_id_x,fit_id_y=np.where(smooth_data>t)
    fit_id=np.vstack((fit_id_x,fit_id_y)).T
    prd_iddata[fit_id_x,fit_id_y]=1
    mean_dens = len(fit_id_x)/(bandw*width)
    min_samples = int(dens*np.pi*eps**2*mean_dens)
    print('  The threshold is %s; The minimum samples of DBSCAN is %s.'%(threshold,min_samples))
    binary_power = np.mean(prd_iddata,0)
    dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(fit_id) # fitting the model
    labels = dbscan.labels_ # getting the labels
    print('test test', threshold,pulse_frac,eps, dens,zscore)
    #labels, core_samples_mask = DBSCAN(fit_id, eps=eps, min_samples=min_samples)
    counter=Counter(labels)
    ct0=np.array(counter.most_common())
    #print(len(labels))
    ct_zscore = stats.zscore(ct0[:,1])
    #pulse_width = [max(fit_id[:,1][labels==i])-min(fit_id[:,1][labels==i]) for i in burst_label]
    #pulse_width = np.array(pulse_width)
    #jj1 = ((pulse<=pulse_frac*width))
    #jj2 = ((ct_zscore>1))
    q3,q1 = np.percentile(ct0[:,1], [99 ,1])
    #iqr_t = q3+1.5*(q3-q1)
    tt = q3
    #jj1 = ((ct0[:,1]>tt))
    print('zscore',zscore,'dens',dens)
    jj1 = ((ct_zscore>=zscore))
    burst_label = ct0[:,0][jj1]
    if len(burst_label)>0:
        burst_label_id = np.arange(len(ct0[:,0]))[jj1]
        fit_id_burstx=[]
        fit_id_burstx = list(map (lambda i:fit_id_burstx+list(fit_id[:,0][labels==i]),burst_label))
        fit_id_bursty=[]
        fit_id_bursty = list(map (lambda i:fit_id_bursty+list(fit_id[:,1][labels==i]),burst_label))
        pulse_fre = np.array(list(map (lambda i:np.array(Counter(fit_id_burstx[i]).most_common(1)[0]),np.arange(len(burst_label)))))[:,0]
        pulse_width = np.array(list(map (lambda i:np.ptp(np.array(fit_id_bursty[i])[np.array(fit_id_burstx[i])==pulse_fre[i]]),np.arange(len(burst_label)))))
        jj1 = ((pulse_width<=pulse_frac*width)&(pulse_width>20))
        burst_label = burst_label[jj1]
    #ct_zscore = stats.zscore(ct0[:,0])
    #jj2 = ((ct_zscore>1))
    #burst_label = burst_label[jj2]

    #pulse_width = [max(fit_id[:,1][labels==i])-min(fit_id[:,1][labels==i]) for i in burst_label]
    #pulse_width = np.array(pulse_width)
    #burst_label = burst_label[pulse_width<pulse_frac*width]
    return prd_iddata,fit_id,labels,burst_label
