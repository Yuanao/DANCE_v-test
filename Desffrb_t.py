import numpy as np
import sys
import warnings
from desffrb_tools import *
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import pywt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import kurtosis
import pylab 
import h5py
import matplotlib
#matplotlib.use('Agg')

warnings.filterwarnings('ignore')


class desffrb(object):
    def __init__(self):
       self.data = None
       self.fres = None
       self.chenum = None    
    def read_ar(self,filename,tdowns=False,fredowns=False,save=False):
        print('\033[1;32;40m READING FILE...\033[0m\n')
        self.basename = filename.replace('.ar','')
        self.data,self.obs_t,self.mjd,self.freq_lo,self.freq_hi = readfile(filename)
        self.raw_data = self.data.copy()
        self.fsamp,self.tsamp = self.data.shape
        self.nchan=self.fsamp
        self.df = (self.freq_hi-self.freq_lo)/self.nchan
        if fredowns:
            fredowns=int(fredowns)
            self.data = self.data.reshape(self.fsamp//fredowns,fredowns,self.tsamp).mean(1)
            self.fsamp,self.tsamp = self.data.shape
        if tdowns:
            self.data = self.data.reshape(self.fsamp,self.tsamp//fredowns,fredowns).mean(2)
            self.fsamp,self.tsamp = self.data.shape
        if save:
            np.save('%s.npy'%self.basename,self.data)
        print('*The filename      : %s'%filename) 
        print('*The frequency band: %.3f ~ %.3f MHz'%(self.freq_lo,self.freq_hi))
        print('*--- channel number: %s'%self.nchan)
        print('*The obs MJD       : %s'%(self.mjd))
        print('*The obs duration  : %s s'%self.obs_t)
        print('*--- time sampling : %s'%self.tsamp)
    
    def read_dat(self,filename,remove_baseline=True,tdowns=False,fredowns=False,save=False):
        print('\033[1;32;40m READING FILE %s...\033[0m\n'%filename)
        try:
            self.basename = filename.replace('.dat','')
            hf = h5py.File(filename,'r')
            self.data = np.array(hf['data'])
        except:
            self.basename = filename.replace('.npy','')
            self.data = np.load(filename)
        self.tsamp,self.fsamp = self.data.shape
        if remove_baseline:
            #self.data = np.array(list(map(lambda i:self.data[i,:]-np.mean(self.data[i,:]),np.arange(self.tsamp).astype(int))))
            self.data = self.data-self.data.mean(axis=1, keepdims=True)
        self.raw_data = self.data.copy()
        print('test test test',self.raw_data.shape)
        self.nchan=self.fsamp
        ###
        self.freq_lo,self.freq_hi = 1000,1500
        self.df = (self.freq_hi-self.freq_lo)/self.nchan
        self.mjd = 'Uknown'
        self.obs_t = 0.2000
        if fredowns:
            fredowns=int(fredowns)
            self.data = self.data.reshape(self.fsamp//fredowns,fredowns,self.tsamp).mean(1)
            self.fsamp,self.tsamp = self.data.shape
        if tdowns:
            self.data = self.data.reshape(self.fsamp,self.tsamp//fredowns,fredowns).mean(2)
            self.fsamp,self.tsamp = self.data.shape
        if save:
            np.save('%s.npy'%self.basename,self.data)
        print('*The filename      : %s'%filename)
        print('*The frequency band: %.3f ~ %.3f MHz'%(self.freq_lo,self.freq_hi))
        print('*--- channel number: %s'%self.nchan)
        print('*The obs MJD       : %s'%(self.mjd))
        print('*The obs duration  : %s s'%self.obs_t)
        print('*--- time sampling : %s'%self.tsamp)

    def mask_rfi(self,level=8,threshold=1,scale=True,threshold_w=2,wavelet='db8',fill=0,rebuid=False,threshold_rfi=1,mask=1):
        print('\033[1;32;40m FLAGGING RFI...\033[0m\n')
        level=int(level)
        print('level',level,threshold_rfi)
        if scale==True:
           self.data=self.data-self.data.mean(axis=1, keepdims=True)
        if threshold_w:
           threshold_w=float(threshold_w)
        else:
           kur=kurtosis(self.data.reshape(-1))
           threshold_w = kur**0.25
        if fill==0:
            replace = 0
        elif fill==1:        
            replace=np.median(self.data)
        else:
            replace=np.float(fill)
        co = pywt.wavedec2(self.data,wavelet, level=level)
        t00 = np.std(co[0])
        data_sumx = self.data.mean(0)
        print('threshold',threshold_w,threshold)
        if max(data_sumx) >= np.mean(data_sumx)+ 5*np.std(data_sumx):
           print('Pulse in it!')
           #co[0][np.where(abs(co[0])>threshold*t00)]=replace
           for i in range(1,level):
               tll=np.std(co[i][0])
               thh=np.std(co[i][2])
               co[i][1][np.where(abs(co[i][0])>threshold_w*np.std(co[i][0]))]=replace
               co[i][2][np.where(abs(co[i][2])>threshold_w*np.std(co[i][2]))]=replace
        else:
           co[0][np.where(abs(co[0])>threshold*t00)]=replace
           for i in range(1,level):
               tll=np.std(co[i][0])
               tlh=np.std(co[i][1])
               thh=np.std(co[i][2])
               co[i][0][np.where(abs(co[i][0])>threshold_w*np.std(co[i][0]))]=replace
               co[i][1][np.where(abs(co[i][1])>threshold_w*np.std(co[i][1]))]=replace
               co[i][2][np.where(abs(co[i][2])>threshold_w*np.std(co[i][2]))]=replace
        recom_data=pywt.waverec2(co,wavelet)
        res=self.data-recom_data
        self.t_rfi,self.f_rfi=np.where(abs(res)>np.mean(res)+threshold_rfi*np.std(res))
        print('*RFI ratio in time-frequency spectrum: %.2f%%'%(len(self.t_rfi)/(self.tsamp*self.fsamp)*100))
        null_data=np.zeros((self.fsamp,self.tsamp))
        null_data[self.t_rfi,self.f_rfi]=1
        bandpass=null_data.mean(1)
        sig,nos=smooth(bandpass)
        badch1=np.where(bandpass>np.median(bandpass)+3*np.std(bandpass))[0]
        badch2=np.where((nos>3*np.std(nos))|(nos<-3*np.std(nos)))[0]
        self.badch=np.unique(np.hstack((badch1,badch2)))
        print('*RFI channel number                  : %d/%d'%(len(self.badch),self.fsamp))
        self.null_data = null_data
        self.rfi_bandpass = bandpass
        self.rfi_smooth_bandpass = sig
        if rebuid==True:
           self.data = recom_data
        if mask==0:
            pass
        elif mask==1:
            self.data[self.t_rfi,self.f_rfi]=replace
            #self.data = recom_data
        elif mask==2:
            self.data[:,self.badch]=replace

    def detect_frb(self,step=20,remove_baseline=True,pulse_frac=0.2,threshold=60,eps=2,dens=1.5,zscore=2,**kwargs):
        print(step,threshold,dens)
        print('\033[1;32;40m DETECTING FRB IN SUB-BANDS...\033[0m\n')
        paras = kwargs
        data_cp = self.data.copy()
        #self.data_cp[self.t_rfi,self.f_rfi]=np.median(self.data)
        width=int(self.nchan/step)
        print('The smooth step width:%d.'%width)
        smooth_data=np.array(list(map(lambda i:data_cp[i:i+width,:].mean(0), np.arange(self.nchan-width).astype(int)))) ####smooothing the data with running mean 
        self.freq_new = np.linspace(self.freq_lo,self.freq_hi,smooth_data.shape[0])
        prd_iddata,fit_id,labels,burst_label = dbscan(smooth_data,threshold,pulse_frac,eps,dens,zscore)
        while len(burst_label)<=0:
              threshold = threshold - 5
              if threshold >=55:
                  print('Detcting no FRB signal!!! Setting the threshold = %s.'%threshold)
                  prd_iddata,fit_id,labels,burst_label = dbscan(smooth_data,threshold,pulse_frac,eps,dens,zscore)
                 # if len(burst_label)<=0:
                 #    prd_iddata,fit_id,labels,burst_label = dbscan(smooth_data,threshold,pulse_frac=1)
                 #    if len(burst_label)<=0:
                 #        print('NO FRB SIGNAL !!!')
              else:
                  print('NO FRB SIGNAL !!!') 
                  break 
        if 'top' in paras:
           top = int(paras['top'])
           if top <= len(burst_label):
               burst_label = burst_label[:top]
           else:
               print('The cluster numbers is less than %d !!!'%top)
               pass
        
        frb_freqid = sum([list(fit_id[:,0][labels==i]) for i in burst_label],[])
        if len(frb_freqid)>2:
            frb_chan_s = np.min(frb_freqid)
            frb_chan_e = np.max(frb_freqid)
        else:
            frb_chan_s = 0
            frb_chan_e = len(self.freq_new)   
        frb_chan=np.arange(frb_chan_s,frb_chan_e)
        self.frb_freq = self.freq_new[frb_chan.astype(int)]
        self.labels = labels
        self.burst_label = burst_label
        self.fit_id = fit_id 
        self.burst_label = burst_label
        self.smooth_data=smooth_data
        self.prd_iddata=prd_iddata
        #cluster_data = np.vstack((self.fit_id.T,self.labels)) 
        #np.save('%s_clustr_data'%self.basename,cluster_data)
        #print(cluster_data.shape)
        print('*The FRB signal band: %.2f~%.2f MHz'%(min(self.frb_freq),max(self.frb_freq)))

class plot_data(desffrb):           
    def __init__(self,desffrb_):
        self.raw_data = desffrb_.raw_data
        self.data = desffrb_.data
        self.freq_lo = desffrb_.freq_lo
        self.freq_hi = desffrb_.freq_hi
        self.mjd = desffrb_.mjd
        self.obs_t = desffrb_.obs_t
        self.basename = desffrb_.basename
        self.nchan = desffrb_.nchan
        self.fit_id = desffrb_.fit_id
        self.labels = desffrb_.labels
        self.burst_label =desffrb_.burst_label
        f_num,t_num = self.data.shape
        self.f = np.linspace(self.freq_lo,self.freq_hi,f_num)
        self.t = np.linspace(0,self.obs_t,t_num)
        ####rfi-flag#### 
        self.null_data = desffrb_.null_data
        self.badch = desffrb_.badch
        self.rfi_bandpass = desffrb_.rfi_bandpass
        self.rfi_smooth_bandpass = desffrb_.rfi_smooth_bandpass
        ####smoooth data####
        self.smooth_data = desffrb_.smooth_data
        self.prd_iddata = desffrb_.prd_iddata
        newf_num,newt_num = self.smooth_data.shape
        self.newf = np.linspace(self.freq_lo,self.freq_hi,newf_num)
        self.newt = np.linspace(0,self.obs_t,newt_num)
        self.frb_freq = desffrb_.frb_freq
        self.fit_id_t = self.newt[self.fit_id[:,1]]
        self.fit_id_f = self.newf[self.fit_id[:,0]]

    def plot_rawdata(self,display=True,save=False):
        print('\n\033[1;35;40m PLOTTING THE RAW DATA...\033[0m')
        plt.close('all')
        plt.figure(figsize=(10,10))
        ax1=plt.subplot2grid((4,2),(0,0),colspan=1,rowspan=1)
        ax2=plt.subplot2grid((4,2),(1,0),colspan=1,rowspan=3)
        ax3=plt.subplot2grid((4,2),(0,1),colspan=1,rowspan=1)
        ax4=plt.subplot2grid((4,2),(1,1),colspan=1,rowspan=3)
        ax2.pcolormesh(self.t,self.f,self.raw_data,vmin=0.2*np.min(self.raw_data),vmax=0.2*np.max(self.raw_data))
        ax1.plot(self.t,self.raw_data.mean(0),color='black',linewidth=0.7)
        data_comp = self.raw_data.reshape(256,16,4096).mean(1).reshape(256,256,16).mean(2)
        ax2.pcolormesh(self.t,self.f,self.raw_data,vmin=0.2*np.min(self.raw_data),vmax=0.2*np.max(self.raw_data))
        ax1.plot(self.t,self.raw_data.mean(0),color='black',linewidth=0.7)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlim(min(self.t),max(self.t))
        new_id = np.arange(0,4095,16).astype(int)
        #ax4.pcolormesh(self.t[new_id],self.f[new_id],data_comp,vmin=0.2*np.min(data_comp),vmax=0.2*np.max(data_comp))
        ax4.imshow(data_comp,vmin=0.5*np.min(data_comp),vmax=0.5*np.max(data_comp), extent=[0,0.2,1000,1500], aspect='auto',origin='lower')
        #ax4.imshow(data_comp,vmin=0.2*np.min(data_comp),vmax=0.2*np.max(data_comp), aspect='auto',origin='lower')
        ax3.plot(self.t[new_id],data_comp.mean(0),color='black',linewidth=0.7)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_xlim(min(self.t),max(self.t))
        #ax3.plot(self.raw_data.mean(1),self.f,color='black',linewidth=0.7)
        #ax3.set_xticks([])
        #ax3.set_xlim(min(self.raw_data.mean(1)),max(self.raw_data.mean(1)))
        #ax3.set_yticks([])
        ax2.set_xlabel('Obs Time (s)',fontsize=25)
        ax2.set_ylabel('Frequency (MHz)',fontsize=25)
        ax4.set_xlabel('Obs Time (s)',fontsize=25)
        ax4.set_yticks([])
        ax2.tick_params(labelsize=25)
        ax4.tick_params(labelsize=25)
        ax1.set_title('MJD:%s'%self.mjd,fontsize=25)
        plt.subplots_adjust(top=0.938,bottom=0.122,left=0.148,right=0.99,hspace=0,wspace=0.05)
        if save==True:
            plt.savefig('%s_raw_data.png'%self.basename)
            plt.close()
        if display==True:
            print('show the png')
            plt.show()
        else:
            pass
    
    def plot_rfi(self,display=True,save=False):
        print('\n\033[1;35;40m PLOTTING THE RFI-FLGGING DATA...\033[0m') 
        #'''
        plt.figure(figsize=(10,10))
        ax1=plt.subplot2grid((8,8),(0,0),rowspan=2,colspan=6)
        ax2=plt.subplot2grid((8,8),(2,0),rowspan=6,colspan=6)
        ax3=plt.subplot2grid((8,8),(2,6),rowspan=6,colspan=2)
        [ax2.axhline(i,color='green',alpha=0.02) for i in self.f[self.badch]]
        [ax3.axhline(i,color='green',alpha=0.1) for i in self.f[self.badch]]
        ax1.plot(self.t,self.null_data.mean(0),color='firebrick',linewidth=1)
        ax1.legend(framealpha=0)
        ax2.pcolormesh(self.t,self.f,self.null_data,cmap='binary')
        ax3.plot(self.rfi_bandpass,self.f,color='royalblue',linewidth=0.8)
        ax3.plot(self.rfi_smooth_bandpass,self.f,linewidth=0.8,color='black')
        ax3.plot(self.rfi_bandpass-self.rfi_smooth_bandpass,self.f,linewidth=0.8)
        ax3.set_ylim(min(self.f),max(self.f))
        ax1.set_xlim(min(self.t),max(self.t))    
        ax2.set_xlabel('Obs Time (s)',fontsize=25)
        ax2.set_ylabel('Frequency (MHz)',fontsize=25)
        ax1.set_xticks([])
        #ax1.set_yticks([])
        ax3.set_yticks([])
        #ax3.yaxis.tick_right()
        ax3.xaxis.tick_top()
        #ln = plt.gca()
        #ln.tick_params(axis='y', labelrotation = 270)
        ax1.tick_params(labelsize=25)
        ax2.tick_params(labelsize=25)
        ax3.tick_params(labelsize=25)
        ax3.set_xlabel('RFI Ratio',fontsize=25)
        ax1.set_ylabel('RFI Ratio',fontsize=25)
        plt.subplots_adjust(top=0.938,bottom=0.122,left=0.148,right=0.99,hspace=0,wspace=0)
        if save==True:
            plt.savefig('%s_rfi.png'%self.basename)
        if display==True:
            plt.show()
        else:
            pass
        #'''
        print('\n\033[1;35;40m PLOTTING THE RAW DATA & RFI-CLEANNING DATA...\033[0m')
        plt.figure(figsize=(12,9))
        ax11=plt.subplot2grid((4,2),(0,0),rowspan=1,colspan=1)
        ax12=plt.subplot2grid((4,2),(1,0),rowspan=3,colspan=1)
        ax21=plt.subplot2grid((4,2),(0,1),rowspan=1,colspan=1)
        ax22=plt.subplot2grid((4,2),(1,1),rowspan=3,colspan=1)
        data_comp = self.data.reshape(256,16,4096).mean(1).reshape(256,256,16).mean(2)
        #ax12.pcolormesh(self.t,self.f,self.data)
        ax12.imshow(self.data,vmin=0.1*np.min(self.data),vmax=0.1*np.max(self.data), extent=[0,0.2,1000,1500], aspect='auto',origin='lower',)
        ax22.imshow(data_comp,vmin=0.3*np.min(data_comp),vmax=0.3*np.max(data_comp), extent=[0,0.2,1000,1500], aspect='auto',origin='lower')
        ax11.plot(self.t,self.data.mean(0),color='black',linewidth=0.8)
        ax11.set_xlim(min(self.t),max(self.t))
        ax12.set_xlabel('Obs Time (s)',fontsize=25)
        ax12.set_ylabel('Frequency (MHz)',fontsize=25)
        ax11.set_xticks([])
        ax11.set_yticks([])
        #ax22.pcolormesh(self.t,self.f,self.data)
        new_id = np.arange(0,4095,16).astype(int)
        ax21.plot(self.t[new_id],data_comp.mean(0),color='black',linewidth=0.8)
        ax21.set_xlim(min(self.t),max(self.t))
        ax21.set_xticks([])
        ax21.set_yticks([])
        ax22.set_xlabel('Obs Time (s)',fontsize=25)
        #ax22.set_ylabel('Frequency (MHz)',fontsize=25)
        ax22.yaxis.tick_right()
        ax12.tick_params(labelsize=25)
        #ax12.tick_params(axis='y',labelrotation=90,verticalalignment = 'center')
        #ax22.tick_params(axis='y',labelrotation=90,va="center")
        ax22.tick_params(labelsize=25)
        plt.setp(ax12.get_yticklabels(), rotation=90, va='baseline')
        plt.setp(ax22.get_yticklabels(), rotation=270,va='baseline')
        plt.subplots_adjust(bottom=0.1,left=0.08,right=0.95,top=0.99,hspace=0.,wspace=0.05)
        if save==True:
            plt.savefig('%s_rfi_excised.png'%self.basename)
        if display==True:
            plt.show()
        else:
            pass
     
    def plot_smooth(self,display=True,save=False):
        print('\n\033[1;35;40m PLOTTING THE SMOOTHING DATA...\033[0m')
        plt.figure(figsize=(12,9))
        ax11=plt.subplot2grid((4,2),(0,0),rowspan=1,colspan=1)
        ax12=plt.subplot2grid((4,2),(1,0),rowspan=3,colspan=1)
        ax21=plt.subplot2grid((4,2),(0,1),rowspan=1,colspan=1)
        ax22=plt.subplot2grid((4,2),(1,1),rowspan=3,colspan=1)
        ax12.pcolormesh(self.newt,self.newf,self.smooth_data)
        ax11.plot(self.newt,self.smooth_data.mean(0),color='black',linewidth=0.8)
        ax11.set_xlim(min(self.newt),max(self.newt))
        ax12.set_xlabel('Obs Time (s)',fontsize=25)
        ax12.set_ylabel('Frequency (MHz)',fontsize=25)
        ax11.set_xticks([])
        ax11.set_yticks([])
        ax22.pcolormesh(self.newt,self.newf,self.prd_iddata,cmap='binary')
        ax21.plot(self.newt,self.prd_iddata.mean(0),color='black',linewidth=0.8)
        ax21.set_xlim(min(self.newt),max(self.newt))
        ax21.set_xticks([])
        ax21.set_yticks([])
        ax22.set_xlabel('Obs Time (s)',fontsize=25)
        #ax22.set_ylabel('Frequency (MHz)',fontsize=25)
        ax22.yaxis.tick_right()
        ax12.tick_params(labelsize=25)
        #ax12.tick_params(axis='y',labelrotation=90,verticalalignment = 'center')
        #ax22.tick_params(axis='y',labelrotation=90,va="center")
        ax22.tick_params(labelsize=25)
        plt.setp(ax12.get_yticklabels(), rotation=90, va='baseline')
        plt.setp(ax22.get_yticklabels(), rotation=270,va='baseline')
        plt.subplots_adjust(bottom=0.1,left=0.08,right=0.95,top=0.99,hspace=0.,wspace=0.05)
        if save==True:
            plt.savefig('%s_smooth_frb.png'%self.basename)
        if display==True:
            plt.show()
    
    def plot_cluster(self,display=True,save=False):
        print('\n\033[1;35;40m PLOTTING THE CLUSTERS ...\033[0m')
        fig=plt.figure(figsize=(12,9))
        params={'axes.labelsize': '20',
                  'xtick.labelsize':'20',
                 'ytick.labelsize':'20','legend.fontsize': '20','axes.titlesize':'20'}
        pylab.rcParams.update(params)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=min(self.labels), vmax=max(self.labels))
        ax11=plt.subplot2grid((2,2),(0,0))
        ax12=plt.subplot2grid((2,2),(0,1))
        ax21=plt.subplot2grid((2,2),(1,0))
        ax22=plt.subplot2grid((2,2),(1,1))
        ax11.set_title('Smoothed Data')
        ax12.set_title('Binary Data')
        ax21.set_title('Cluster data')
        ax22.set_title('Burst Data')
        c1=ax11.pcolormesh(self.newt,self.newf,self.smooth_data)
        #cb1=plt.colorbar(c1,ax=ax11)
        #cb1.set_label('Relative Power',rotation=-90,pad=10)
        alpha = len(self.fit_id_f)/(len(self.newt)*len(self.newf))
        print(alpha)
        ax12.scatter(self.fit_id_t,self.fit_id_f,color='black',s=0.0001/alpha,alpha=0.2)
        #ax12.pcolormesh(self.newt,self.newf,self.prd_iddata,cmap='binary')
        c2=ax21.scatter(self.fit_id_t,self.fit_id_f,c = self.labels,s=0.0001, alpha=0.2,cmap= "jet")
        #cb2=plt.colorbar(c2,ax=ax21)
        #cb2.set_label('Cluster Labels',rotation=90)
        #ax11.set_yticklabels(ax11.get_yticks(),rotation=90, va='top')
        #ax21.set_yticklabels(ax21.get_yticks(),rotation=90, va='bottom')
        ax11.set_ylabel('Frequency (MHz)')
        ax21.set_ylabel('Frequency (MHz)')
        ax21.set_xlabel('Time (s)')
        ax22.set_xlabel('Time (s)')
        ax11.set_xticks([])
        ax12.set_xticks([])
        if len(self.burst_label)>=1: 
            save_figname='%s_cluster_yes'%self.basename
            for i in self.burst_label:
                ax22.scatter(self.fit_id_t[self.labels==i],self.fit_id_f[self.labels==i],s=0.0001/alpha,c = cmap(norm(i)),alpha=0.4)
        else:
            save_figname='%s_cluster_no'%self.basename
            ax22.scatter(self.fit_id_t[self.labels==-1],self.fit_id_f[self.labels==-1],s=0.0001/alpha,c = cmap(norm(0)),alpha=0.4)
        ax12.set_xlim(min(self.newt),max(self.newt))
        ax12.set_ylim(min(self.newf),max(self.newf))
        ax21.set_xlim(min(self.newt),max(self.newt))
        ax21.set_ylim(min(self.newf),max(self.newf))
        ax22.set_xlim(min(self.newt),max(self.newt))
        ax22.set_ylim(min(self.newf),max(self.newf))
        ax12.set_yticks([])
        ax22.set_yticks([])
        plt.subplots_adjust(bottom=0.1,left=0.1,right=0.975,top=0.95,hspace=0.1,wspace=0.1)
        if save==True:
            plt.savefig('%s.png'%save_figname)
            plt.close('all')
        if display==True:
            plt.show() 

    def plot_frb(self,display=True,save=False):
        print('\n\033[1;35;40m PLOTTING THE FRB SIGNAL...\033[0m')
        plt.figure(figsize=(12,9))
        params={'axes.labelsize': '25',
                  'xtick.labelsize':'25',
                 'ytick.labelsize':'25'}
        pylab.rcParams.update(params)
        
        ax11=plt.subplot2grid((4,2),(0,0),rowspan=1,colspan=1)
        ax12=plt.subplot2grid((4,2),(1,0),rowspan=3,colspan=1)
        ax21=plt.subplot2grid((4,2),(0,1),rowspan=1,colspan=1)
        ax22=plt.subplot2grid((4,2),(1,1),rowspan=3,colspan=1)
        ax12.pcolormesh(self.t,self.f,self.data)
        ax11.plot(self.t,self.data.mean(0),color='black',linewidth=0.8)
        ax11.set_xlim(min(self.t),max(self.t))
        ax12.set_xlabel('Obs Time (s)',fontsize=25)
        ax12.set_ylabel('Frequency (MHz)',fontsize=25)
        ax11.set_xticks([])
        ax11.set_yticks([])
        data1=self.data
        #zapchan = np.setdiff1d(np.arange(self.nchan).astype(int),self.frb_chan.astype(int))
        #np.savetxt('frb_chan.txt',self.frb_chan.astype(int))
        #np.savetxt('nchan.txt',np.arange(self.nchan).astype(int))
        zapchan = np.arange(self.nchan)[(self.f>=max(self.frb_freq))|(self.f<=min(self.frb_freq))]
        data1[zapchan]=np.nan
        ax22.pcolormesh(self.t,self.f,data1)
        ax21.plot(self.t,np.nanmean(data1,0),color='grey',linewidth=0.8)
        ax21.plot(self.t,running_smooth(np.nanmean(data1,0)),color='black',linewidth=0.8)
        ax21.set_xlim(min(self.t),max(self.t))
        ax21.set_xticks([])
        ax21.set_yticks([])
        ax22.set_xlabel('Obs Time (s)',fontsize=25)
        #ax22.set_ylabel('Frequency (MHz)',fontsize=25)
        ax22.yaxis.tick_right()
        ax12.tick_params(labelsize=25)
        #ax12.tick_params(axis='y',labelrotation=90,verticalalignment = 'center')
        #ax22.tick_params(axis='y',labelrotation=90,va="center")
        ax22.tick_params(labelsize=25)
        plt.setp(ax12.get_yticklabels(), rotation=90, va='baseline')
        plt.setp(ax22.get_yticklabels(), rotation=270,va='baseline')
        plt.subplots_adjust(bottom=0.1,left=0.08,right=0.95,top=0.99,hspace=0.,wspace=0.05)
        if save==True:
            plt.savefig('%s_real_frb.png'%self.basename)
        if display==True:
            plt.show()
    
