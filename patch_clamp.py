# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:32:35 2023

@author: kh19883
"""


import numpy as np
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import butter, filtfilt, find_peaks
import os, sys




def butterworth_filter(x, sr, cutoff, order):
    # constract the filter
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # apply the filter to the data
    filtered_data = filtfilt(b, a, x)
    return filtered_data
    
    
    


def IClamp_baseline_end(expNum):
        ### generate a list of file names
        exp_path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents\PhD Neural Dynamics\ARC-Bursting\Patch Data\Lyons\Whole Cell"
        experiment_list=[]     
        for filename in os.listdir(exp_path):
            if filename.endswith(".abf"):
                experiment_list.append(filename[0:-4])
            else:
                continue
                
        end_times = { '01-02-22 S1C1': -1, '01-02-22 S3C1': -1, '01-02-22 S4C1': -1,
                '01-02-22 S5C1': -1, '02-02-22 S1C1': -1, '02-02-22 S2C1': -1,
                '02-02-22 S3C1': -1, '02-02-22 S4C1': -1, '03-02-22 S1C1': 210,
                '03-02-22 S2C1': 260, '03-02-22 S3C1': 215, '03-02-22 S4C1': 185,
                '08-02-22 S1C1': 280, '09-02-22 S1C1': -1, '09-02-22 S2C1': 225,
                '20-01-22 S2C2': -1, '20-01-22 S3C1': -1, '22-02-22 S2C1': 180,
                '23-02-22 S2C1': 190, '23-02-22 S3C1': 230, '23-02-22 S5C1': 150,
                '24-01-22 S1C1': -1, '24-01-22 S3C1': -1, '24-02-22 S2C1': 280,
                '24-02-22 S5C1': 270, '26-01-22 S2C1': -1, '26-01-22 S5C1': -1,
                '26-01-22 S6C1': -1, '27-01-22 S1C1': -1,
               }
        return end_times[experiment_list[expNum]]


    
# method to calculate the dominant frequecy of a signal (raw or filtered)
def fourier_freq(x, sr):
    # define the array, x, to be analyzied 
    x = x - np.mean(x)

    # apply fourier transform to the time series
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1/sr)

    # Find the dominant frequency and return it
    dominant_freq = freqs[np.argmax(np.abs(X))]
    return dominant_freq
                
        


class PatchRec():
     
    # constructor method generated the time series that defines the patch clamp recording 
    def __init__(self, expNum):
        
        self.expNum = expNum;
        
        exp_path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents\PhD Neural Dynamics\ARC-Bursting\Patch Data\Lyons\Whole Cell"

        ### generate a list of file names
        experiment_list=[]     
        for filename in os.listdir(exp_path):
            if filename.endswith(".abf"):
                experiment_list.append(filename[0:-4])
            else:
                continue
      
        # load the ABF
        abf = pyabf.ABF(exp_path+"\\"+experiment_list[expNum]+".abf")
        abf.setSweep(0)
        
        # define cutt-off time for baseline recording
        tEnd =  IClamp_baseline_end(expNum)
        if tEnd != -1:
            tEnd = tEnd * abf.sampleRate
        
        # output of the constructor
        self.sampleRate = abf.sampleRate   # 10,000 Hz
        self.abf  = abf
        self.time = abf.sweepX[0:tEnd]
        self.data = abf.sweepY[0:tEnd]
        
        
        
        
    # method to simpliy plot the voltage trace of the recording   
    def plot(self, ax=None):
        # create axes, if none already given
        if ax==None:
            fig, ax = plt.subplots()
              
        # plot data vs. time
        ax.set_ylabel(self.abf.sweepLabelY)
        ax.set_xlabel(self.abf.sweepLabelX)    
        ax.set_xlim(self.time[0], self.time[-1])
        ax.plot(self.time, self.data)
        return ax
    
    
    
    # method to filter the recording without introducing a phase-lag
    def filt(self, cutoff=0.5, order=2):
        sr = self.sampleRate
        filtered_data = butterworth_filter(self.data, sr=sr, cutoff=cutoff, order=order) 
        return filtered_data
    
    
    
    # method to calculate the dominant frequecy of a signal (raw or filtered)
    def fourier_freq(self, slow_only=False):
        # define the array, x, to be analyzied 
        if slow_only==False:
            x = self.data
        if slow_only==True:
            x = butterworth_filter(self.data, sr=self.sampleRate, cutoff=0.5, order=2) 
        x = x - np.mean(x)

        # apply fourier transform to the time series
        sf = self.sampleRate
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x), d=1/sf)

        # Find the dominant frequency and return it
        dominant_freq = freqs[np.argmax(np.abs(X))]
        return dominant_freq
    
    
    def isi(self, distance=30, ib_isi_threshold=5):
        # find the spike times
        spks,_ = find_peaks(self.data, height=0, distance=distance)
        stimes = self.time[spks]
        
        # calculate ISI distribution
        isi = np.diff(stimes)
        
        # define inter-burst ISI as any ISI less than 5s
        ib_isi = isi[isi<ib_isi_threshold]
        bb_isi = isi[isi>=ib_isi_threshold] # between-burst ISI 
        
        # 
        return isi, ib_isi, bb_isi
    
    
    
    def slow_min_max(self):
        # filter the signal 
        sr = self.sampleRate
        pfilt = butterworth_filter(self.data, sr=sr, cutoff=0.1, order=2)   # 0.1 manually checked and confirmed to be okay
        x = pfilt - np.mean(pfilt)
        max_amp = max(x) - min(x)
    
        # find slow frequency to constrain peak_finding
        slow_f = fourier_freq(x, sr=self.sampleRate)
    
        # distance and prominence inputs
        dist_threshold = 0.1  * abs(round(1/slow_f * self.sampleRate))
        prom_threshold = 0.05 * max_amp
    
        # find peaks of the slow signal
        max_idx,_ = find_peaks(x, height=0, distance=dist_threshold, prominence=prom_threshold)
        max_t = self.time[max_idx]
        max_v = pfilt[max_idx]
    
        # find mins of the slow signal
        min_idx,_ = find_peaks(-x, height=0, distance=dist_threshold, prominence=prom_threshold)
        min_t = self.time[min_idx]
        min_v = pfilt[min_idx]
        
        return max_t, max_v, min_t, min_v
    
      