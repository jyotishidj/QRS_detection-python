#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:27:35 2019

@author: debasish
"""

def qrs_detection(ecg,Fs):
    
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    
    
     #=============================================================Function Declarations==============================================================#   
    
    def sthresh_update(spki,threshi1,npki,threshi2):
        
        spki=0.125*isig[i]+0.875*spki
        threshi1=npki+0.25*(spki-npki)
        threshi2=0.3*threshi1
        return(spki,threshi1,threshi2)
    
    
    def nthresh_update(spki,threshi1,npki,threshi2):
        
        npki=0.125*isig[i]+0.875*npki
        threshi1=npki+0.25*(spki-npki)
        threshi2=0.3*threshi1
        return(npki,threshi1,threshi2)
        
        
    def RR_update():
        
        if np.size(RRinterval1)<8:
            RRaverage1=np.sum(RRinterval1)/np.size(RRinterval1)
        else:
            RRaverage1=np.sum(RRinterval1[np.size(RRinterval1)-7:])/8
            
        if np.size(RRinterval2)<8:
            RRaverage2=np.sum(RRinterval2)/np.size(RRinterval2)
        else:
            RRaverage2=np.sum(RRinterval2[np.size(RRinterval2)-7:])/8
            
        RR_low_limit=round(0.92*RRaverage2)
        RR_high_limit=round(1.16*RRaverage2)
        RR_missed_limit=round(1.66*RRaverage2)
        
        return(RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)
        
    def check_T(i,Fs):
        
        slope1=np.average(np.diff(isig[i-np.round(round(0.075*Fs)):i]))
        slope2=np.average(np.diff(isig[beat_loc[-1]-np.round(round(0.075*Fs)):beat_loc[-1]]))
        if slope1<0.5*slope2 and i<=int(0.36*Fs)+beat_loc[-1]:
            return(True)
        else:
            return(False)
        
    def search_back(threshi2):
        
        val=-2
        for k in n_loc:
            if isig[k]>=threshi2 and isig[k]>val:
                val=isig[k]
                index=k
                
        return index
        
    def peak_cond(loc,crange):
        
        val=(np.sum([x for x in np.sign(isigd[loc:loc+crange]) if x>0])-np.sum([x for x in np.sign(isigd[loc-crange:loc]) if x<0]))/(2*crange)
        if val<=0.15:
            return(True)
        else:
            return(False)
        
        
    
    #---------------Band Pass filtering--------------#
    b,a=signal.butter(3,[12/Fs,24/Fs],'bandpass')
    fsig=signal.filtfilt(b,a,ecg)
    
    #    plt.figure(1)
    #    plt.subplot(411)
    #    plt.plot(fsig)
    
    #---------------derivative-----------------------#
    b=np.array([1,2,0,-2,-1])/8
    a=1
    dsig=signal.filtfilt(b,a,fsig)
    
    #    plt.subplot(412)
    #    plt.plot(dsig)
    #----------------squaring------------------------#
    ssig1=np.power(dsig,2)
    ssig=ssig1[int(0.30*Fs):] #First 300 ms data is neglected
    
    #    plt.subplot(413)
    #    plt.plot(ssig)
    
    #----------------integration---------------------#
    window1= int(0.150*Fs)
    isig1=ssig
    for i in range(1,window1):
        isig1=isig1+np.append(ssig[i:],np.zeros(i))   
    isig1=isig1/window1
    isig1=np.power(isig1,2)
    isig1=np.round(isig1/(2*np.max(isig1)),10)
    
    window2= int(0.050*Fs)
    isig=isig1
    for i in range(1,window2):
        isig=isig+np.append(isig1[i:],np.zeros(i))   
    isig=isig/window2
    
    delay=(window1+window2)/2
    
    #    plt.subplot(414)
    #    plt.plot(isig)
    
    #-----------------Fiducial Point detection------------------#
    
    #------------------training phase 1---------------------------#
    
    #for integrated signal
    npki=np.average(isig[:2*Fs])*0.75
    spki=0.8*np.amax(isig[:2*Fs])
    threshi1=npki+0.25*(spki-npki)
    threshi2=0.3*threshi1
    
    #for filtered signal
    npkf=np.average(fsig[:2*Fs])*0.75
    spkf=0.8*np.amax(fsig[:2*Fs])
    threshf1=npkf+0.25*(spkf-npkf)
    threshf2=0.3*threshf1
    
     #------------------training phase 2---------------------------#
    
    isigd=np.diff(isig)
    beat_loc=[np.argmax(isig[:2*Fs])]
    number_beats=1
    crange=int(0.01*Fs)
    if beat_loc[-1]-int(0.2*Fs)>crange:
        i=crange+1
    else:
        i=beat_loc[-1]+int(0.2*Fs)
    RRinterval1=[]
    RRinterval2=[]
    
    while (number_beats<2 and i<np.size(isigd)):
        
        if i<beat_loc[-1]:
            while i<beat_loc[0]-5:
                if isigd[i]<0 and isigd[i-1]>0 and peak_cond(i,crange):
                    if isig[i]>=threshi1:
                        
                        i=i+np.argmax(isig[i:i+int(0.2*Fs)])
                        beat_loc=np.array([i,beat_loc[0]])
                        number_beats=number_beats+1
                        #updating threshold parameters
                        (spki,threshi1,threshi2)=sthresh_update(spki,threshi1,npki,threshi2)
                        
                        #updating RR interval
                        RRinterval1=np.append(RRinterval1,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                        RRinterval2=np.append(RRinterval2,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                        (RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)=RR_update()
                        
                        i=beat_loc[-1]+int(0.2*Fs)
                    else:
                        (npki,threshi1,threshi2)=nthresh_update(spki,threshi1,npki,threshi2)
                i=i+1
                
                if i==beat_loc[-1]-5:
                    i=beat_loc[-1]+int(0.2*Fs)
                    break
                        
        else:
            if isigd[i]<0 and isigd[i-1]>0 and peak_cond(i,crange):
                if isig[i]>=threshi1:
                    
                    i=i+np.argmax(isig[i:i+int(0.2*Fs)])
                    beat_loc=np.append(beat_loc,[i])
                    number_beats=number_beats+1
                    #updating threshold parameters
                    (spki,threshi1,threshi2)=sthresh_update(spki,threshi1,npki,threshi2)
                    
                    #updating RR interval
                    RRinterval1=np.append(RRinterval1,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                    RRinterval2=np.append(RRinterval2,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                    (RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)=RR_update()
                        
                    i=beat_loc[-1]+int(0.2*Fs)
                    
                else:
                    (npki,threshi1,threshi2)=nthresh_update(spki,threshi1,npki,threshi2)
                
        i=i+1
        
    if np.size(beat_loc)==1:
        
        #print('recheck')
        beat_loc1=np.argmax(fsig[:int(1.8*Fs)])
        beat_loc2=np.argmax(fsig[(beat_loc1+int(0.2*Fs)):(beat_loc1+int(1.7*Fs))])+beat_loc1+int(0.2*Fs)
        i=np.argmax(isig[beat_loc1+int(0.2*Fs):beat_loc2-int(0.2*Fs)])+beat_loc1+int(0.2*Fs)
        
        if isigd[i]<0 and isigd[i-1]>0 and peak_cond(i,crange) and (isig[i]>0.3*(isig[beat_loc1]+isig[beat_loc2])):
            beat_loc2=i
            
        
        number_beats=2
        i=beat_loc2+int(0.2*Fs)
        beat_loc=np.array([beat_loc1,beat_loc2])
        RRinterval1=[]
        RRinterval2=[]
        
        (spki,threshi1,threshi2)=sthresh_update(spki,threshi1,npki,threshi2)
        
        #updating RR interval
        RRinterval1=np.append(RRinterval1,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
        RRinterval2=np.append(RRinterval2,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
        (RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)=RR_update()
    
    
    #-----------------------------------------------------detection phase--------------------------------------------------#
    n_loc=[]
    #error=0
    
    while (i<np.size(isigd)):
        if isigd[i]<0 and isigd[i-1]>0 and peak_cond(i,crange):
            if (isig[i]>=threshi1) and (~check_T(i,Fs)):
                
                i=i+np.argmax(isig[i:i+int(0.2*Fs)])
                beat_loc=np.append(beat_loc,[i])
                number_beats=number_beats+1
                #updating threshold parameters
                (spki,threshi1,threshi2)=sthresh_update(spki,threshi1,npki,threshi2)
                n_loc=[]
                
                #updating RR interval
                RRinterval1=np.append(RRinterval1,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                if (RR_high_limit>=[beat_loc[number_beats-1]-beat_loc[number_beats-2]]>=RR_low_limit):
                    RRinterval2=np.append(RRinterval2,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                (RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)=RR_update()
                    
                i=i+int(0.2*Fs)
                
            else:
                
                n_loc.append(i)
                (npki,threshi1,threshi2)=nthresh_update(spki,threshi1,npki,threshi2)
                
        if i>RR_missed_limit+beat_loc[number_beats-1]:
            try:
                beat_loc=np.append(beat_loc,search_back(threshi2))
                number_beats=number_beats+1
                #updating threshold parameters
                (spki,threshi1,threshi2)=sthresh_update(spki,threshi1,npki,threshi2)
                n_loc=[]
            
                #updating RR interval
                RRinterval1=np.append(RRinterval1,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                if (RR_high_limit>=[beat_loc[number_beats-1]-beat_loc[number_beats-2]]>=RR_low_limit):
                    RRinterval2=np.append(RRinterval2,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                (RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)=RR_update()
                
                i=i+int(0.2*Fs)
            
            except :
                #error=error+1
                #print('Error occured:',error)
                if np.size(n_loc)==0:
                    beat_loc=np.append(beat_loc,int(beat_loc[-1]+int(RRaverage2*1.1)))
                else:
                    beat_loc=np.append(beat_loc,n_loc[np.argmax(isig[n_loc])])
                number_beats=number_beats+1
                #updating threshold parameters
                (spki,threshi1,threshi2)=sthresh_update(spki,threshi1,npki,threshi2)
                n_loc=[]
                
                #updating RR interval
                RRinterval1=np.append(RRinterval1,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                if (RR_high_limit>=[beat_loc[number_beats-1]-beat_loc[number_beats-2]]>=RR_low_limit):
                    RRinterval2=np.append(RRinterval2,[beat_loc[number_beats-1]-beat_loc[number_beats-2]])
                (RRaverage1,RRaverage2,RR_low_limit,RR_high_limit,RR_missed_limit)=RR_update()
                
                i=i+int(0.2*Fs)
        i=i+1
        
    beat_loc=beat_loc+int(delay)+int(0.30*Fs)
    
    return(beat_loc,RRinterval1)