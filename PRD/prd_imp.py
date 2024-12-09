import numpy as np
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PyAstronomy import pyasl
import heapq
import os
import pandas as pd
import sys
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
import heapq
import pandas as pd
from datetime import datetime, timedelta
import astropy.io.fits as pyfits
from decimal import Decimal
import ephem
import itertools
from itertools import groupby
from itertools import chain
import pywt
import math
from scipy import interpolate
from prd_tools import *

class Data_pro:
     def __init__(self,fitsname):
         self.fitsname=fitsname
         self.basename = fitsname.replace('\n','').replace('.fits','')
     
     def readfile(self,pola='I',downsam=32):
         print("\n\033[1;32;40m Reading data '%s'...\033[0m\n"%self.fitsname)
#         try:
         secperday = 3600 * 24
         hdulist = pyfits.open(self.fitsname)
         hdu0 = hdulist[0]
         hdu1 = hdulist[1]
         #data1 = hdu0.data['data']
         data1 = hdu1.data['DATA']
         self.tsamp = hdu1.header['TBIN']
         self.ra = hdu0.header['RA']
         self.dec = hdu0.header['DEC']
         obsfreq = hdu0.header['OBSFREQ']
         self.obsnchan = hdu0.header['OBSNCHAN']
         obsbw = hdu0.header['OBSBW']
         self.fmin = obsfreq - obsbw/2.
         self.fmax = obsfreq + obsbw/2.
         self.df = hdu1.header['CHAN_BW']
         subintoffset = hdu1.header['NSUBOFFS']
         samppersubint = int(hdu1.header['NSBLK'])
         tstart = "%.18f" % (Decimal(hdu0.header['STT_IMJD']) + Decimal(hdu0.header['STT_SMJD'] + self.tsamp * samppersubint * subintoffset )/secperday )
         self.jd=float(tstart)+2400000.5
         date=ephem.julian_date('1899/12/31 12:00:00')
         djd=self.jd-date
         str1=ephem.Date(djd).datetime()
         str2=str1.strftime('%Y-%m-%d %H:%M:%S')
         if len(data1.shape)>2:
             self.datai=(data1[:,:,0,:,:]+data1[:,:,1,:,:]).squeeze().reshape((-1,self.obsnchan))
             self.dataq=(data1[:,:,0,:,:]-data1[:,:,1,:,:]).squeeze().reshape((-1,self.obsnchan))
             self.datau=2*data1[:,:,2,:,:].squeeze().reshape((-1,self.obsnchan))
             self.datav=2*data1[:,:,3,:,:].squeeze().reshape((-1,self.obsnchan))
             if pola=='I' or pola==False:
                print("### Loading data with polarization 'I'! ###")
                data = self.datai
             elif pola=='Q':
                print("### Loading data with polarization 'Q'! ###")
                data = self.dataq
             elif pola=='U':
                print("### Loading data with polarization 'U'! ###")
                data = self.datau
             elif pola=='V':
                print("### Loading data with polarization 'V'! ###")
                data = self.datav
             elif pola=='all':
                print("### Loading data with all polarizations: I,Q,U,V ! ###")
                data=np.vstack((self.datai,self.dataq,self.datau,self.datav)).reshape(4,*self.datai.shape)
             else:
                print("### WRONG INPUT of pola! The pola should be 'I','Q','U','V','all'!!! ###")
                os.exit(0)
         else:
              print('### Data is not recorded by 4 polarizations!!!')
              data = data1.squeeze().reshape((-1,self.obsnchan))
         dim = data.shape
         self.dim = np.array(dim)
         self.t_total = self.tsamp*dim[1]
         self.start=str2
         print('**********************Observation Information***************************')
         print('   File            : %s'%(self.fitsname))
         print('   RA & DEC        : %s & %s'%(self.ra,self.dec))
         print('   Start Time      : %s'%(str2))
         print('   Duration Time   : %s s'%(self.t_total))
         print('   Frequency       : %s~%s MHz'%(self.fmin,self.fmax))
         print('   Resolution      : %s us (*%d),%s MHz (*%d).'%(self.tsamp, samppersubint*subintoffset,self.df,self.dim[-1]))
         print('   Raw Data Shape  : ',self.dim)
         print('**********************Observation Information***************************')
#         except:
#             print('Something is wrong with this file!!!')

     def exrfi(self,time=0.1,threshold=0.1,level=5,wavelet='db10'):
          print('\n\033[1;32;40m Detecting RFI Channels by using 2D-WAVELET...\033[0m\n') 
          print('\n~~~~~~~~ rfifind from PRESTO!!! ~~~~~~~')
          data_grid=int(time/self.tsamp)
          data_grid=int(2**(np.round(np.log2(data_grid))))
          data = self.datai.reshape(self.dim[0]//data_grid,data_grid,self.dim[1]).mean(1)        
          #print(data.shape,data_grid)
          data=np.array(list(map(lambda i:data[:,i]-np.mean(data[:,i]),np.arange(self.dim[1]).astype(int)))).T
          #print(data.shape)
          a,b=data.shape
          rfifind_t=self.tsamp*data_grid
          #print(rfifind_t,self.tsamp)
          os.system('rfifind -o %s -time %f %s'%(self.basename,rfifind_t,self.fitsname))
          co = pywt.wavedec2(data,wavelet, level=level)
          wave_threshold=0.1
          for i in range(1,level):
              co[i][0][np.where((co[i][0]>wave_threshold*np.std(co[i][0]))|(co[i][0]<-wave_threshold*np.std(co[i][0])))]=0
              co[i][1][np.where((co[i][1]>wave_threshold*np.std(co[i][1]))|(co[i][1]<-wave_threshold*np.std(co[i][1])))]=0
              #co[i][2][np.where((co[i][2]>wave_threshold*np.std(co[i][2]))|(co[i][2]<-wave_threshold*np.std(co[i][2])))]=0
          new=pywt.waverec2(co,wavelet)
          re=data-new
          x,y=np.where((re>np.mean(re)+threshold*np.std(re))|(re<np.mean(re)-threshold*np.std(re)))
          print('RFI ratio',len(x)/(a*b))
          break_id=np.arange(len(y)-1)[np.diff(x)>0]+1
          rfi_arr=np.split(y,break_id)
          rfi_arr_num=[len(rfi_arr[i]) for i in range(a)]
          rfi_arr_min=rfi_arr[np.argmin(rfi_arr_num)]
          readm = open('%s_rfifind.mask'%self.basename)
          time_sig,freq_sig, MJD,dtint,lofreq,df = np.fromfile(readm, dtype=np.float64, count=6)
          nchan, nint, ptsperint = np.fromfile(readm, dtype=np.int32, count=3)
          nzap_f = np.fromfile(readm, dtype=np.int32, count=1)[0]
          mask_zap_chans=np.fromfile(readm, dtype=np.int32, count=nzap_f)
          #print('PRESTO zap chan number:',len(mask_zap_chans),mask_zap_chans)
          #print('Wavelet zap chan number:',len(rfi_arr_min),rfi_arr_min)
          #print('Difference between wavelet & PRESTO:',len(np.setdiff1d(mask_zap_chans,rfi_arr_min)))
          mask_zap_chans=np.union1d(mask_zap_chans,rfi_arr_min)
          #print('The total zap chan number:',len(mask_zap_chans))
          nzap_t = np.fromfile(readm, dtype=np.int32, count=1)[0]
          mask_zap_ints = np.fromfile(readm, dtype=np.int32, count=nzap_t)
          nzap_per_int = np.fromfile(readm, dtype=np.int32, count=nint)
          new_rfi_perint=np.array([0])
          new_nzap_per_int=[]
          for i in range(a):
              rfifind_grid=np.fromfile(readm, dtype=np.int32,count=nzap_per_int[i])
              new_grid = np.union1d(rfi_arr[i],rfifind_grid)
              #print(i,'PRESTO:%s; Wavelet:%s; Common:%s'%(len(rfifind_grid),len(rfi_arr[i]),len(new_grid)))
              new_nzap_per_int.append(len(new_grid))
              new_rfi_perint=np.hstack((new_rfi_perint,new_grid))
          new_nzap_per_int=np.array(new_nzap_per_int)
          new_rfi_perint=new_rfi_perint[1:]
          writem = open('%s_rfifind.mask'%self.basename,'wb+')
          writem.write(time_sig)
          writem.write(freq_sig)
          writem.write(MJD)
          writem.write(dtint)
          writem.write(lofreq)
          writem.write(df)
          writem.write(nchan)
          writem.write(nint)
          writem.write(ptsperint)
          writem.write(np.int32(len(mask_zap_chans)))
          writem.write(mask_zap_chans.astype(np.int32))
          writem.write(np.int32(nzap_t))
          writem.write(np.array(new_nzap_per_int).astype(np.int32))
          writem.write(new_rfi_perint.astype(np.int32))
          writem.close()
