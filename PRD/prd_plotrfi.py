import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
filename=sys.argv[1]
basename=filename.replace('\n','').replace('.fits','')

print('\n\033[1;32;40m Plot rfi mask data from %s_rfifind.mask...\033[0m\n'%basename)
readm = open('%s_rfifind.mask'%basename)
time_sig,freq_sig, MJD,dtint,lofreq,df = np.fromfile(readm, dtype=np.float64, count=6)
nchan, nint, ptsperint = np.fromfile(readm, dtype=np.int32, count=3)
nzap_f = np.fromfile(readm, dtype=np.int32, count=1)[0]
mask_zap_chans=np.fromfile(readm, dtype=np.int32, count=nzap_f)
nzap_t = np.fromfile(readm, dtype=np.int32, count=1)[0]
mask_zap_ints = np.fromfile(readm, dtype=np.int32, count=nzap_t)
nzap_per_int = np.fromfile(readm, dtype=np.int32, count=nint)

rfi_ch_num=sum(nzap_per_int)
rfi_ch_arr=np.fromfile(readm, dtype=np.int32,count=rfi_ch_num)
rfi_ratio = rfi_ch_num/(nchan*nint)
print('RFI ratio:%.2f %%'%(100*rfi_ratio))

null_data=np.zeros((nint,nchan))
a,b=null_data.shape
x=[np.zeros(nzap_per_int[i])+i for i in range(nint)]
x=np.array(list(itertools.chain(*x))).astype(int)
y=np.array(rfi_ch_arr).astype(int)
null_data[x,y]=1
x_time=np.arange(0,dtint*nint,dtint)
y_fre=np.arange(lofreq,lofreq+nchan*df,df)
ax1=plt.subplot2grid((8,8),(0,0),rowspan=2,colspan=6)
ax2=plt.subplot2grid((8,8),(2,0),rowspan=6,colspan=6)
ax3=plt.subplot2grid((8,8),(2,6),rowspan=6,colspan=2)
ax1.plot(null_data.mean(1),color='firebrick',linewidth=1)
#ax1.plot(null_data.mean(1),color='grey',linewidth=1,label='Reconsturcted Centering-data')
ax1.legend(framealpha=0)
ax1.set_xlim(0,a)
          #ax2.imshow(new.T,aspect='auto',origin='lower',vmin=0.01*np.min(data),vmax=0.005*np.max(data),cmap='rainbow')
ax2.pcolormesh(x_time,y_fre,null_data.T,cmap='binary')
ax3.plot(null_data.mean(0),np.arange(b),color='royalblue',linewidth=0.8)
ax3.set_ylim(0,b)
ax2.set_xlabel('Time')
ax2.set_ylabel('Frequency')
ax1.set_xticks([])
ax1.set_yticks([])
#ax3.set_yticks([])
ax3.yaxis.tick_right()
ax3.set_xticks([])
ln = plt.gca()
ln.tick_params(axis='y', labelrotation = 270)
plt.subplots_adjust(
           top=0.995,
           bottom=0.085,
           left=0.1,
           right=0.96,
           hspace=0.0,
           wspace=0.0)
plt.savefig('%s_rfi-mask.png'%basename,dpi=800)
