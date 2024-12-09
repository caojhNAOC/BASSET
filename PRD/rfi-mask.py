import prd_imp as prd
import sys,getopt
import argparse
parser = argparse.ArgumentParser()
parser.despeciption='Please enter at least the filename'
parser.add_argument("-file","--Fitsfile",help="please input the flename",type=str)
parser.add_argument("-time", "--inputA",help="This is the time for rfifind (in sec)",type=float,default="0.5")
parser.add_argument("-sigma", "--inputB",help="This is the threshold for masking RFI data grid.",type=float,default="0.5")

args=parser.parse_args()

#filename=sys.argv[1]
p=prd.Data_pro(args.Fitsfile)
#p.load('J1518+4904_swiftcalibration-M01_0100.fits',pola='all')
p.readfile(pola='I')
time=args.inputA
threshold=args.inputB
p.exrfi(time=time,threshold=threshold)

#data=p.exrfi(mask=True,plot_bandpass=False,recon=False,feature=20,wideband=True)
#data=p.filter(window=5,stepsize=5,thre=0.05)
#p.de_desperse(data,dm=93.965)
