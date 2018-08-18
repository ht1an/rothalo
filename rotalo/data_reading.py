import numpy as np
import csv
import matplotlib.pyplot as plt
import math as m
import PyAstronomy as PA
from PyAstronomy import pyasl as PAP
from scipy import signal
import scipy.optimize as SO
import consts as C
import CoordinateConversion as CC
import galpy.util.bovy_coords as gub
import random

class read_data:
    '''
    This is used to read data.
    Output variables: l_o,b_o,feh_o,dist_o,rv_o,Z_o,R_o
    '''

    def __init__(self, min_feh,max_feh):
        self.min_feh,self.max_feh = min_feh,max_feh

    def read_kgiant(self):
        dpath = "/Users/htian/Documents/GitHub/rothalo/data/LMDR3_diskRGB2.dat"
        data_halo = np.loadtxt(dpath, skiprows=1)
        ind_o = (data_halo[:, 21] > -10000000) & (data_halo[:,7]>self.min_feh) & (data_halo[:,7]<self.max_feh)\
                & (data_halo[:,4]>30) & (np.abs(data_halo[:,3]-180)>50)
        # ra_o = data_halo[ind_o, 1]
        # dec_o = data_halo[ind_o, 2]
        self.l_o = data_halo[ind_o, 3]
        self.b_o = data_halo[ind_o, 4]
        self.feh_o = data_halo[ind_o, 7]
        self.rv_o = data_halo[ind_o, 8]
        self.dist_o = data_halo[ind_o, 15]
        self.Z_o = data_halo[ind_o, 18]
        self.R_o = data_halo[ind_o, 19]
        self.X_o = C.X_sun-self.dist_o*np.cos(self.b_o*m.pi/180)*np.cos(self.l_o*m.pi/180)
        self.Y_o = -1*self.dist_o*np.cos(self.b_o*m.pi/180)*np.sin(self.l_o*m.pi/180)
        self.name = "DR3_RGB"

    def read_mgiant(self):
        dpath = "/Users/htian/Documents/GitHub/rothalo/data/dr4_mgiant.csv"
        ra,dec,rv,dist,sn,feh = np.loadtxt(dpath, skiprows=1,usecols=(0,1,2,9,10,11),delimiter=',',unpack=True)
        ind_o = (rv > -10000000) & (sn>10) & (feh>self.min_feh) & (feh<self.max_feh) & (dist >-1000)
        print(len(sn[sn<10]),len(sn))
        print("there are ",len(ra[ind_o])," stars readout!")
        # ra_o = data_halo[ind_o, 1]
        # dec_o = data_halo[ind_o, 2]
        lb = gub.radec_to_lb(ra,dec,degree=True)
        l = lb[:,0]
        b = lb[:,1]
        xyz = gub.lbd_to_XYZ(l,b,dist,degree=True)
        self.l_o = l[ind_o]
        self.b_o = b[ind_o]
        self.feh_o = feh[ind_o]
        self.rv_o = rv[ind_o]
        self.dist_o = dist[ind_o]
        self.Z_o = xyz[ind_o,2]
        self.R_o = np.sqrt((8-xyz[ind_o,0])**2 + xyz[ind_o,1]**2)
        self.name="DR4_mgiant"

    def read_BHB(self):
        dpath = "/Users/htian/Documents/GitHub/rothalo/data/BHB_Xue2011.dat"
        l,b,d,x,y,z,rv = np.loadtxt(dpath,usecols=(3,4,12,14,15,16,17),unpack=True)
        print(x[0],y[0],z[0],rv[0])
        self.l_o = l
        self.b_o = b
        self.Z_o = z
        self.R_o = np.sqrt(x**2+y**2)
        self.feh_o = l*0.0-1
        self.dist_o = d
        self.rv_o = rv
        self.name = "BHB_Xue2011"

class generate_mock:
    '''
    generate velocity, u,v,w, with Gaussian distribution
    generate position uniformly distributed
    All six columes are relative to the Galactic center, rather than the LSR
    '''
    def __init__(self,Xmean,Xdisp,Ymean,Ydisp,Zmean,Zdisp,Umean,Udisp,Vmean,Vdisp,Wmean,Wdisp,FeHmin,FeHmax,N):
        self.Xmean = Xmean
        self.Xdisp = Xdisp
        self.Ymean = Ymean
        self.Ydisp = Ydisp
        self.Zmean = Zmean
        self.Zdisp = Zdisp
        self.Umean = Umean
        self.Udisp = Udisp
        self.Vmean = Vmean
        self.Vdisp = Vdisp
        self.Wmean = Wmean
        self.Wdisp = Wdisp
        self.FeHmin = FeHmin
        self.FeHmax = FeHmax
        self.N = N
        self.name = "X_{Xmean}_Y_{Ymean}_Z_{Zmean}_U_{Umean}_V_{Vmean}_W_{Wmean}_N_{N}".format(**locals())

    def MOCK_POS_UNIFORM(self):
        self.X = np.random.uniform(self.Xmean-self.Xdisp/2,self.Xmean+self.Xdisp/2,self.N)*0+self.Xmean
        self.Y = np.random.uniform(self.Ymean-self.Ydisp/2,self.Ymean+self.Ydisp/2,self.N)*0+self.Ymean
        self.Z = np.random.uniform(self.Zmean-self.Zdisp/2,self.Zmean+self.Zdisp/2,self.N)*0+self.Zmean
        mr,mtheta,mphi = CC.xyz2sph(C.X_sun-self.X,self.Y,self.Z,Degree=True)
        self.l_o = 360-mphi
        self.b_o = 90-mtheta
        self.R_o = np.sqrt(self.X**2+self.Y**2)
        self.Z_o = self.Z
        self.dist_o = np.sqrt((8-self.X)**2+self.Y**2+self.Z**2)

    def MOCK_VEL_UNIFORM(self):
        self.U = np.random.uniform(self.Umean-self.Udisp/2,self.Umean+self.Udisp/2,self.N)*0+self.Umean
        self.V = np.random.uniform(self.Vmean-self.Vdisp/2,self.Vmean+self.Vdisp/2,self.N)*0+self.Vmean
        self.W = np.random.uniform(self.Wmean-self.Wdisp/2,self.Wmean+self.Wdisp/2,self.N)*0+self.Wmean

    def MOCK_POS_NORMAL(self):
        self.X = np.random.normal(self.Xmean,self.Xdisp,self.N)
        self.Y = np.random.normal(self.Ymean,self.Ydisp,self.N)
        self.Z = np.random.normal(self.Zmean,self.Zdisp,self.N)

    def MOCK_VEL_NORMAL(self):
        self.U = np.random.normal(self.Umean, self.Udisp, self.N)*0+self.Umean
        self.V = np.random.normal(self.Vmean, self.Vdisp, self.N)*0+self.Vmean
        self.W = np.random.normal(self.Wmean, self.Wdisp, self.N)*0+self.Wmean

    def MOCK_FeH(self):
        self.feh_o = np.random.uniform(self.FeHmin,self.FeHmax,self.N)