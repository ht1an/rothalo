import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import CoordinateConversion as CC
import PyAstronomy as PA
from PyAstronomy import pyasl as PAP
from scipy import signal
import scipy.optimize as SO
import consts as C
import lib
from matplotlib.colors import BoundaryNorm
import galpy as gp
import data_reading as dr
'''
assuming the rotation is not perfectly along z-axis, 
with an offset (theta, phi), theta is the angle from
z-axis, and phi is the angle from the x axis, y-axis
with (theta, phi) = (90,90) degree.
'''
path = "/Users/htian/Documents/GitHub/rothalo/"
dpath = path + "data/"
ppath = "/Users/htian/Documents/GitHub/rothalo/plot/"

# *************** this part includes the code for reading data ***************
obj = "halo_"  # "disk" or "halo"
data = dr.read_data(-5,-0.8)
data.read_BHB()
l_o = data.l_o
b_o = data.b_o
rv_o = data.rv_o
feh_o = data.feh_o
dist_o = data.dist_o
Z_o = data.Z_o
R_o = data.R_o
Rl,Rr,Rstp = 0,40,5
Zl,Zr,Zstp = -40,40,5

fig= plt.figure(figsize=(4,4))
plt.plot(R_o,Z_o,'k.',alpha=0.5)
plt.savefig("BHB_RA.eps")

print(len(R_o),' stars are used in this work!')

# min_feh, max_feh = -5,-0.8
# min_R, max_R = 10,15
# min_Z, max_Z = 0,5
Vl,Vr,Vstp = -300,300,10
Dl,Dr,Dstp = 5,305,10
Nv = int((Vr-Vl)/Vstp)
Nd = int((Dr-Dl)/Dstp)

Vphis = np.linspace(Vl,Vr,Nv+1)
Disps = np.linspace(Dl,Dr,Nd+1)
Vgsrs2,Disps2 = np.meshgrid(Vphis,Disps)
Nr = int((Rr-Rl)/Rstp)
Nz = int((Zr-Zl)/Zstp)

Rarray = np.linspace(Rl,Rr,Nr+1)
Zarray = np.linspace(Zl,Zr,Nz+1)
max_v = np.zeros((len(Zarray)-1,len(Rarray)-1))-1000000
max_d = np.zeros_like(max_v)-1000000
nmb = np.zeros_like(max_v)
for iR in range(0,Nr):
    print(iR)
    min_R = Rarray[iR]
    max_R = Rarray[iR+1]
    for iZ in range(0,Nz):
        min_Z = Zarray[iZ]
        max_Z = Zarray[iZ+1]
        ind = (Z_o>min_Z) & (Z_o<max_Z) & (R_o>min_R) & (R_o<max_R)
        nmb[iZ, iR] = len(Z_o[ind])
        if nmb[iZ,iR]>10:
            l = l_o[ind]
            b = b_o[ind]
            feh = feh_o[ind]
            rv = rv_o[ind]
            dist = dist_o[ind]
            Z = Z_o[ind]
            R = R_o[ind]
            DD = np.sqrt(R**2+Z**2)
            Xs,Ys,Zs = CC.sph2xyz(90-b,l,r=dist,Degree=True)
            rcc,thetacc,phicc = CC.xyz2sph(C.X_sun-Xs,Ys,Zs,Degree=True)
            Lgc = phicc
            Bgc = 90-thetacc

            Vgsr = rv+C.U_sun*np.cos(b*math.pi/180)*np.cos(l*math.pi/180)+ \
                (C.V_LSR+C.V_sun)*np.cos(b*math.pi/180)*np.sin(l*math.pi/180)+ \
                C.W_sun*np.sin(b*math.pi/180)
            LH_log = np.zeros((len(Disps),len(Vphis)))
            LH = np.zeros_like(LH_log)
            for i in range(0,Nv+1):
                sVphi = Vphis[i]
                for j in range(0,Nd+1):
                    sDisp = Disps[j]
                    V_star = np.array([-1 * sVphi * np.sin(Lgc * math.pi / 180) - C.U_sun,
                                       sVphi * np.cos(Lgc * math.pi / 180) - C.V_LSR-C.V_sun, -1*C.W_sun])
                    P_star = np.array([R * np.sin(Lgc * math.pi / 180) - C.X_sun, R * np.cos(Lgc * math.pi / 180), Z])
                    rrv = (V_star[0]*P_star[0]+V_star[1]*P_star[1]+V_star[2]*P_star[2])/\
                          (np.sqrt((P_star[0]**2)+(P_star[1]**2)+(P_star[2]**2)))
                    rvgsr = rrv+C.U_sun*np.cos(b*math.pi/180)*np.cos(l*math.pi/180)+ \
                            (C.V_LSR+C.V_sun)*np.cos(b*math.pi/180)*np.sin(l*math.pi/180)+ \
                            C.W_sun*np.sin(b*math.pi/180)
                    halostar = lib.calc_likelihood(rv,sDisp,rrv)
                    LH_log[j,i],LH[j,i] = halostar.with_gaussian
            max_v[iZ,iR],max_d[iZ,iR] = Vgsrs2[LH_log==np.max(LH_log)],Disps2[LH_log==np.max(LH_log)]

plt.figure(figsize=(6,10))
# plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,max_v,vmin=-300,vmax=300,levels=np.linspace(-300,300,101),cmap='jet')
levels = np.linspace(-300,300,101)
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],max_v,cmap=cmap,vmin=-300,vmax=300,norm=norm)
plt.colorbar()
plt.xlabel("R(kpc)")
plt.ylabel("Z(kpc)")
ROT_LSR= C.V_LSR
plt.savefig(ppath+obj+'V_Sun{ROT_LSR}.eps'.format(**locals()))

plt.figure(figsize=(6,10))
levels = np.linspace(0,300,101)
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,max_d,vmin=0,vmax=300,levels=np.linspace(0,300,101),cmap='jet')
plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],max_d,cmap=cmap,vmin=0,vmax=300,norm=norm)
plt.colorbar()
plt.xlabel("R(kpc)")
plt.ylabel("Z(kpc)")
plt.savefig(ppath+obj+'D_Sun{ROT_LSR}.eps'.format(**locals()))

plt.figure(figsize=(6,10))
levels = np.linspace(1,501,101)
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,nmb,origin=None,vmin=1,vmax=501,levels=np.linspace(1,501,101),cmap='jet')
plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],nmb,cmap=cmap,vmin=1,vmax=501,norm=norm)
# plt.imshow(nmb,cmap=mpl.cm.jet)
plt.colorbar()
plt.xlabel("R(kpc)")
plt.ylabel("Z(kpc)")
plt.savefig(ppath+obj+'N_Sun{ROT_LSR}.eps'.format(**locals()))
print(np.max(max_v))