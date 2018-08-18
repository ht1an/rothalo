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


path = "/Users/htian/Documents/GitHub/rothalo/"
dpath = path + "data/"
ppath = "/Users/htian/Documents/GitHub/rothalo/plot/"

# *************** this part includes the code for reading data ***************
# this part can be modified
data = dr.read_data(-5,-0.8)
data.read_kgiant()
#*************** this part for constants *************************************
Rl,Rr,Rstp = 0,40,5
Zl,Zr,Zstp = -40,40,5
l_o = data.l_o
b_o = data.b_o
rv_o = data.rv_o
feh_o = data.feh_o
dist_o = data.dist_o
Z_o = data.Z_o
R_o = data.R_o
dist_o = data.dist_o
obj = data.name
# print(np.mean(rv_o),np.min(rv_o),np.max(rv_o))
# #*************** this part for mocking data **********************************
#  CAUTION!! This part is only for local volume!!!!
# data = dr.generate_mock(12.,0.001,4,0.001,4.1,0.001,-50.,0.001,150.,0.001,50.,0.0001,-2.,-1.,20)
# data.MOCK_FeH()
# data.MOCK_POS_UNIFORM()
# data.MOCK_VEL_NORMAL()
# l_o = data.l_o
# b_o = data.b_o
# feh_o = data.feh_o
# dist_o = data.dist_o
# Z_o = data.Z_o
# R_o = data.R_o
# obj = data.name
# rv_o = np.zeros_like(Z_o)
# for i in range(0,20):
#     v_pos = np.array([data.X[i]-8,data.Y[i],data.Z[i]])
#     v_vel = np.array([data.U[i]-C.U_sun,data.V[i]-C.V_LSR-C.V_sun,data.W[i]-C.W_sun])
#     rv_o[i] = (v_pos[0]*v_vel[0]+v_pos[1]*v_vel[1]++v_pos[2]*v_vel[2])/np.sqrt(np.sum(v_pos**2))
# print(np.mean(rv_o),np.mean(data.V))
# # print(np.mean(data.X-8),np.mean(data.Y),np.mean(data.Z))
# # print(np.mean(data.U),np.mean(data.V-220),np.mean(data.W))
# # print(480/np.sqrt(32))
lxxx = np.linspace(0,360,1001)

Xsa, Ysa, Zsa = CC.sph2xyz(90 - b_o, l_o - 180, r=dist_o, Degree=True)
# print(np.mean(Xs),np.mean(Ys),np.mean(Zs),np.mean(l_o))
rcca, thetacca, phicca = CC.xyz2sph(C.X_sun + Xsa, Ysa, Zsa, Degree=True)
Lgca = phicca
Bgca = 90 - thetacca

Vgsr_o = rv_o + C.U_sun * np.cos(b_o * math.pi / 180) * np.cos(l_o * math.pi / 180) + \
       (C.V_LSR + C.V_sun) * np.cos(b_o * math.pi / 180) * np.sin(l_o * math.pi / 180) + \
       C.W_sun * np.sin(b_o * math.pi / 180)
fig = plt.figure(figsize=(6,4))
plt.plot(l_o,Vgsr_o,'k.')
plt.axis([0,360,-400,400])
plt.savefig(ppath+"l_RV.eps")
# print(np.min(data.X),np.max(data.X),'----------')
#*************** this part for constants *************************************
Rl,Rr,Rstp = 0,40,5
Zl,Zr,Zstp = -40,40,5
#*****************************************************************************

print(len(R_o),' stars are used in this work!')

Vl,Vr,Vstp = 0,1000,10
Dl,Dr,Dstp = 0.5,400.5,5
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
MMV = np.zeros_like(max_v)-0
DMV = np.zeros_like(max_v)-100000
nmb = np.zeros_like(max_v)
# print(np.min(Z_o),np.max(Z_o))
# print(np.min(R_o),np.max(R_o))
i_check = 0
print(np.min(l_o),np.max(l_o))
Lbin0 = np.linspace(0,360,19)-180
Lbin = Lbin0[:18]
L_V = np.zeros_like(Lbin)
iR = 0
for iL in range(0,18):#Nr):
    print(iR)
    min_R = 0#Rarray[iR]
    max_R = 35#Rarray[iR+1]
    for iZ in range(0,1):#Nz):
        min_Z = -20#Zarray[iZ]
        max_Z = 20#Zarray[iZ+1]
        ind = (Z_o>min_Z) & (Z_o<max_Z) & (R_o>min_R) & (R_o<max_R) & (Lgca>Lbin0[iL]) & (Lgca<Lbin0[iL+1])#& (l_o>45) & (l_o<135)# & (np.abs(Z_o)<4)
        nmb[iZ, iR] = len(Z_o[ind])
        if (nmb[iZ,iR]>10):# & (i_check<1):
            # nmb[iZ, iR] = nmb[iZ, iR]+50
            i_check = i_check+1
            l = l_o[ind]
            b = b_o[ind]
            feh = feh_o[ind]
            rv = rv_o[ind]
            dist = dist_o[ind]
            Z = Z_o[ind]
            R = R_o[ind]
            DD = np.sqrt(R**2+Z**2)
            Xs,Ys,Zs = CC.sph2xyz(90-b,l-180,r=dist,Degree=True)
            # print(np.mean(Xs),np.mean(Ys),np.mean(Zs),np.mean(l_o))
            rcc,thetacc,phicc = CC.xyz2sph(C.X_sun+Xs,Ys,Zs,Degree=True)
            Lgc = phicc
            Bgc = 90-thetacc

            Vgsr = rv+C.U_sun*np.cos(b*math.pi/180)*np.cos(l*math.pi/180)+ \
                (C.V_LSR+C.V_sun)*np.cos(b*math.pi/180)*np.sin(l*math.pi/180)+ \
                C.W_sun*np.sin(b*math.pi/180)
            # print(np.mean(Vgsr))
            # MMV[iZ, iR] = np.mean(Vgsr)
            # DMV[iZ, iR] = np.std(Vgsr)
            LH_log = np.zeros((len(Disps),len(Vphis)))
            LH = np.zeros_like(LH_log)
            # # print(np.mean(rv),np.mean(l),np.mean(b),min_R,max_R,min_Z,max_Z,np.mean(Ys),np.mean(Zs),np.mean(Xs),np.mean(Lgc),np.mean(Bgc))
            # mV = np.zeros(18)
            # dV = np.zeros(18)
            # nV = np.zeros(18)
            # lV0 = np.linspace(0,360,19)-180
            # lV = lV0[:18]
            # for ilv in range(0,18):
            #     indv = (Lgc>lV0[ilv]) & (Lgc<lV0[ilv+1])
            #     nV[ilv] = len(Vgsr[indv])
            #     if nV[ilv]>1:
            #         mV[ilv] = np.mean(Vgsr[indv])
            #         dV[ilv] = np.std(Vgsr[indv])
            # indv0 = nV>1

            # fig = plt.figure()
            # plt.plot([-180,360],[0,0],'r--')
            # plt.plot([-180,360],[np.mean(Vgsr),np.mean(Vgsr)],'b--')
            # plt.plot(Lgc,Vgsr,'k.',markersize=0.5)
            # # plt.plot(lV[indv0],mV[indv0],'rp')
            # plt.errorbar(lV[indv0], mV[indv0], yerr=dV[indv0], fmt='gp')
            # plt.axis([-180,180,-400,400])
            # plt.xlabel("$l_{gsr}$")
            # plt.ylabel("$V_{gsr}$")
            # plt.ylabel("$V_{gsr}$")
            # plt.savefig(ppath + "lvgsr_A_A.eps".format(**locals()))
            # plt.close(fig)

            fig = plt.figure()
            plt.hist(Vgsr,bins=100)
            plt.savefig(ppath + "lvgsr_hist.eps".format(**locals()))
            plt.close(fig)

            Px_star = R * np.cos(Lgc * math.pi / 180) - C.X_sun
            Py_star = R * np.sin(Lgc * math.pi / 180)
            Pz_star = Z
            Pm_star = np.sqrt(Px_star ** 2 + Py_star ** 2 + Pz_star ** 2)
            print(np.min(Pm_star),np.max(Pm_star),' min and max module of stars from sun')
            for i in range(0,Nv+1):
                sVphi = Vphis[i]
                for j in range(0,Nd+1):
                    sDisp = Disps[j]
                    nss = nmb[iZ,iR]
                    Vx_star = -1 * sVphi * np.sin(Lgc * math.pi / 180) + C.U_sun
                    Vy_star = sVphi * np.cos(Lgc * math.pi / 180) - C.V_LSR - C.V_sun
                    Vz_star = sVphi*0-C.W_sun
                    rrv = (Vx_star * Px_star + Vy_star * Py_star + Vz_star * Pz_star)/Pm_star
                    rvgsr = rrv + C.U_sun * np.cos(b * math.pi / 180) * np.cos(l * math.pi / 180) + \
                                (C.V_LSR + C.V_sun) * np.cos(b * math.pi / 180) * np.sin(l * math.pi / 180) + \
                                C.W_sun * np.sin(b * math.pi / 180)
                    halostar = lib.calc_likelihood(rv,sDisp,rrv)
                    # halostar = lib.calc_likelihood(Vgsr,sDisp,rvgsr)
                    LH_log[j,i],LH[j,i] = halostar.with_gaussian
            max_v[iZ,iR],max_d[iZ,iR] = Vgsrs2[LH_log==np.max(LH_log)],Disps2[LH_log==np.max(LH_log)]
            L_V[iL] =max_v[iZ,iR]
            # print(max_v[iZ,iR],max_d[iZ,iR],'//////')
            # print(np.mean(rv), np.mean(l), np.mean(b), min_R, max_R, min_Z, max_Z, np.mean(Ys), np.mean(Zs),
            #       np.mean(Xs), np.mean(Lgc), np.mean(Bgc),max_v[iZ,iR])

            fig = plt.figure(figsize=(6,4))
            vmin = -1e5
            vmax = -3e4
            plt.contourf(Vphis,Disps,LH_log,cmap = 'jet',vmin=vmin,vmax = vmax,levels = np.linspace(vmin,vmax,101))
            plt.colorbar()
            plt.savefig(ppath+"contour_{iR}_AA{iZ}.eps".format(**locals()))
            # print(np.max(LH_log),np.min(LH_log),'    -------')
            # print(len(Vphis),len(Disps))
# # plt.xlabel("$l_{gsr}$")
# # plt.ylabel("$V_{gsr}$")
# # plt.ylabel("$V_{gsr}$")
# #
# # plt.savefig(ppath+"lvgsr.eps".format(**locals()))
# # plt.close(fig)
#
# plt.figure(figsize=(6,10))
# # plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,max_v,vmin=-300,vmax=300,levels=np.linspace(-300,300,101),cmap='jet')
# levels = np.linspace(-300,1700,101)
# cmap = plt.get_cmap('rainbow')
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],max_v,cmap=cmap,vmin=-300,vmax=1700,norm=norm)
# plt.colorbar()
# plt.xlabel("R(kpc)")
# plt.ylabel("Z(kpc)")
# ROT_LSR= C.V_LSR
# plt.savefig(ppath+obj+'V_Sun{ROT_LSR}.eps'.format(**locals()))
#
# plt.figure(figsize=(6,10))
# levels = np.linspace(0,300,101)
# cmap = plt.get_cmap('rainbow')
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# # plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,max_d,vmin=0,vmax=300,levels=np.linspace(0,300,101),cmap='jet')
# plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],max_d,cmap=cmap,vmin=0,vmax=300,norm=norm)
# plt.colorbar()
# plt.xlabel("R(kpc)")
# plt.ylabel("Z(kpc)")
# plt.savefig(ppath+obj+'D_Sun{ROT_LSR}.eps'.format(**locals()))
#
plt.figure(figsize=(6,10))
levels = np.linspace(0,300,101)
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,nmb,origin=None,vmin=1,vmax=501,levels=np.linspace(1,501,101),cmap='jet')
plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],nmb,cmap=cmap,vmin=0,vmax=301,norm=norm)
plt.colorbar()
plt.xlabel("R(kpc)")
plt.ylabel("Z(kpc)")
plt.savefig(ppath+obj+'N_Sun.eps'.format(**locals()))
print(np.max(max_v),np.max(max_d),np.max(nmb))
print(max_v[max_v>30])

plt.figure(figsize=(6,10))
levels = np.linspace(-80,50,101)
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,nmb,origin=None,vmin=1,vmax=501,levels=np.linspace(1,501,101),cmap='jet')
plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],MMV,cmap=cmap,vmin=-80,vmax=50,norm=norm)
plt.colorbar()
plt.xlabel("R(kpc)")
plt.ylabel("Z(kpc)")
plt.savefig(ppath+obj+'Contour_M.eps'.format(**locals()))
print(np.min(MMV[MMV>-500]),np.max(MMV[MMV>-500]))

plt.figure(figsize=(6,10))
levels = np.linspace(0,200,101)
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.contourf(Rarray[:len(Rarray)-1]+2.5,Zarray[:len(Zarray)-1]+2.5,nmb,origin=None,vmin=1,vmax=501,levels=np.linspace(1,501,101),cmap='jet')
plt.pcolormesh(Rarray[:len(Rarray)-1],Zarray[:len(Zarray)-1],DMV,cmap=cmap,vmin=0,vmax=200,norm=norm)
plt.colorbar()
plt.xlabel("R(kpc)")
plt.ylabel("Z(kpc)")
plt.savefig(ppath+obj+'Contour_D.eps'.format(**locals()))
# print(np.min(MMV[MMV>-500]),np.max(MMV[MMV>-500]))

plt.figure()
plt.plot(Lbin,L_V,'gp')
plt.savefig(ppath+'V_L_max.eps')
