#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:20:58 2017

@author: cliu
"""

# common functions used in the project of VR-R in GAC direction
import numpy as np
#import scipy.linalg as splin
import numpy.linalg as nl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
import scipy.stats as stats
import scipy.special as special
import astropy.io.fits as fits
import emcee
import corner

matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

def MC_error(x,err):
    return np.random.normal(loc=x,scale=err)
   
def straightline(x,a,b):
    '''
    Straightline function
    '''
    return a + b*x

#Calculate velocity distribution at each bin, using Bayesian model
def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
#hierachical Beyasian model

def gauss_model(y):
    # sampling
    n = len(y)
    y_bar = np.mean(y)
    s2 = np.var(y)
    
    #step 1: draw sigma2 from posterior density of 
    #        sigma2 given v, p(sigma2|y)
    N = 5000
    sigma2 = randDraw_SInvChi2(n-1,s2, N)
    #step 2 : draw mu from p(mu|sigma2,y)
    mu = np.random.normal(loc=y_bar, \
            scale=np.sqrt(sigma2/n))
    popt = np.array([n,np.mean(mu),np.mean(np.sqrt(sigma2))])
    pcov = np.array([[1.0,0.0,0.0],[0.0,np.var(mu),0.0],\
                     [0.0,0.0,np.var(np.sqrt(sigma2))]])
    return popt, pcov

def lnprob_gauss(x,y):
    mu1 = x[0]
    sig1 = x[1]
    if np.isinf(mu1) or np.isinf(sig1) or sig1<0 or\
               np.abs(mu1)>100 :
        return -1e50
    g = stats.norm.pdf(y,mu1,sig1)# np.exp(-(y-mu1)**2/(2*sig1**2))/(np.sqrt(2*np.pi)*sig1)
    ind_g = (np.isinf(g)==False) & (np.isnan(g)==False) & (g>0)
    return np.sum(np.log(g[ind_g]))

def gauss_mcmcmodel(y):
    # MCMC sampling
    #n = len(y)
    
    #start to configure emcee
    nwalkers = 20
    ndim = 2
    p0=np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.rand(nwalkers)*50-25
    p0[:,1] = np.random.rand(nwalkers)*40+20
      
    sampler = emcee.EnsembleSampler(nwalkers, \
            ndim, lnprob_gauss, args=[y])
    
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    
    sampler.run_mcmc(pos, 10000)
    
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    #corner.corner(samples)
    popt = np.median(samples, axis=0)
    pcov = np.zeros((ndim,ndim))
    for i in range(ndim):
        for j in range(ndim):
            pcov[i,j] = (np.sum((samples[:,i]-popt[i])*\
                (samples[:,j]-popt[j])))/len(samples)
    return popt, pcov


def lnprob_gauss2(x,y):
    #n = np.float(len(y))
    f1 = x[0]
    mu1 = x[1]
    sig1 = x[2]
    mu2 = x[3]
    sig2 = x[4]
    if np.isinf(f1) or np.isinf(mu1) or np.isinf(sig1) or\
               np.isinf(mu2) or np.isinf(sig2) or\
               f1<0 or f1>1 or sig1<0 or sig2<0 or\
               np.abs(mu1)>50 or np.abs(mu2-160)>50:
        return -1e50
    g1 = f1*stats.norm.pdf(y,mu1,sig1)# np.exp(-(y-mu1)**2/(2*sig1**2))/(np.sqrt(2*np.pi)*sig1)
    g2 = (1-f1)*stats.norm.pdf(y,mu2,sig2)#np.exp(-(y-mu2)**2/(2*sig2**2))/(np.sqrt(2*np.pi)*sig2)
    g = g1+g2
    #print g
    ind_g = (np.isinf(g)==False) & (np.isnan(g)==False) & (g>0)
    return np.sum(np.log(g[ind_g]))

def gauss2_model(y):
    # MCMC sampling
    #n = len(y)
    
    #start to configure emcee
    nwalkers = 50
    ndim = 5
    p0=np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.rand(nwalkers)*0.2+0.7
    p0[:,1] = np.random.rand(nwalkers)*30+0
    p0[:,2] = np.random.rand(nwalkers)*30+55
    p0[:,3] = np.random.rand(nwalkers)*30+150
    p0[:,4] = np.random.rand(nwalkers)*30+40
    # p0[:,0] = np.random.rand(nwalkers)*0.2+0.7
    # p0[:,1] = np.random.rand(nwalkers)*50-25
    # p0[:,2] = np.random.rand(nwalkers)*40+20
    # p0[:,3] = np.random.rand(nwalkers)*50-25
    # p0[:,4] = np.random.rand(nwalkers)*5
    
    print("---------")
    sampler = emcee.EnsembleSampler(nwalkers, \
            ndim, lnprob_gauss2, args=[y])
    
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print("----1111111111111-----")
    sampler.run_mcmc(pos, 10000)
    
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    print("-----22222222222----")
    #corner.corner(samples)
    popt = np.median(samples, axis=0)
    print("-----33333333333----")
    pcov = np.zeros((ndim,ndim))
    print("-----00000000000----")
    for i in range(ndim):
        for j in range(ndim):
            pcov[i,j] = (np.sum((samples[:,i]-popt[i])*\
                (samples[:,j]-popt[j])))/len(samples)
    return popt, pcov, samples



#velocity dispersion profile
def exp_profile(x,a,h):
    '''
    exponential profile for velocity dispersion
    '''
    return a*np.exp(-(x-8.34)/h)

 #velocity dispersion profile
def brokenexp_profile(x,a,h,r0):
    '''
    broken-exponential profile for velocity dispersion
    '''
    y1 = a*np.exp(-(x-8.34)/h)
    y2 = a*np.exp(-(r0-8.34)/h)
    ind = (x>r0) & (x>0)
    y = y1
    #print y,y1,y2,np.sum(ind)
    y[ind] = y2
    
    return y

def straightline_profile(x,a,b):
    '''
    linear declining profile for velocity dispersion
    '''
    return a*(x-8.34)+b

#read data recalculate distance and spatial coordinates    
def spatialCoords(l,b,D, origin='galcenter'):
    '''
    converts a set of galactic coordinates to 3-D cartesian
    coordinates centered at the sun or the galactic center.
    
    Parameters:
       origin: 'sun' or 'galcenter'
       l, b: galactic coordinates in degrees
       D: distance from the sun to each object, in parsec
    '''
    cl=np.cos(l*np.pi/180)
    sl=np.sin(l*np.pi/180)
    cb=np.cos(b*np.pi/180)
    sb=np.sin(b*np.pi/180)
    Rsun=8340 #distance from the sun to the GC 
    zsun=27 #the height scale of the sun is set to 27pc according to Chen et al. 2003 
    if origin=='galcenter':
        x=Rsun-D*cl*cb
        z=zsun+D*sb
    else:
        if origin=='galcenterz0':
            x=Rsun-D*cl*cb
            z=0+D*sb
        else:
            x=D*cl*cb
            z=D*sb
    y=-D*sl*cb
    
    return x,y,z



def InvGammaln(x,alpha,beta):
    return np.log(beta)*alpha-(special.gammaln(alpha))+\
           np.log(x)*(-alpha-1)-beta/x

def Scl_InvChi2ln(x, nu, s2):
    return InvGammaln(x, nu/2.,nu/2.*s2)
#Solve with Bayesian normal linear regression
def randDraw_SInvChi2(nu,s2, N, \
    xmin=100., xmax=2000, ymin=0., ymax=0.006, dy=0.0):
    x = []
    k = 0
    m = 0
    while k<N and m <= 100:
        x0 = np.random.uniform(low=xmin,\
                    high=xmax,size=N*20)
        y0 = np.log(np.random.uniform(\
                    low=ymin,high=ymax,size=N*20))
        y1 = Scl_InvChi2ln(x0, nu, s2)-dy
        ind = (y0<y1)
        if m==0:
            x = x0[ind]
        else:
            x = np.append(x,x0[ind])
        k = k + np.sum(ind)
        m += 1
        #print k,m
    xx = x#np.array(x).reshape((k,1))
    
    return (xx[0:N])

#linear regression, Bayesian method
def linear_reg(x,y,n,k,N = 5000,\
              xmin=0.035, xmax=0.06, ymin=0., ymax=1, dy=5.816):
    
    V_beta0 = nl.inv(np.dot(x.T,x))
    beta_hat0 = np.dot(np.dot(V_beta0,x.T),y)
    s20 = 1./(n-k)*np.dot((y-np.dot(x,beta_hat0)).T,\
                (y-np.dot(x,beta_hat0)))
    #step 1: draw sigma from Inv-Chi2
    #N = 5000
    sigma2 = randDraw_SInvChi2(n-k,s20, N,\
          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, dy=dy)
    #step 2: draw beta from N(beta_hat, V_beta*sigma^2)
    #print dy
    beta = np.array([np.random.\
        multivariate_normal(beta_hat0,\
        V_beta0*sigma2[i])  for i in range(len(sigma2))])

    #print(np.shape(beta))
    return np.mean(beta,axis=0), np.std(beta,axis=0), beta

def readData(filename='/Users/cliu/mw/lamost_regular/data/DR3/DR3_Kgiants_short.fits'):
    '''
     read data
     '''
    
    hdulist = fits.open(filename)
    rgb = hdulist[1].data
    #use RJCE extinction to derive distance from the absolute magnitude from Carlin+2015
    [rgb.X2,rgb.Y2,rgb.Z2]=spatialCoords(rgb.glon,rgb.glat,rgb.distK50_RJCE)
    rgb.X2=rgb.X2/1000.0
    rgb.Y2=rgb.Y2/1000.0
    rgb.Z2=rgb.Z2/1000.0
    rgb.Rgc2=np.sqrt(rgb.X2**2+rgb.Y2**2)
    rgb.r_gc2=np.sqrt(rgb.Rgc2**2+rgb.Z2**2)
    #correct for 5.7 km/s to the radisl velocity
#    rgb.rv2 = rgb.rv+5.7
    #correct for 4.4 km/s to the radial velocity
    rgb.rv2 = rgb.rv+4.4
    rgb.RGB_nodup = rgb.RGBnew_nodup
    rgb.Mk_50 = rgb.M_K50
    return rgb #rgb.Rgc2, rgb.Z2, rgb.rv2, rgb.feh, 
    
def readDataRel4(filename='/Users/cliu/mw/lamost_regular/data/DR4/dr4_dist_short.fits'):
    '''
     read data
     '''
    
    hdulist = fits.open(filename)
    rgb = hdulist[1].data
    #use RJCE extinction to derive distance from the absolute magnitude from Carlin+2015
    [rgb.X2,rgb.Y2,rgb.Z2]=spatialCoords(rgb.glon,rgb.glat,rgb.distk50_RJCE*1000.0)
    rgb.X2=rgb.X2/1000.0
    rgb.Y2=rgb.Y2/1000.0
    rgb.Z2=rgb.Z2/1000.0
    rgb.Rgc2=np.sqrt(rgb.X2**2+rgb.Y2**2)
    rgb.r_gc2=np.sqrt(rgb.Rgc2**2+rgb.Z2**2)
    #correct for 5.7 km/s to the radisl velocity
    rgb.rv2 = rgb.rv+7.1
    return rgb #rgb.Rgc2, rgb.Z2, rgb.rv2, rgb.feh, 

def selection(rgb, MKrange):
    '''
    select proper subsamples for detailed investigation on the 
    metallicity-velocity distribution relation
    '''
    ind0 = (rgb.RGB_nodup==84) & (np.abs(rgb.glat)<5) &\
            (np.abs(rgb.glon-180.)<5) & (rgb.Rgc2<=15) &\
            (rgb.feh>-1.2) & \
            (rgb.Mk_50>MKrange[0]) & (rgb.Mk_50<MKrange[1]) &\
            (np.abs(rgb.rv2)<120)
    # Select the stars within |z|<0.3
    ind_GAC1 = ind0 & (np.abs(rgb.Z2)<=0.1)
    ind_GAC2 = ind0 & (np.abs(rgb.Z2)>0.1) & (np.abs(rgb.Z2)<0.2) 
    ind_GAC3 = ind0 & (np.abs(rgb.Z2)>0.2) & (np.abs(rgb.Z2)<0.3)
    ind_GAC = ind_GAC1 | ind_GAC2 | ind_GAC3
    ind_GACN = ind_GAC & (rgb.Z2>=0)
    ind_GACS = ind_GAC & (rgb.Z2<0)
    print('Ntot=%(n)d' % {'n':np.sum(ind_GAC)})
    print('N(|Z|<0.1)=%(n)d' % {'n':np.sum(ind_GAC1)})
    print('N(0.1<|Z|<0.2)=%(n)d' % {'n':np.sum(ind_GAC2)})
    print('N(0.2<|Z|<0.3)=%(n)d' % {'n':np.sum(ind_GAC3)})
    print('N_north=%(n)d' % {'n':np.sum(ind_GACN)})
    print('N_south=%(n)d' % {'n':np.sum(ind_GACS)})
    return ind_GAC, ind_GACN, ind_GACS, ind_GAC1, ind_GAC2, ind_GAC3

def vel_distR(R, rv, Rgrid, ind_GAC, Ncrit, filename):
    '''
    Fit the velocities with a Gaussian profile using Bayesian technique
    '''
    xfinegrid = np.arange(-150,150,1)
    Rcenter = (Rgrid[1:]+Rgrid[0:len(Rgrid)-1])/2.
    gauss_params = np.zeros([len(Rcenter),6])
    RVgrid = np.arange(-200,200,14)
    RVcenter = (RVgrid[1:len(RVgrid)]+RVgrid[0:len(RVgrid)-1])/2.
    N_all = np.zeros([len(Rcenter),1])
    fig = plt.figure(figsize=[5*3.5,5*(len(Rcenter)/5+1)])
    k = 1
    for i in range(len(Rcenter)):
        ind = ind_GAC & (R>Rgrid[i]) & (R<=Rgrid[i+1])# & (np.abs(rv)<120.)
        N_all[i] = np.sum(ind)
        if np.sum(ind)>Ncrit:
            h,xedges = np.histogram(rv[ind],RVgrid)
            ax = fig.add_subplot(len(Rcenter)/5+1,5,k)
            ax.text(-110,np.max(h)*1.1,r'N=%(d)d' % {'d':N_all[i]},fontsize=11)
            ax.text(-110,np.max(h)*1.2,r'R=%(n).2f kpc' % {'n':Rcenter[i]},fontsize=11)
            
            ax.errorbar(RVcenter,h.T,yerr=np.sqrt(h.T),fmt='ko')
            ax.plot(RVcenter,h.T,'k-')
            #popt, pcov = curve_fit(gaussian, RVcenter, h, p0=[1.,0.,30.]) 
            #Beyasian
            #popt, pcov = gauss_model(rv[ind])
            #MCMC
            popt2, pcov2 = gauss_mcmcmodel(rv[ind])
            #print popt,popt2
            #print pcov.diagonal(),pcov2.diagonal()
            gauss_params[i,0:2] = popt2
            gauss_params[i,2:4] = np.sqrt(pcov2.diagonal())
            #gauss_params[i,2] = np.abs(gauss_params[i,2])
            hg2 = stats.norm.pdf(xfinegrid,popt2[0],popt2[1])
            ax.plot(xfinegrid,hg2/np.sum(hg2)*np.sum(h)*14,'b-',linewidth=1.5)
            ax.set_xlim([-120,120])
            plt.xticks([-100+j*50 for j in range(5)])
            ax.set_ylim([0,np.max(h)*1.25])
            ax.set_xlabel(r'$v_R$ (km s$^{-1}$)', fontsize=12)
            k = k+1
            
    fig.show()
    fig.savefig(filename,bbox_inches='tight')
    return N_all, gauss_params, Rcenter


def vel_distR2(R, rv, Rgrid, ind_GAC, Ncrit, filename):
    '''
    Fit the velocities with 2-Gaussian profile using Bayesian technique
    '''
    xfinegrid = np.arange(-150,150,1)
    Rcenter = (Rgrid[1:]+Rgrid[0:len(Rgrid)-1])/2.
    gauss_params = np.zeros([len(Rcenter),10])
    RVgrid = np.arange(-200,200,17)
    RVcenter = (RVgrid[1:len(RVgrid)]+RVgrid[0:len(RVgrid)-1])/2.
    N_all = np.zeros([len(Rcenter),1])
    fig = plt.figure(figsize=[5*3.5,5*(len(Rcenter)/5+1)])
    k = 1
    for i in range(len(Rcenter)):
        ind = ind_GAC & (R>Rgrid[i]) & (R<=Rgrid[i+1])# & (np.abs(rv)<120.)
        N_all[i] = np.sum(ind)
        if np.sum(ind)>Ncrit:
            h,xedges = np.histogram(rv[ind],RVgrid)
            ax = fig.add_subplot(len(Rcenter)/5+1,5,k)
            ax.text(-110,np.max(h)*0.9,r'N=%(d)d' % {'d':N_all[i]},fontsize=11)
            ax.text(-110,np.max(h)*0.97,r'R=%(n).2f kpc' % {'n':Rcenter[i]},fontsize=11)
            
            ax.errorbar(RVcenter,h.T,yerr=np.sqrt(h.T),fmt='ko')
            #popt, pcov = curve_fit(gaussian, RVcenter, h, p0=[1.,0.,30.]) 
            #Beyasian
            popt2, pcov2 = gauss2_model(rv[ind])
            #print popt2
            gauss_params[i,0:5] = popt2
            gauss_params[i,5:10] = np.sqrt(pcov2.diagonal())
            gauss_params[i,2] = np.abs(gauss_params[i,2])
            hg1 = gaussian(xfinegrid,popt2[0],popt2[1],popt2[2])
            hg2 = gaussian(xfinegrid,1-popt2[0],popt2[3],popt2[4])
            hg = hg1+hg2
            ax.plot(xfinegrid,hg/np.sum(hg)*np.sum(h)*17,'k-',linewidth=1)
            ax.plot(xfinegrid,hg1/np.sum(hg)*np.sum(h)*17,'b--',linewidth=1)
            ax.plot(xfinegrid,hg2/np.sum(hg)*np.sum(h)*17,'r--',linewidth=1)
            ax.set_xlim([-120,120])
            plt.xticks([-100+j*50 for j in range(5)])
            ax.set_ylim([0,np.max(h)*1.15])
            ax.set_xlabel(r'$v_R$ (km s$^{-1}$)', fontsize=12)
            k = k+1
            
    fig.show()
    fig.savefig(filename,bbox_inches='tight')
    return N_all, gauss_params, Rcenter

#Bayesian method to derive velocity distribution for R-[Fe/H] subgroup stars
def vel_distFeHR(R,rv,feh,Rgrid,FeHgrid,ind_GAC,Ncrit,filename):
    
    FeHcenter = (FeHgrid[:-1]+FeHgrid[1:])/2.0
    dfeh=0.1
    Rcenter = (Rgrid[1:]+Rgrid[0:len(Rgrid)-1])/2.
    RVgrid = np.arange(-200,200,17)
    RVcenter = (RVgrid[1:len(RVgrid)]+RVgrid[0:len(RVgrid)-1])/2.
    N_fehR = np.zeros([len(Rcenter),len(FeHcenter)])
    N_fehRnofilt = np.zeros([len(Rcenter),len(FeHcenter)])
    gauss_paramsfehR = np.zeros([len(Rcenter),len(FeHcenter),4])
    ######################################
    ##### all z
    xfinegrid = np.arange(-150,150,1)
    fig = plt.figure(figsize=[30,30])
    cc = 'b'
    for i in range(len(Rcenter)):
        for j in range(len(FeHcenter)):
            if Rcenter[i]>=1. and Rcenter[i]<=11.5:
                cc = 'b'
            else:
                if Rcenter[i]>=12 and Rcenter[i]<=13 and\
                          FeHcenter[j]>-0.85 and FeHcenter[j]<-0.05:
                          cc = 'b'
                else:
                    cc = 'b'
            if j==0:
                ind = ind_GAC & (R>Rgrid[i]) & (R<=Rgrid[i+1]) &\
                        (feh<=(FeHgrid[j+1]+dfeh))
            else:
                ind = ind_GAC & (R>Rgrid[i]) & (R<=Rgrid[i+1]) &\
                        (feh>(FeHgrid[j]-dfeh)) & (feh<=(FeHgrid[j+1]+dfeh))# &\
                #             (np.abs(rv)<120.)
            N_fehR[i,j] = np.sum(ind)
            ind1 = ind_GAC & (R>Rgrid[i]) & (R<=Rgrid[i+1]) &\
                    (feh>(FeHgrid[j])) & (feh<=(FeHgrid[j+1]))# &\
            #             (np.abs(rv)<120.)
            N_fehRnofilt[i,j] = np.sum(ind1)
            xx1 = r'%(n).1f' % {'n':Rcenter[i]}
            xx2 = r'%(m)+.1f' % {'m':FeHcenter[j]}
            if np.sum(ind)>Ncrit:
                h,xedges = np.histogram(rv[ind],RVgrid)
                ax = fig.add_subplot(len(Rgrid),len(FeHgrid),i*len(FeHgrid)+j+1)#(len(FeHgrid)-j-1)*len(Rgrid)+(i+1))
                ax.text(-140,np.max(h)*0.7,r'%(d)d' % {'d':N_fehR[i,j]},\
                        fontsize=11)
                ax.text(-140,np.max(h)*0.97,xx1,fontsize=11)
                ax.text(-140,np.max(h)*0.85,xx2,fontsize=11)
                ax.errorbar(RVcenter,h.T,yerr=np.sqrt(h.T),fmt='k.')
                ax.plot(RVcenter,h.T,'k-')
                #popt, pcov = gauss_model(rv[ind])
                #popt, pcov = curve_fit(gaussian, RVcenter, h, p0=[np.double(N_fehR[i,j]),0,40])
                popt2, pcov2 = gauss_mcmcmodel(rv[ind])
                gauss_paramsfehR[i,j,0:2] = popt2
                gauss_paramsfehR[i,j,2:4] = np.sqrt(pcov2.diagonal())
                #gauss_paramsfehR[i,j,2] = np.abs(gauss_paramsfehR[i,j,2])
                hg2 = stats.norm.pdf(xfinegrid,popt2[0],popt2[1])
                ax.plot(xfinegrid,hg2/np.sum(hg2)*np.sum(h)*17,\
                    '-',linewidth=1.5,color=cc)
                ax.set_xlim([-150,150])
                ax.set_xticks([-100,50,0,50,100])
                ax.set_xticklabels(['-100','','0','','+100'])
                ax.set_ylim([0,np.max(h)*1.15])
                plt.xticks([-100+j*50 for j in range(5)])
                ax.set_xlabel(r'$v_R$ (km s$^{-1}$)', fontsize=12)
    fig.show()
    fig.savefig(filename,bbox_inches='tight')
    
    return N_fehR, N_fehRnofilt, gauss_paramsfehR, Rcenter
