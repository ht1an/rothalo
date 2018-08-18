# this is the library for the project of testing rotation of the halo
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import PyAstronomy as PA
from PyAstronomy import pyasl as PAP
from scipy import signal
import scipy.optimize as SO
import consts as C
import CoordinateConversion as CC

class calc_likelihood:
    '''
    --------------------------------------------------------------------------------------------------
    |This is used to calculate the likelihood of an array.                                           |
    |Function for syntax is y = calc_likelihood(Mvaluearray, Sexpectvalue, existarray).              |
    |************************************************************************************************|
    |existarry is an array of a distribution.                                                        |
    |Sexpectvalue includes the value(s) which should be the expected dispersion for the existarray.  |
    |valuearray inludes the value(s) which should be the expected mean value for the existarray.     |
    |************************************************************************************************|
    |The function returns an array which has the same number of values with valuearray.              |
    --------------------------------------------------------------------------------------------------
    '''

    def __init__(self,Texpectvalue,Sexpectvalue,existvalue):
        self.Texpectvalue = Texpectvalue
        self.Sexpectvalue = Sexpectvalue
        self.existvalue = existvalue
    @property
    def with_gaussian(self):
        '''
        CAUTION: Calculate with Gaussian kernel. Sexpectvalue should be no zero!!!!!
        CAUTION: Lengthes of the three input arrays should be the same!!!!!!!!!!!!!!
        '''
        if self.Sexpectvalue <= 0:
            print("I have told you that the value of Sexpectvalue should be larger than 0!!!!")
        else:
            nv  = len(self.Texpectvalue)
            ns = len(self.existvalue)
            if (nv == ns):
                Glog = np.sum(-1*(self.existvalue-self.Texpectvalue)**2*0.5/(self.Sexpectvalue)**2)-\
                       ns*np.log(np.sqrt(2*C.PI)*self.Sexpectvalue)
                Ggs = np.exp(Glog)
                return Glog,Ggs
            else:
                print("I have told you that the two input arrays, Texpectvalue & existvalue, should have the same length!!!")

# class RV2VGSR:
#     '''
#     This is used to convert radial velocity (RV) to Vgsr
#     '''

