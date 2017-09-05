# this is used to analyse the rotation of the halo with K-giants

import numpy as np
import csv


path = "/Users/naoc/Documents/tian/rothalo/"
dpath = path + "data/"
ppath = path + "plot/"

fn_halo = "LMDR3_haloRGB2.dat"
fn_disk = "LMDR3_diskRGB2.dat"

data_halo = np.loadtxt(dpath+fn_halo,skiprows=1)
print(data_halo)