# this is a library for coordinate conversion
import numpy as np
import math as m

def xy2polar(x,y,Degree=True):
	rho = np.sqrt(x**2+y**2)
	if Degree:
		phi = np.arctan2(y,x)*180/m.pi
	else:
		phi = np.arctan2(y,x)
	return rho,phi

def polar2xy(rho,phi,Degree=True):
	if Degree:
		x = rho*np.cos(phi*m.pi/180)
		y = rho*np.sin(phi*m.pi/180)
	else:
		x = rho*np.cos(phi)
		y = rho*np.sin(phi)
	return x,y

def xyz2sph(x,y,z,Degree=True):
	r = np.sqrt(x**2+y**2+z**2)
	if Degree:
		theta = np.arccos(z/r)*180/m.pi
		phi = np.arctan2(y,x)*180/m.pi
		# phi[phi<0]=phi[phi<0]+360
	else:
		theta = np.arccos(z/r)
		phi = np.arctan2(y,x)
		# phi[phi<0]=phi[phi<0]+2*m.pi

	return r,theta,phi

def sph2xyz(theta,phi,r=1,Degree=True):
	if Degree==True:
		z = r*np.cos(theta*m.pi/180)
		x = r*np.sin(theta*m.pi/180)*np.cos(phi*m.pi/180)
		y = r*np.sin(theta*m.pi/180)*np.sin(phi*m.pi/180)
	else:
		z = r*np.cos(theta)
		x = r*np.sin(theta)*np.cos(phi)
		y = r*np.sin(theta)*np.sin(phi)
	return x,y,z
			
def rotate_sph(theta,phi,theta0,phi0,r=1,Degree=True):
	# first rotate phi0 around z-axis
	x1,y1,z1 = sph2xyz(theta,phi-phi0,r=r,Degree=Degree)
	# second do rotate theta around y-axis
	xtmp = z1
	ytmp = x1
	ztmp = y1

	rtmp,thetatmp,phitmp = xyz2sph(xtmp,ytmp,ztmp,Degree=Degree)
	xnew,ynew,znew = sph2xyz(thetatmp,phitmp-theta0,r=rtmp,Degree=Degree)

	x2 = ynew
	y2 = znew
	z2 = xnew

	rnew,thetanew,phinew = xyz2sph(x2,y2,z2,Degree=Degree)
	# phinew[phinew<0] = phinew[phinew<0]+360
	return rnew,thetanew,phinew

def cyd2xyz(R,Z,phi,Degree=True):
	if Degree:
		x = R * np.cos(phi * m.pi/180)
		y = R * np.sin(phi * m.pi/180)
	else:
		x = R * np.cos(phi)
		y = R * np.sin(phi)
	z = Z
	return x,y,z

def xyz2cyd(X,Y,Z,Degree=True):
	if Degree:
		phi = np.tan2(y,x)*180/m.pi
	else:
		phi = np.tan2(y,x)
	R = np.sqrt(x**2+y**2)
	return R,Z,phi

def lb2aitoff(l,b,Degree=True):
	if Degree:
		z2 = 1 + np.cos(b*m.pi/180) * np.cos(l*m.pi/180/2)
		x = np.cos(b*m.pi/180) * np.sin(l*m.pi/180/2) / np.sqrt(z2)
		y = np.sin(b*m.pi/180) / np.sqrt(z2)
	else:
		z2 = 1 + np.cos(b) * np.cos(l/2)
		x = np.cos(b) * np.sin(l/2) / np.sqrt(z2)
		y = np.sin(b) / np.sqrt(z2)
	return x,y






