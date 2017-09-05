import csv
import numpy as np
import sys
def Inp_cat_out(OBJID,RADEG,DECDEG,z,mag1,mag0,plate,mjd,fiber,separation,fn):
	n1 = len(OBJID)
	n4 = len(RADEG)
	n5 = len(DECDEG)
	n6 = len(z)
	n7 = len(mag1)
	n8 = len(mag0)
	n9 = len(plate)
	n10 = len(mjd)
	n2 = len(fiber)
	n3 = len(separation)
	if (n1==n4) and (n4==n5) and (n5==n6) and (n6==n7) and (n7==n8) and (n8==n9) and (n9==n10) and (n10==n2) and (n2==n3):
		csvfile = file(fn,"wb")
		writer = csv.writer(csvfile)
		writer.writerow(['OBJID','RAHMS','DECDMS','RADEG','DECDEG','z','mag1','mag0','plate','mjd','fiber','separation'])
		for i in range(0,n1):
			rah = int(np.floor(RADEG[i]/15.))
			ram = int(np.floor((RADEG[i]/15-rah)*60))
			ras = round(((RADEG[i]/15-rah)*60-ram)*60,2)
			rahms = "{rah}:{ram}:{ras}".format(**locals())
			decd = int(np.floor(np.abs(DECDEG[i])))
			decm = int(np.floor((np.abs(DECDEG[i])-decd)*60))
			decs = round(((np.abs(DECDEG[i])-decd)*60-decm)*60,2)
			if DECDEG[i]<0:
				decdms = "-{decd}:{decm}:{decs}".format(**locals())
			else:
				decdms = "+{decd}:{decm}:{decs}".format(**locals())

			writer.writerow([OBJID[i],rahms,decdms,round(RADEG[i],5),round(DECDEG[i],5),round(z[i],5),
				round(mag1[i],5),round(mag0[i],5),int(plate[i]),int(mjd[i]),int(fiber[i]),round(separation[i],5)])
		csvfile.close() 
	else:
		print ("The numbers of the columns are not consistent!! Please have a check!!")


def Inp_cat_out_NEW(OBJID,RADEG,DECDEG,MAG0,fn):
	n1 = len(OBJID)
	n2 = len(RADEG)
	n3 = len(DECDEG)
	n4 = len(MAG0)
	if (n1==n2) and (n2==n3) and (n3==n4):
		csvfile = file(fn,"wb")
		writer = csv.writer(csvfile)
		writer.writerow(['OBJID','RADEG','DECDEG','MAG0'])
		for i in xrange(0,n1):
			writer.writerow([OBJID[i],round(RADEG[i],5),round(DECDEG[i],5),round(MAG0[i],5)])
		csvfile.close() 
	else:
		print ("The numbers of the columns are not consistent!! Please have a check!!")

def Inp_cat_output(OBJID=None,RADEG=None,DECDEG=None,MAG0=None,fn='CATALOG.csv',PMRA=None,PMDE=None,
	MAGTYPE=None,MAG1=None,MAG2=None,MAG3=None,MAG4=None,MAG5=None,
	MAG6=None,MAG7=None,PRI=None,OBJTYPE=None,OBJFROM=None,OBJCATALOG=None):
	names = ['OBJID','RADEG','DECDEG','PMRA','PMDE','MAG0','MAGTYPE','MAG1','MAG2','MAG3','MAG4','MAG5','MAG6','MAG7','PRI','OBJTYPE','OBJFROM','OBJCATALOG']
	print ('===================================     CAUTIONS    ================================')
	print ('the format for calling this function is: Inp_cat_output(OBJID=objID,fn=\'CATALOG.csv\')')
	print ('----------------')
	print ('The key words in the function are OBJID, RADEG, DECDEG,MAG0,fn,PMRA,PMDE,MAGTYPE,MAG1,MAG2,MAG3,MAG4,MAG5,MAG6,MAG7,PRI,OBJTYPE,OBJFROM,OBJCATALOG')
	print ('----------------')
	print ('RADEG, DECDEG,MAG0 are necessary!!!')
	print ('====================================================================================')
	output_cat = {}
	ind_names = np.zeros(18)
	if OBJID!=None:
		ind_names[0]=1
		output_cat.setdefault('OBJID',OBJID)
	else:
		print ("WARNING: The column OBJID is not necessary, but recommended!")
		print ('--------------------------------------------')
	if RADEG!=None:
		ind_names[1]=1
		output_cat.setdefault('RADEG',RADEG)
	else:
		print ("ERROR: You must have the column RADEG")
		print ('--------------------------------------------')
	if DECDEG!=None:
		ind_names[2]=1
		output_cat.setdefault('DECDEG',DECDEG)
	else:
		print ("ERROR: You must have the column DEDEG")
		print ('--------------------------------------------')
	if PMRA!=None:
		ind_names[3]=1
		output_cat.setdefault('PMRA',PMRA)
	if PMDE!=None:
		ind_names[4]=1
		output_cat.setdefault('PMDE',PMDE)
	if MAG0!=None:
		ind_names[5]=1
		output_cat.setdefault('MAG0',MAG0)
	else:
		print ("ERROR: You must have the column MAG0")
		print ('--------------------------------------------')
	if MAGTYPE!=None:
		ind_names[6]=1
		output_cat.setdefault('MAGTYPE',MAGTYPE)
	if MAG1!=None:
		ind_names[7]=1
		output_cat.setdefault('MAG1',MAG1)
	if MAG2!=None:
		ind_names[8]=1
		output_cat.setdefault('MAG2',MAG2)
	if MAG3!=None:
		ind_names[9]=1
		output_cat.setdefault('MAG3',MAG3)
	if MAG4!=None:
		ind_names[10]=1
		output_cat.setdefault('MAG4',MAG4)
	if MAG5!=None:
		ind_names[11]=1
		output_cat.setdefault('MAG5',MAG5)
	if MAG6!=None:
		ind_names[12]=1
		output_cat.setdefault('MAG6',MAG6)
	if MAG7!=None:
		ind_names[13]=1
		output_cat.setdefault('MAG7',MAG7)
	if PRI!=None:
		ind_names[14]=1
		output_cat.setdefault('PRI',PRI)
	else:
		print ("WARNING: The column PRI is not necessary, but recommended!")
		print ('--------------------------------------------')
	if OBJTYPE!=None:
		ind_names[15]=1
		output_cat.setdefault('OBJTYPE',OBJTYPE)
	else:
		print ("WARNING: The column OBJTYPE is not necessary, but recommended!")
		print ('--------------------------------------------')
	if OBJFROM!=None:
		ind_names[16]=1
		output_cat.setdefault('OBJFROM',OBJFROM)
	if OBJCATALOG!=None:
		ind_names[17]=1
		output_cat.setdefault('OBJCATALOG',OBJCATALOG)

	n1 = len(OBJID)
	n2 = len(RADEG)
	n3 = len(DECDEG)
	n4 = len(MAG0)
	if (n1==n2) and (n2==n3) and (n3==n4):
		csvfile = open(fn,"w")
		writer = csv.writer(csvfile)
		outnames = output_cat.keys()
			# writer.writerow()
		writer.writerow(outnames)
		for i in range(0,n1):
			outvalues  = []
			for j in output_cat.keys():
				tmpl = output_cat[j]
				outvalues.append(tmpl[i])
				del tmpl
			writer.writerow(outvalues)
			del outvalues
		csvfile.close()
	else:
		print ("The numbers of the columns are not consistent!! Please have a check!!")
		print (n1,n2,n3,n4)
		print ('=============================================')

	

