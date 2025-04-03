import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt 
import sys

# Assemble individual ME inversions into a time series 

# Path + name, until the number
path_name = sys.argv[1]

# Extension, everything after the number
ext = sys.argv[2]

# Number:
start = int(sys.argv[3])
step = 150
number = int(sys.argv[4])

cube = []

from tqdm import tqdm
for i in tqdm(range(0,number)):

	currentfilename = path_name+str(start+i*step)+ext
	temp = fits.open(currentfilename)[0].data
	if (i==0):
		cube = temp
		print ("info::input cube size is: ", temp.shape)
		NX,NY = temp.shape
		cube = cube.reshape(1,NX,NY)
	else:
		cube = np.concatenate((cube,temp[None,:,:]), axis=0)


print ("info::final cube shape is: ", cube.shape)

kek = fits.PrimaryHDU(cube)
kek.writeto(path_name+"series"+ext, overwrite=True)



