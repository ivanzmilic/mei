import numpy as np 
import matplotlib.pyplot as plt 
import pyana 
from astropy.io import fits 

#import MilneEddington # This requires that you copied the up-to-date ME version and / or pointed to it.

def data_loader(filename): # loads an f0 or a fits file to a num array:

	if (filename[-2:] == 'f0'): # its a pyana file

		print ("info::data_loader:: opening pyana file... ")

		stokes = pyana.fzread(filename)["data"]

		# wavelength usually hard coded

		wave = np.linspace(5249.5, 5251.0, 151)

		print ("info::data_loader::wave grid looks like this: ", wave)

		return stokes,wave

	elif (filename[-4:] == 'fits'): # its a fits file

		print ("info::data_loader::opening fits file... ")

		stokes = fits.open(filename)[0].data

		# wavelength usually hard coded

		wave = np.linspace(5249.5, 5251.0, 151)

		print ("info::data_loader::wave grid looks like this: ", wave)

		return stokes, wave

	else:

		print ("info::data_loader::data format not supported, returning zero")

		return 0


import sys

filename = sys.argv[1]

stokes, wave = data_loader(filename)

print ("info::main::stokes has the shape: ", stokes.shape)
print ("info::main::wavelength grid has the shape: ", wave.shape)


