import numpy as np 
import matplotlib.pyplot as plt 
import pyana 
from astropy.io import fits 
from tqdm import tqdm
import sys
from scipy.ndimage import gaussian_filter1d

def data_loader(filename, wavelength_grid_filename, left, right, skip): # loads an f0 or a fits file to a num array, returns it

	# filename - the file to read, f0 or .fits
	# wavelength_grid_filename - whatever snapi used to synthesize the data
	# line - 5250, 5172
	# left - left boundary for extracting wavelenght
	# right - right boundary
	# skip - usually 1, but we can downsample

	if (filename[-2:] == 'f0'): # its a pyana file

		print ("info::data_loader:: opening pyana file... ")

		stokes = pyana.fzread(filename)["data"]

	elif (filename[-4:] == 'fits'): # its a fits file

		print ("info::data_loader::opening fits file... ")

		stokes = fits.open(filename)[0].data

		# wavelength usually hard coded

	else:

		print ("info::data_loader::data format not supported, returning zero")

		return 0


	wave = np.loadtxt(wavelength_grid_filename, skiprows=1)

	# probably you want to normalize the cube before the inversion:
	I_qs = np.mean(stokes[:,:,0,0])
	print ("info::data_loader:: normalizing the intensity to:", I_qs)
	stokes /= I_qs
	stokes = stokes[:,:,:,left:right:skip]
	wave = wave[left:right:skip]
	print ("info::data_loader:: final stokes shape: ", stokes.shape)
	print ("info::data_loader:: final wave shape: ", wave.shape)
	return stokes[:,:,:,:],wave

def wfa_blos_estimate(spectrum, wave, line):

	# first find the minimum:

	id_l0 = np.argmin(spectrum[0])
	span = 10

	left = id_l0-span
	if (left <0):
		left = 0

	right = id_l0+span+1
	if (right > spectrum.shape[-1]):
		right = spectrum.shape[-1]


	I = spectrum[0, left:right]
	V = spectrum[3, left:right]
	ll = wave[left:right]

	dI_dll = np.gradient(I) / np.gradient(ll)

	p = np.polyfit(dI_dll,V, 1)
	print (p)
	k = p[0]

	g = 1.0

	if (line == '5172'):
		g = 1.75
	elif (line == '5250'):
		g = 3.0 

	B_los = -k / float(line) ** 2.0 / g / 4.67E-13

	return B_los

# Main part of the program:

spectrafile = sys.argv[1]
wavelengthfile = sys.argv[2]
line = sys.argv[3]
left = int(sys.argv[4])
right = int(sys.argv[5])
skip = int(sys.argv[6])

outputfile = sys.argv[7]

stokes, wave = data_loader(spectrafile, wavelengthfile, left, right, skip)

print(wfa_blos_estimate(stokes[100,200], wave, line))


