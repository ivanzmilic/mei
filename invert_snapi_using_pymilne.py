import numpy as np 
import matplotlib.pyplot as plt 
import pyana 
from astropy.io import fits 
from tqdm import tqdm
import MilneEddington  # This requires that you copied the up-to-date ME version and / or pointed to it.

def data_loader(filename, line): # loads an f0 or a fits file to a num array:

	if (filename[-2:] == 'f0'): # its a pyana file

		print ("info::data_loader:: opening pyana file... ")

		stokes = pyana.fzread(filename)["data"]

		# wavelength usually hard coded

		# hardcoded:

		if (line == '5250'):

			wave = np.linspace(5249.5, 5250.5, 71)

			print ("info::data_loader::wave grid looks like this: ", wave)

			stokes = stokes[:,:,:,-121:-50]

		elif (line == '5172'):

			wave = np.linspace(5170,5175,501)

			wave = wave[100:401]

			print ("info::data_loader::wave grid looks like this: ", wave)

			stokes = stokes[:,:,:,100:401]


		# probably you want to normalize the cube before the inversion:
		I_qs = np.mean(stokes[:,:,0,0])
		print ("info::data_loader:: normalizing the intensity to:", I_qs)
		stokes /= I_qs
		print ("info::data_loader:: final stokes shape: ", stokes.shape)
		print ("info::data_loader:: final wave shape: ", wave.shape)
		return stokes[:,:,:,:],wave

	elif (filename[-4:] == 'fits'): # its a fits file

		print ("info::data_loader::opening fits file... ")

		stokes = fits.open(filename)[0].data

		# wavelength usually hard coded

		# hardcoded:

		if (line == '5250'):

			wave = np.linspace(5249.5, 5250.5, 71)

			print ("info::data_loader::wave grid looks like this: ", wave)

			stokes = stokes[:,:,:,-121:-50]

		elif (line == '5172'):

			wave = np.linspace(5170,5175,501)

			wave = wave[100:401]

			print ("info::data_loader::wave grid looks like this: ", wave)

			stokes = stokes[:,:,:,100:401]


		# probably you want to normalize the cube before the inversion:
		I_qs = np.mean(stokes[:,:,0,0])
		print ("info::data_loader:: normalizing the intensity to:", I_qs)
		stokes /= I_qs
		print ("info::data_loader:: final stokes shape: ", stokes.shape)
		print ("info::data_loader:: final wave shape: ", wave.shape)
		return stokes[:,:,:,:],wave

	else:

		print ("info::data_loader::data format not supported, returning zero")

		return 0


import sys

filename = sys.argv[1]
line = sys.argv[2]

stokes, wave = data_loader(filename, line)

print ("info::main::stokes has the shape: ", stokes.shape)
print ("info::main::wavelength grid has the shape: ", wave.shape)

# Time to setup pymilne 

# We are inverting only the first one
regions = [[wave, None]]

# These are the lines
lines   = [int(line)]
# so now our code has wavelength grid and knows what to do

n_threads = int(sys.argv[3])

me = MilneEddington.MilneEddington(regions, lines, nthreads=n_threads)

# Give some generic values for the noise:
noise_level = 1.e-3
noise = np.zeros((4, me.get_wavelength_array().size), dtype='float64', order='c')
noise += noise_level
noise[0] *= 5. #noise is typicaly larger for I, because of systematics - Discuss!

# Now we are going to select a region:
i = 0 # we start from here
j = 0 
nx = stokes.shape[0] # and take this big chunk
ny = stokes.shape[1]

# The same as before
model_guess = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
models_guess  = me.repeat_model(model_guess, nx, ny)

# Select a Stokes subset to fit:
to_fit = stokes

# This is where the inversion happens, be mindful it will take some time
#
model_out, syn_out, chi2 = me.invert(models_guess, to_fit, noise, nRandom = 10, nIter=40, chi2_thres=0.01, verbose=False)

# write out the model:
kek = fits.PrimaryHDU(model_out)
kek.writeto(sys.argv[4],overwrite=True)



