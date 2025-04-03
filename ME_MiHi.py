import MilneEddington
import numpy as np
from astropy.io import fits
import sys
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import firtez_dz as frz

#load the data
#filepath = sys.argv[1]
filepath = 'stokes.fits'
print("info::reading the input file...")

stokes = fits.open(filepath)[0].data
print("info::read. stokes shape is: ", stokes.shape)

#wavelength calibration
ll = fits.open(filepath)[1].data
    

nx = stokes.shape[0]
ny = stokes.shape[1]  


#noise
noise_level = 1.e-2
noise = np.zeros([4,200]) # need to change
noise += noise_level
noise[1] /= 2.
noise[2] /= 2.
noise[3] /= 2. 
#apply huge noise to the telluric lines
noise[:,90:100] += 1.e32
noise[:,165:175] += 1.e32
print("DEBUG")

#model
regions = [[ll[:],None]] # select the wavelength
print("DEBUG")
lines = [6301,6302] # lines to calculate
print("DEBUG")
#n_threads = int(sys.argv[2]) # input number of cores as an argument
n_threads = int(sys.argv[1])
print("DEBUG")
me = MilneEddington.MilneEddington(regions, lines, nthreads=n_threads) # multiple threads
print("info::MilneEddington ready ")

model_guess = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
models_guess  = me.repeat_model(model_guess, nx, ny)
to_fit = stokes[:,:,:,:] # !


#me inversion
model_out, syn_out, chi2 = me.invert(models_guess, to_fit, noise, nRandom = 10, nIter=25, chi2_thres=1.0, verbose=False)
model_smoothened = gaussian_filter(model_out,(2,2,0))

model_out, syn_out, chi2 = me.invert_spatially_regularized(model_smoothened, to_fit, noise, mu = 1.0, nIter = 25, chi2_thres = 0.1, alpha = 2., alphas = np.float64([1,1,1,0.1,0.1,0.1,0.01,0.001,0.001]))
#model_smoothened = gaussian_filter(model_out[0],(2,2,0))
#model_out, syn_out, chi2 = me.invert_spatially_regularized(model_smoothened, to_fit, noise, mu = 1.0, nIter = 25, chi2_thres = 0.1, alpha = 1., alphas = np.float64([1,1,1,1,3,3,5,5,5]))
print("info:: model shape is: ", model_out.shape)

#save results
hdu1 = fits.PrimaryHDU(model_out[0])
hdu2 = fits.ImageHDU(syn_out[0])
hdulist = fits.HDUList([hdu1,hdu2])
hdulist.writeto('sr_inverted.fits',overwrite=True)



