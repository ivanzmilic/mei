import MilneEddington
import numpy as np
from astropy.io import fits
import sys
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import xtalk as xt

# ME for inversion of MiHI timeseries:
filepath = sys.argv[1]
start = int(sys.argv[2])
number = int(sys.argv[3])

name_start = 'img_fit.'
name_end  = '.00.00.lr0500.cube.fits'
step = 1600

l_start = 40
l_end   = 100

print("info::reading the first input file...")

filename = filepath+name_start+str(start*step)+'..'+str(start*step+step)+name_end

stokes = fits.open(filename)[0].data[12:-12, 12:-12,:,:]
stokes = xt.correct_x_talk_simple(stokes, plot_results=False)
#exit();
stokes = stokes[:,:,:,l_start:l_end]

print("info::stokes cube read. shape is: ", stokes.shape)
ll = fits.open(filename)[1].data[l_start:l_end]
print("info::wavelength cube read. shape is: ", ll.shape)
    
# Extract dimensions
nx = stokes.shape[0]
ny = stokes.shape[1]
nl = stokes.shape[3] 

# normalize
Iqs = np.mean(stokes[:,:,0,-1])
stokes /= Iqs

#noise
noise_level = 2.577e-2
noise = np.zeros([4,nl]) # need to change
noise += noise_level
noise[1] /= 0.5
noise[2] /= 0.5
noise[3] /= 2. 
#apply huge noise to the telluric lines - if there are any

#model
regions = [[ll[:],None]] # select the wavelength
lines = [5892,5893] # lines to calculate
#n_threads = int(sys.argv[2]) # input number of cores as an argument
n_threads = int(sys.argv[4])
me = MilneEddington.MilneEddington(regions, lines, nthreads=n_threads) # multiple threads
print("info::MilneEddington set up ")

model_guess = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
models_guess  = me.repeat_model(model_guess, nx, ny)
to_fit = stokes[:,:,:,:] # !


#me inversion
model_out, syn_out, chi2 = me.invert(models_guess, to_fit, noise, nRandom = 10, nIter=25, chi2_thres=0.1, verbose=False)
model_smoothened = gaussian_filter(model_out,(2,2,0))

model_out, syn_out, chi2 = me.invert_spatially_regularized(model_smoothened, to_fit, noise, mu = 1.0, nIter = 25, chi2_thres = 0.1, alpha = 2., alphas = np.float64([1,2,2,0.1,0.1,0.1,0.01,0.001,0.001]))
#print("info:: model shape is: ", model_out.shape)

#plot something simple:
import matplotlib.pyplot as plt 
mean_o = np.mean(stokes[:,:,0,:],axis=(0,1))
mean_f = np.mean(syn_out[0,:,:,0,:], axis=(0,1))
plt.plot(mean_o, label='obs')
plt.plot(mean_f, label='fit')
plt.legend()
plt.tight_layout()
plt.savefig("test.png",bbox_inches='tight')

#save results
hdu1 = fits.PrimaryHDU(model_out[0])
hdu2 = fits.ImageHDU(syn_out[0])
hdulist = fits.HDUList([hdu1,hdu2])
hdulist.writeto('sr_inverted.fits',overwrite=True)



