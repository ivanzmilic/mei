import MilneEddington
import numpy as np
from astropy.io import fits
import sys
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import xtalk as xt

# Let's write a routine that inverts a single cube of MiHI data
# using the Milne-Eddington code
def invert_one_cube(stokescube, wavelengt, nthreads=1, full_output=False):

    # Extract dimensions
    nx = stokescube.shape[0]
    ny = stokescube.shape[1]
    nl = stokescube.shape[3] 

    # normalize
    Iqs = np.mean(stokescube[:,:,0,-1])
    stokescube /= Iqs

    #noise
    noise_level = 2.577e-2 # This is contrast with the MiHI paper! (Or maybe not so much because this is for Stokes profiles?)
    noise = np.zeros([4,nl]) # need to change
    noise += noise_level
    noise[1] /= 2.0
    noise[2] /= 2.0
    noise[3] /= 2.0 
    #apply huge noise to the telluric lines - if there are any, there are none in this case

    #model
    regions = [[wavelengt[:],None]] # select the wavelength
    lines = [5892,5893] # lines to calculate
    me = MilneEddington.MilneEddington(regions, lines, nthreads=nthreads) # multiple threads
    print("info::MilneEddington set up ")

    model_guess = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
    models_guess  = me.repeat_model(model_guess, nx, ny)
    to_fit = stokescube[:,:,:,:] 

    #me inversion
    model_out, syn_out, chi2 = me.invert(models_guess, to_fit, noise, nRandom = 10, nIter=25, chi2_thres=0.1, verbose=False)
    model_smoothened = gaussian_filter(model_out,(2,2,0))

    model_out, syn_out, chi2 = me.invert_spatially_regularized(model_smoothened, to_fit, noise, mu = 1.0, nIter = 25, \
            chi2_thres = 0.1, alpha = 5., alphas = np.float64([5,5,5,0.1,0.1,0.1,0.01,0.001,0.001]))
    # Settled on the alpha of 5 - not sure about the other regularizations. might be actually smart to have them stronger. 
    
    if full_output:
        return model_out, syn_out, chi2
    else:
        return model_out



# ME for inversion of MiHI timeseries:
filepath = sys.argv[1]
start = int(sys.argv[2])
number = int(sys.argv[3])
n_threads = int(sys.argv[4])

name_start = 'img_fit.'
name_end  = '.00.00.lr0500.cube.fits'
step = 1600

l_start = 40
l_end   = 100

print("info::reading the first input file...")

# Define the output arrays:
model_all = 0
syn_all   = 0
chi2_all  = 0

for i in range(number):
    if (i==0):
        filename = filepath+name_start+str(start)+'..'+str(start+step)+name_end
    else:
        start += step
        filename = filepath+name_start+str(start)+'..'+str(start+step)+name_end

    stokes = fits.open(filename)[0].data[12:-12, 12:-12,:,:]
    #stokes = fits.open(filename)[0].data

    stokes = xt.correct_x_talk_simple(stokes, plot_results=False)
    #exit();
    stokes = stokes[:,:,:,l_start:l_end]

    #print("info::stokes cube read. shape is: ", stokes.shape)
    ll = fits.open(filename)[1].data[l_start:l_end]
    #print("info::wavelength cube read. shape is: ", ll.shape)

    # Invert the cube    
    model_out, syn_out, chi2 = invert_one_cube(stokes, ll, nthreads=n_threads, full_output=True)
    #print("info:: model shape is: ", model_out.shape)

    # If it is the first cube, define the output arrays:
    if (i==0):
        model_all = np.zeros([number, model_out.shape[1], model_out.shape[2], model_out.shape[3]], dtype=np.float64)
        syn_all   = np.zeros([number, syn_out.shape[1], syn_out.shape[2], syn_out.shape[3], syn_out.shape[4]], dtype=np.float64)
        #chi2_all  = np.zeros([number, chi2.shape[0]], dtype=np.float64) 
   
    model_all[i] = model_out[0]
    syn_all[i]   = syn_out[0]
    #chi2_all[i]  = chi2[0]
        

#plot something simple:
import matplotlib.pyplot as plt 
mean_o = np.mean(stokes[:,:,0,:],axis=(0,1))
mean_f = np.mean(syn_out[0,:,:,0,:], axis=(0,1))
plt.plot(mean_o, label='obs')
plt.plot(mean_f, label='fit')
plt.legend()
plt.tight_layout()
plt.savefig("test.png",bbox_inches='tight')

#plot parameter maps:
Blos = model_out[0,:,:,0]*np.cos(model_out[0,:,:,1])
Btrans = model_out[0,:,:,0]*np.sin(model_out[0,:,:,1])
phi = model_out[0,:,:,2]

plt.figure(figsize=(11,6))
plt.subplot(2,3,1)
plt.imshow(Blos.T, origin='lower', cmap='PuOr', vmin=-1000, vmax=1000)
plt.colorbar()
plt.title('Blos [G]')
plt.subplot(2,3,2)
plt.imshow(Btrans.T, origin='lower', cmap='cividis', vmin=0, vmax=1000)
plt.colorbar()
plt.title('Btrans [G]')
plt.subplot(2,3,3)
plt.imshow(phi.T, origin='lower', cmap='twilight', vmin=0, vmax=np.pi)
plt.colorbar()
plt.title('phi [rad]')
plt.subplot(2,3,4)
plt.imshow(model_out[0,:,:,3].T, origin='lower', cmap='bwr', vmin=-4, vmax=4)
plt.colorbar()
plt.title('Velocity [km/s]')
plt.subplot(2,3,5)
plt.imshow(model_out[0,:,:,4].T, origin='lower', cmap='inferno')
plt.colorbar()
plt.title('Line width [nm]')
plt.subplot(2,3,6)
plt.imshow((model_out[0,:,:,7]+model_out[0,:,:,8]).T, origin='lower', cmap='inferno', vmin=0.7, vmax=1.3)
plt.colorbar()
plt.title('Continuum intensity')       


plt.tight_layout()
plt.savefig("maps.png", bbox_inches='tight')


#save results
hdu1 = fits.PrimaryHDU(model_all)
hdu2 = fits.ImageHDU(syn_all)
hdulist = fits.HDUList([hdu1,hdu2])
hdulist.writeto('sr_inverted.fits',overwrite=True)



