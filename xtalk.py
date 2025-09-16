import numpy as np
import matplotlib.pyplot as plt 

def correct_x_talk_simple(stokescube, plot_results = False, silent = True):

	#1 corr xtalk from I, using continuum means
	mean = np.mean(stokescube[:,:,:,:], axis=(0,1))
	print("info::correct_x_talk_simple:: shape of the mean spectrum is: ", mean.shape)

	# The correction will work only on the Stokes cube that is normalized to the continuum!
	Iqs = np.mean(mean[0,100:230])
	stokescube /= Iqs
	mean /= Iqs

	#I-V
	a = np.mean(mean[3][100:230])
	b1 = np.mean(mean[1][100:230])
	g1 = np.mean(mean[2][100:230])
	if (not silent):
		print("info::correct_x_talk_simple:: I to V crosstalk is: ", a)
		print("info::correct_x_talk_simple:: I to Q crosstalk is: ", b1)
		print("info::correct_x_talk_simple:: I to U crosstalk is: ", g1)

	#def new stk
	stk_new = np.zeros(stokescube.shape)
	stk_new[:,:,0,:] = stokescube[:,:,0,:]
	stk_new[:,:,1,:] = stokescube[:,:,1,:] - b1*stokescube[:,:,0,:]
	stk_new[:,:,2,:] = stokescube[:,:,2,:] - g1*stokescube[:,:,0,:]
	stk_new[:,:,3,:] = stokescube[:,:,3,:] - a*stokescube[:,:,0,:]

	if (plot_results):
		plt.figure(figsize=(8,5))
		plt.plot(mean[1,50:500], label='Stokes Q old')
		plt.plot(mean[2,50:500], label='Stokes U old')
		plt.plot(mean[3,50:500], label='Stokes V old')
		plt.plot(np.mean(stk_new[:,:,1,50:500],axis=(0,1)), label='Stokes Q new')
		plt.plot(np.mean(stk_new[:,:,2,50:500],axis=(0,1)), label='Stokes U new')
		plt.plot(np.mean(stk_new[:,:,3,50:500],axis=(0,1)), label='Stokes V new')
		plt.legend();
		#plt.xlim([])
		plt.savefig("x_talk_debug.png", bbox_inches='tight')

	return stk_new