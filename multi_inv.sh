#!/bin/bash

for i in {0..2850..150}
do
	echo "We are now inverting snapshot number $i"
	#echo "/dat/milic/3D/3D_snapi_cubes/loc_dyn_32_32_16_""$i""_tumag.f0"
	#python invert_snapi_using_pymilne.py /dat/milic/3D/3D_snapi_cubes_2/loc_dyn_32_32_16_""$i""_tumag.f0 5250 64 /dat/milic/3D/l2/loc_dyn_32_32_16_""$i""_tumag_l2.fits
	python invert_snapi_wfa.py /dat/milic/3D/3D_snapi_cubes/loc_dyn_32_32_16_""$i""_tumag.f0 tumag_finegrid.dat 5172 150 350 1 /dat/milic/3D/l2/loc_dyn_32_32_16_""$i""_tumag_l2_wfa.fits
	echo "Done!"
done
