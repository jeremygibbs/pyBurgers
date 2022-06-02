#!/usr/bin/env python
import argparse
import sys
import time
import numpy as np
from models import Burgers
from utils import io

# class Burgers:
# 	
# 	# model class initialization
# 	def __init__(self,inputSCM,outputSCM,mode):
# 		"""Constructor method
# 		"""
# 		# set the input and output fields
# 		self.input  = inputSCM
# 		self.output = outputSCM
# 		self.mode   = mode
# 		
# 		# inform users of the simulation type
# 		print("[pyBurgers: Info] \t You are running in %s mode"%self.mode.upper())
# 		
# 		# read configuration variables
# 		print("[pyBurgers: Setup] \t Reading input settings")
# 		nx         = self.input.nz
# 		sgs_model  = self.input.les.sgs
# 		
# 	print("[NSSL-SCM: Setup] \t Creating output file")    
# 	# set reference to output dimensions
# 	self.output_dims = {
# 		't':0,
# 		'zf':nz,
# 		'zh':nz-1
# 	}
# 	self.output.set_dims(self.output_dims)
# 	
# 	# set reference to output fields
# 	self.output_fields = {
# 		'zf':self.zf,
# 		'zh':self.zh[:-1],
# 		'zi':self.zi,
# 		'u':self.U[:-1],
# 		'v':self.V[:-1],
# 		'T':self.T[:-1],
# 		'l':self.pbl.ell,
# 		'L':self.obukL,
# 		'us':self.ustar,
# 		'wu':self.mflux,
# 		'wT':self.hflux,
# 		'Km':self.Km,
# 		'Kh':self.Kh
# 	}
# 	if pbl_model==2 or pbl_model==3:
# 		self.output_fields['tke'] = self.tke
# 	self.output.set_fields(self.output_fields)
# 	
# 	# write initial data
# 	self.output.save(self.output_fields,0,0,initial=True)
# 
# 	# main run loop
# 	def run(self):
# 		"""Main run loop of pyBurgers
# 		"""
# 		
# 		# config variables
# 		tt        = self.input.tt
# 		dt        = self.input.dt
# 		nt        = int(tt*3600//dt)
# 		nz        = self.input.nz
# 		dz        = self.input.dz
# 		fc        = self.input.fc
# 		Ug        = self.input.ug
# 		Vg        = self.input.vg
# 		lapse     = self.input.lapse
# 		ns        = self.input.n_save

# main program to run pyBurgers
if __name__ == "__main__":

	# let's time this thing
	t1 = time.time()

	# a nice welcome message
	print("##############################################################")
	print("#                                                            #")
	print("#                   Welcome to pyBurgers                     #")
	print("#      A fun tool to study turbulence using DNS and LES      #")
	print("#                                                            #")
	print("##############################################################")
	
	# get case from user
	parser = argparse.ArgumentParser(description="Run a case with pyBurgers")
	parser.add_argument("-m", "--mode", dest='mode', action='store', 
						type=str, default="dns", help="Choose dns or les")
	parser.add_argument("-o", "--output", dest='outfile', action='store', 
						type=str, help="Output file name")
	args = parser.parse_args()
	mode = args.mode
	outf = args.outfile
	
	# create Input instance
	namelist = 'namelist.json'
	inputObj = io.Input(namelist)

	# create Output instance
	if not outf:
		outf='pyburgers_%s.nc'%mode
	outputObj = io.Output(outf)

	# create Burgers instance
	burgers = Burgers.get_model(mode,inputObj)

	# run the model
	burgers.run()

	# time info
	t2 = time.time()
	tt = t2 - t1
	print("\n[pyBurgers: Run] \t Done! Completed in %0.2f seconds"%tt)
	print("##############################################################")