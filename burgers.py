#!/usr/bin/env python
import argparse
import sys
import time
import numpy as np
from models import DNS, LES
from utils import io

# custom error message for user entry
class InvalidMode(Exception):
	pass 
	
# main program to run pyBurgers
if __name__ == "__main__":

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
	try:
		if mode == "dns":
			burgers = DNS(inputObj,outputObj)
		elif mode == "les":
			burgers = LES(inputObj,outputObj)
		else:
			raise InvalidMode('Error: Invalid mode (must be \"dns\" or \"les\")')
	except InvalidMode as e:
		print(e)
		sys.exit(1)
	
	# let's time this thing
	t1 = time.time()
	
	# run the model
	burgers.run()
	
	# time info
	t2 = time.time()
	tt = t2 - t1
	print("\n[pyBurgers: Run] \t Done! Completed in %0.2f seconds"%tt)
	print("##############################################################")