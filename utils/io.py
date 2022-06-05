import json
import sys
import time
import netCDF4 as nc
import numpy as np

class Input(object):

	def __init__(self, namelist):
		
		# open and parse the json namelist
		try:
			with open(namelist) as json_file:
				data = json.load(json_file)
		# report file path error to user and exit program
		except FileNotFoundError as e:
			print('There was an issue opening \'%s\'.'%(namelist))
			print('Error: ',e.strerror)
			sys.exit(1)
		# report json parsing error to user and exit program
		except json.decoder.JSONDecodeError as e:
			print('There was an issue parsing \'%s\'.'%(namelist))
			print('Error: ',e)
			sys.exit(1)
		# process the json data
		else:
			# load data from json dictionary into local variables
			try:
				# time section
				self.dt    = data["time"]["dt"]
				self.nt    = data["time"]["nt"]
				
				# DNS section
				self.nxDNS = data["dns"]["nx"]
				
				# LES section
				self.nxLES = data["les"]["nx"]
				self.sgs   = data["les"]["sgs"]
				
				# physics section
				self.visc  = data["physics"]["visc"]
				self.namp  = data["physics"]["namp"]
				
				# output section
				self.t_save    = data["output"]["t_save"]
				self.n_save    = np.floor(self.t_save/self.dt)   
			# report a json dictionary error to user and exit program
			except (KeyError) as e:
				print("There was an issue accessing data from \'%s\'"%namelist)
				print("Error: The key",e,"does not exist")
				sys.exit(1)

class Output(object):
			
	def __init__(self,outfile):

		# create output file
		self.outfile             = nc.Dataset(outfile,'w')
		self.outfile.description = "PyBurgers output"
		self.outfile.source      = "Jeremy A. Gibbs"
		self.outfile.history     = "Created " + time.ctime(time.time())
	
		# dictionary of fields to be saved
		self.fields_time   = {}
		self.fields_static = {}
	
		self.attributes = {
			'time': {
				'dimension':("t",),
				'long_name':'time',
				'units':'s'
			},
			'x': {
				'dimension':("x",),
				'long_name':'x-distance',
				'units':'m'
			},
			'u': {
				'dimension':("t","x",),
				'long_name':'u-component velocity',
				'units':'m s-1'
			},
			'tke': {
				'dimension':("t",),
				'long_name': 'turbulence kinetic energy',
				'units': 'm2 s-2'
			}
		}
	
	# function to set the dimensions of each variable
	def set_dims(self,dims):
	
		# iterate through keys in dictionary
		for dim in dims:
			size = dims[dim]
			if size==0:
				self.outfile.createDimension(dim)
			else:
				self.outfile.createDimension(dim,size)
	
	# function to create desired output fields
	def set_fields(self,fields):
		
		# add time manually
		dims  = self.attributes['time']['dimension']
		name  = self.attributes['time']['long_name']
		units = self.attributes['time']['units']
		ncvar = self.outfile.createVariable('time', "f4", dims)
		ncvar.long_name          = name
		ncvar.units              = units
		self.fields_time['time'] = ncvar
	
		# iterate through keys in dictionary
		for field in fields:
			dims  = self.attributes[field]['dimension']
			name  = self.attributes[field]['long_name']
			units = self.attributes[field]['units']
			ncvar = self.outfile.createVariable(field, "f4", dims)
			ncvar.long_name    = name
			ncvar.units        = units
			if 't' in dims:
				self.fields_time[field] = ncvar
			else:
				self.fields_static[field] = ncvar
	
	# function to save data to the output file
	def save(self,fields,tidx,time,initial=False):
	
		# save static only for initial time
		if initial:
			for field in self.fields_static:
				self.fields_static[field][:] = fields[field]
		
		# save time
		for field in self.fields_time:
			
			dim = self.attributes[field]['dimension']
	
			if len(dim)==1:
				if field=='time':
					self.fields_time[field][tidx] = time
				else:
					self.fields_time[field][tidx] = fields[field]
			else:
				self.fields_time[field][tidx,:] = fields[field]
	
		# sync
		self.outfile.sync()
	
	# function to close the output file
	def close(self):
		self.outfile.close()