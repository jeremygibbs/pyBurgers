import numpy as np

class Burgers(object):
	
	# model class initialization
	def __init__(self,inputObj):
		"""Constructor method
		"""
		# set the input
		self.input  = inputObj
		
		# set superclass variables
		self.dt   = self.input.dt
		self.nt   = self.input.nt
		self.visc = self.input.visc
		self.namp = self.input.namp

	@staticmethod
	def get_model(key,inputObj):
		
		if key=="dns":
			from .burgers_dns import DNS
			model = DNS(inputObj)
			model.mode = "dns"
			return model
		if key=="les":
			from .burgers_les import LES
			model = LES(inputObj)
			model.mode = "les"
			return model