import numpy as np

class Burgers(object):
	
	# model class initialization
	def __init__(self,inputObj):
		"""Constructor method
		"""
		# set the class values
		self.input = inputObj
		self.dt    = self.input.dt
		self.nt    = self.input.nt
		self.visc  = self.input.visc
		self.namp  = self.input.namp
		self.nx    = 0
		self.dx    = 0
		self.u     = 0
		
	@staticmethod
	def get_model(mode,inputObj):
		if mode=="dns":
			from .burgers_dns import DNS
			return DNS(inputObj)
		if mode=="les":
			from .burgers_les import LES
			return LES(inputObj)