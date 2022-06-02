import numpy as np

class Burgers(object):
	
	# model class initialization
	def __init__(self,inputObj):
		"""Constructor method
		"""
		# set the input
		self.input  = inputObj
		
		# set superclass variables
		self.nx   = 0
		self.mp   = 0
		self.dx   = 0
		self.dt   = self.input.dt
		self.nt   = self.input.nt
		self.visc = self.input.visc
		self.namp = self.input.namp
	
	# def __init__(self,inputSCM,sfc):
	# 	self.input = inputSCM
	# 	self.sfc   = sfc
	# 	self.ell   = np.zeros(self.input.nz)
	# 	self.gM    = np.zeros(self.input.nz-1)
	# 	self.gH    = np.zeros(self.input.nz-1)
	# 	self.Km    = np.zeros(self.input.nz)
	# 	self.Kh    = np.zeros(self.input.nz)

	@staticmethod
	def get_model(key,inputObj):
		
		if key=="dns":
			from .burgers_dns import DNS
			return DNS(inputObj)
		if key=="les":
			from .burgers_les import LES
			return LES(inputObj)