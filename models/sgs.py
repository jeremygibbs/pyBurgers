import numpy as np

class SGS(object):
		
	@staticmethod
	def get_model(model,input):
		
		if model == 0:
			from .sgs import SGS
			return SGS(input)
		if model == 1:
			from .sgs_smagcon import SmagConstant
			return SmagConstant(input)
		if model == 2:
			from .sgs_smagdyn import SmagDynamic
			return SmagDynamic(input)
		if model == 3:
			from .sgs_wonglilly import WongLilly
			return WongLilly(nx,dx)
		if model == 3:
			from .sgs_deardorff import Deardorff
			return Deardorff(input)
	
	# model class initialization
	def __init__(self,input):
		"""Constructor method
		"""
		
		# user values
		self.input = input
		self.nx    = input.nx
		self.dx    = input.dx
		self.dt    = input.dt
		
		# sgs terms
		self.sgs = {
			'tau'	:	np.zeros(self.nx),
			'coeff'	:	0 
		}
		
	# compute sgs terms
	def compute(self,u,dudx,tke_sgs):
		
		return self.sgs