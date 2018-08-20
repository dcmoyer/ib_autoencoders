import numpy as np
import keras.backend as K
import tensorflow as tf

class Analyzer(object):
	def __init__(self, model):
		self.model = model

	def run_analyses(self, suite):
		if suite== 'all':
			pass #methods = [vis_reconstruction, ]
		elif suite == '2d':
			pass
		elif suite == 'disentanglement':
			pass	
		elif suite == 'loss':
			pass
		elif suite == 'visualize':
			pass
	def vis_reconstruction(self):
		raise NotImplementedError
	def vis_manifold(self):
		raise NotImplementedError
	def vis_classes(self):
		raise NotImplementedError
	def vis_traversals(self):
		raise NotImplementedError
	def calc_mig(self):
		raise NotImplementedError
	def calc_kim_disentanglement(self):
		raise NotImplementedError

	def calc_tc(self):
		raise NotImplementedError

	def 