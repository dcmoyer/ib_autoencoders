import numpy as np
import json
import itertools
from utils import make_dir


def Session(object):

	# LOOP OVER CONFIG FILES ?  or LOOP within CONFIG FILES
	def __init__(self, config = None):
		self.model_args = None

		if config is not None:
			self.run_from_config(config)

		
	def load_config(self, config):
		# save all files from session in one folder
		self.folder = split(config, '.')[:-1]
		make_dir(self.folder)

		with open(os.path.join(os.getcwd(), config)) as conf:
    		config_dict = json.load(conf)

	    self.model_args = self._parse_config(config_dict)
		print(model_args)


	def run_from_config(self, config):
		self.load_config(config)	
		self.run_all()


	def run_all(self):
		if self.model_args is None:
			raise ValueError("Please enter model args")
		else:
			pass
		    #for job in self.model_args:
	    	#	arg_dump = json.dumps(self.model_args)
	    	# run_dummy script with args_dict as argument (inside, have json.loads(argv[1]))
	    	# bash run_model.py arg_dump .... but how to get new script?
	    	# yield args_dict


	def _parse_config(self, config)
		# for each dictionary item which is a list
		product_list = []
		key_list = []
		for arg_key, arg_value in config.items():
			# list of lists should be appended as such
			if isinstance(arg_value, list) and isinstance(arg_value[0], list):
				#EXCEPTIONS? all taken care of by above???
				# if arg_key == 'latent_dims' and not isinstance(arg_value[0], list):
				# 	# latent_dims is a list, so don't want to take product over individual entries
				# 	product_list.append([arg_value])
				# elif arg_key == 'anneal_sched':
				# 	pass
				# else:
				product_list.append(arg_value)	
			else:
				product_list.append([arg_value])
			key_list.append(arg_key)
			arg_lists = itertools.product(*product_list)
			for model_args in range(len(argument_lists)):
				arg_dicts.append(dict(zip(key_list, model_args)))
		
		print(arg_dicts[0])
		print(arg_dicts[1])
		return arg_dicts # list of argument dictionaries

	def plot_rd(self):
		raise NotImplementedError
	def plot_all_losses(self):
		raise NotImplementedError
	def save_all_losses(self):
		raise NotImplementedError
