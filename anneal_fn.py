
def broken_elbo_mnist(epoch, base = 3*10**-4, constant = 100, total = 200):
	if epoch < constant:
		return base
	elif epoch < total:
		return (1 - (epoch - constant)/(total-constant))*base
	else:
		raise ValueError('Epoch outside anneal range of Alemi et al 2018 range. Modify "total" argument in anneal_fn')
