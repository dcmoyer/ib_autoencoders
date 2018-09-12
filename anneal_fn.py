def kl_warmup(epoch, warmup_time = 100):
	if epoch <= warmup_time:
		return (epoch)/warmup_time
	else:
		return 1.0