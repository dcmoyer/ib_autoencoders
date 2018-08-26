import pickle 
import os
import matplotlib.pyplot as plt
from collections import defaultdict
#fn = 'echo_'
search_dir = 'results/'
recon = 'bce_loss'
#reg = 'vae'
reg = 'mi_echo_z_loss'
#noise_type = 'multiplicative'
noise_type = 'additive_' #'additive'
train_recon = []
train_reg = []
recon_over_time = defaultdict(lambda: defaultdict(list))
for root, dirs, files in os.walk(search_dir): #os.getcwd()):
    for fn in files:
        print("File ", fn)
        with open(os.path.join(search_dir, fn), "rb") as pkl_data:
            results = pickle.load(pkl_data)
        if noise_type in fn:
            for k in results.keys():
            	if recon in k:
            		recon = k
            	if reg in k:
            		reg = k
            	if "lagr" in k:
            		print(k, " : ", results[k][-1][-1])
            	else:
            		print(k, " : ", results[k][-1])	            
            #if len(train_reg) == 0:
            #    print("KEYS: ", results.keys())
            train_recon.append(results[recon][-1])
            train_reg.append(results[reg][-1])
            recon_over_time[round(results[reg][-1], 0)][reg] = results[recon]
print(recon_over_time.keys())
print(recon_over_time[11.0])
plt.scatter(train_reg, train_recon)
plt.savefig(str("lagr_plots"+noise_type+".png"))