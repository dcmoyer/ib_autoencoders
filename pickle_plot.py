import pickle 
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
#fn = 'echo_'
search_dir = 'results/smalltrain/'
search_dir = 'results/vae_constrained/'
#recon = 'bce_loss'
#reg = 'vae'
#reg = 'mi_echo_z_loss'
#noise_type = 'multiplicative'
#noise_type = 'additive_' #'additive'

def check_key(fn):
    if 'echomultiplicative' in fn:
        method = 'ido'
        mi = 'ido'
    # don't use val
        #val = fn.split('multiplicative')[1].split('.')[0]
    elif 'echoadditive' in fn:
        method = 'vae'
        mi = 'vae'
        #val = fn.split('additive')[1].split('.')[0]   
    elif 'echoinfovae' in fn:
        method = 'infovae'
        mi = 'vae'
        #val = fn.split('infovae')[1].split('.')[0]
    elif 'multiplicative_1' in fn:
        method = 'echo_mult'
        mi= 'mi_echo'
        #val = fn.split('echo')[1][:3]#.split('.')[0] 
    elif 'additive_1' or 'additive_' in fn:
        method = 'echo_add'
        mi= 'mi_echo'
        #val = fn.split('echo')[1][:3]#.split('.')[0] 
    elif 'bir_vae' in fn:
        method = 'bir'
        #val = fn.split('echo')[1][:3]#.split('.')[0]
    else:
        raise ValueError("cant understand filename") 
    return str(method)#+'_'+val)

models = defaultdict(lambda: defaultdict(list))
train_recon = []
train_reg = []
recon_over_time = defaultdict(lambda: defaultdict(list))
for root, dirs, files in os.walk(search_dir): #os.getcwd()):
    print("root ", root, " dirs ", dirs)
    for fn in files:
        print("File ", fn)
        #if 'smalltrain' not in fn:
            #continue
            
        with open(os.path.join(search_dir, fn), "rb") as pkl_data:
            try:
                results = pickle.load(pkl_data)
            except:
                continue
        model_type = check_key(fn)

        for k in results.keys():
            
            if "lagr" in k:
                try:
                    final_val = results[k][-1][-1]
                except:
                    pass
            elif isinstance(results[k], list):
                final_val = results[k][-1]
            else:
                final_val = results[k]
            loss_key = k.split("/")[0]
            print(k, " : ", final_val)
            models[model_type][loss_key].append(final_val)

cols = ['method', 'compression', 'bce_train', 'bce_test', 'gap', 'mse_train', 'mse_test', 'gap']
rows = []
#df = pd.DataFrame(columns = cols)
for method in models.keys():
    print("method ", method, " keys: ", models[method].keys())
    for k in models[method].keys():
        if 'test' not in k and ('vae' in k or 'ido' in k or 'mi_echo' in k or 'bir' in k):
            train_reg = models[method][k]
            print("train_reg ", len(train_reg)) 
        elif 'test' not in k and ('bce' in k):
            if len(models[method][k]) > 2:
                bce_train = models[method][k]
            print("bce len ", len(bce_train)) 
        elif 'test' not in k and ('mse' in k):
            mse_train = models[method][k]
            print("mse_len ", len(mse_train))  
        elif 'test' in k and ('bce' in k):
            bce_test = models[method][k]
            print("bce test ", len(bce_test)) 
        elif 'test' in k and ('mse' in k):
            mse_test = models[method][k]
            print("mse test ", len(mse_test)) 
    for i in range(len(train_reg)-1):
        #df.append(pd.DataFrame(
        rows.append([method, train_reg[i], bce_train[i], bce_test[i], bce_test[i]-bce_train[i], mse_train[i], mse_test[i], mse_test[i]-mse_train[i]])#, index = [df.shape[0]], columns = cols))

    #plt.figure()
    plt.scatter(train_reg, bce_train, label = method)
    #plt.title(method)
plt.legend()
plt.savefig(str("lagr_plots"+".png"))

print(len(rows))
for c in cols:
    print(c, '\t & ', end="")
print()
for i in range(len(rows)):
    for j in range(len(rows[i])):
        print(str(round(float(rows[i][j]),2)) if j>0 else rows[i][j], '\t & ', end="")
    print('\\\\')
        # if noise_type in fn:
        #     for k in results.keys():
        #     	if recon in k:
        #     		recon = k
        #     	if reg in k:
        #     		reg = k
        #     	if "lagr" in k:
        #     		print(k, " : ", results[k][-1][-1])
        #     	else:
        #     		print(k, " : ", results[k][-1])	            
        #     #if len(train_reg) == 0:
        #     #    print("KEYS: ", results.keys())
        #     train_recon.append(results[recon][-1])
        #     train_reg.append(results[reg][-1])
        #     recon_over_time[round(results[reg][-1], 0)][reg] = results[recon]
#print(recon_over_time.keys())
#print(recon_over_time[11.0])
#plt.scatter(train_reg, train_recon)
#plt.savefig(str("lagr_plots"+noise_type+".png"))