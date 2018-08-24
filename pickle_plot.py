import pickle 
import os
import matplotlib.pyplot as plt

#fn = 'echo_'
search_dir = 'results/'
recon = 'bce_loss_loss'
reg = 'mi_echo_z_loss'
#noise_type = 'multiplicative'
noise_type = 'additive'
train_recon = []
train_reg = []
for root, dirs, files in os.walk(search_dir): #os.getcwd()):
    for fn in files:
        print("File ", fn)
        with open(os.path.join(search_dir, fn), "rb") as pkl_data:
            results = pickle.load(pkl_data)
        if noise_type in fn:
            if len(train_reg) == 0:
                print("KEYS: ", results.keys())
            train_recon.append(results[recon][-1])
            train_reg.append(results[reg][-1])

plt.scatter(train_reg, train_recon)
plt.savefig("echo_add.png")