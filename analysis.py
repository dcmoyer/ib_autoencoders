"""Utilities for visualizing Keras autoencoders."""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
import sys
#sys.path.insert(0, '../three-letter-mnist/')
#import emnist_words
matplotlib.use('Agg')
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import imread
from keras import backend as K
from keras.models import Model as KerasModel
#from dataset import Dataset
import tensorflow as tf
import pandas as pd
import os
from sklearn.cross_validation import cross_val_score
from scipy.special import entr
from collections import defaultdict
import pickle

# if isinstance( data, Dataset):
#   data = data.data
#def rd_curve(hist, test = None, legend = None, prefix = ''):
def rd_curve(folder, beta = False, savefig = True):
    recons = defaultdict(list)
    regs = defaultdict(list)
    lagrs = defaultdict(list)
    test_recons = defaultdict(list)
    test_regs = defaultdict(list)
    test_lagrs = defaultdict(list)
    param_idx = []
    offset = 2
    print("RD For folder ", folder)
    models = defaultdict(lambda: defaultdict(list))
    recon_over_time = defaultdict(lambda: defaultdict(list))
    print(os.path.join(os.getcwd(), folder))
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), folder)): #os.getcwd()):
        #print("root ", root)
        #print("dirs ", dirs)
        #print("files ", files)
        for fn in files:
        #print("File ", fn)
            if ".pickle" in fn:
                prefix = fn.split(".pickle")[0].split("_")[:2]
                param = fn.split(".pickle")[0]
                print("multiplicative? ", len(param.split("multiplicative")[-1]))
                mult = 0 if len(param.split("multiplicative")[-1])<1 else -1
                add = -1 if len(param.split("additive")[0]) >= len(param.split("additive")[-1]) else 0 #or len(param.split("additive")[-1])>0
                param = fn.split(".pickle")[0].split("additive")[add].split("multiplicative")[mult].split("_")[-1]
                #print(fn.split(".pickle")[0])
                #print(fn.split(".pickle")[0].split("additive")[-1])
                #print(fn.split(".pickle")[0].split("additive")[-1].split("multiplicative")[-1])
                #print(fn.split(".pickle")[0].split("additive")[-1].split("multiplicative")[-1].split("_")[-1])
                param_idx.append(param)
                #print("prefix : ", prefix)
                #print(fn)

                with open(os.path.join(os.getcwd(), folder, fn), "rb") as pkl_data:
                    try:
                        results = pickle.load(pkl_data)
                    except:
                        print("Could not open ", fn)
                        continue
                    #print(list(results.keys()))
                    for loss in results.keys():
                        k = loss.split("/")[0]
                        #print(loss)
                        if 'test' not in loss:
                            if 'recon' in loss:
                                recons[k].append(results[loss][-1] if 'test' not in loss else results[loss])
                                try:
                                    test_recons[k].append(results[str('test_'+loss)])
                                except:
                                    test_recons[k].append(results[str('test_'+k+'_loss')])
                            elif 'reg' in loss:
                                regs[k].append(results[loss][-1] if 'test' not in loss else results[loss])
                                try:
                                    test_regs[k].append(results[str('test_'+loss)])
                                except:
                                    test_regs[k].append(results[str('test_'+k+'_loss')])

                            elif 'lagr' in loss:
                                if 'test' in loss:
                                    warn('*** TEST LAGRANGIAN EXISTS **** for ', fn)
                                lagrs[k].append(results[loss][-1] if 'test' not in loss else results[loss])
                                #test_lagrs[loss].append(results[str('test_'+loss)])
    csv_str=''
    print("Params ", param_idx)
    # rows = increasing param value
    for i in sorted(range(len(param_idx)), key=lambda k: param_idx[k]): #range(len(param_idx)):
        # headers
        if csv_str=='':
            #print("len params ", len(param_idx))
            csv_str = "{} \n".format(fn)
            csv_str += 'Param \t'
            for k in regs.keys():
                csv_str += "{} \t".format(k.split('_')[:-2]) #k.split('loss')[:-2]
            for k in recons.keys():
                csv_str += "{} \t".format(k.split('_')[:-2])
                csv_str += "test_{} \t".format(k.split('_')[:-2])
            for k in lagrs.keys():
                csv_str += "{} \t".format(k.split('_')[:-2])
                #csv_str += "test_{} \t".format(k)
        csv_str += "\n"
        csv_str += '{} \t'.format(param_idx[i])
        # columns
        for k in regs.keys():
            csv_str += "{} \t".format(round(regs[k][i],2))
        for k in recons.keys():
            csv_str += "{} \t".format(round(recons[k][i],2))
            csv_str += "{} \t".format(round(test_recons[k][i]),2)
        for k in lagrs.keys():
            csv_str += "{} \t".format(round(lagrs[k][i]),2)

        csv_str += "\n"

        #knn_results += "Echo,\t\t{:0.4f}\n".format(knn_score)
        with open('{}/results.txt'.format(folder), 'w') as f:
            f.write(csv_str)
        #csv.write(param_idx[i], regs[k][i] for k in regs.keys(), recons[k][i] for k in recons.keys())

    for reg in regs.keys():
        for recon in recons.keys():
            if 'beta' in fn or beta:
                val, idx = min((val, idx) for (idx, val) in enumerate(param_idx))
            else: # constrainted optimization, maximal regularizer
                val, idx = max((val, idx) for (idx, val) in enumerate(param_idx))

            #val, idx = min((val, idx) for (idx, val) in enumerate(recons[recon]))
            print('AE idx ', recon, ' : ', idx, ' , ', val, ' : test recon len ', len(test_recons[recon]))
            print('reg: ', reg, ' other keys: ', list(regs.keys()))
            print('regs len ', len(regs[reg]), ' recons len ', len(recons[recon]))
            
            plt.figure(figsize=(15,15))
            plt.title(str("Train/Test "+ recon))
            # scatter except for minimum recon => dotted line

            try:
                plt.axhline(y= recons[recon][idx], color = 'b', linestyle='-', label = str('train_AE (param '+ str(round(float(val),0))+')'))
                plt.axhline(y= test_recons[recon][idx], color = 'r', linestyle='-', label= str('test_AE (param '+ str(round(float(val),0))+')'))
            except:
                plt.axhline(y= recons[recon][idx], color = 'b', linestyle='-', label = 'train_AE')
                plt.axhline(y= test_recons[recon][idx], color = 'r', linestyle='-', label= 'test_AE')
          
            plt.scatter([regs[reg][i] for i in range(len(regs[reg])) if i != idx], 
                [recons[recon][i] for i in range(len(recons[recon])) if i != idx])
            plt.scatter([test_regs[reg][i] for i in range(len(test_regs[reg])) if i != idx], 
                [test_recons[recon][i] for i in range(len(test_recons[recon])) if i != idx])

            plt.legend()
            for i in range(len(regs[reg])):
                plt.annotate(str(round(float(param_idx[i]),1)), xy=(regs[reg][i]-offset, recons[recon][i]-offset), size = 'large')
            

            plt.savefig(os.path.join(folder, str(*recon.split('loss')[:-1])+'_'+str(*reg.split('loss')[:-1])+'.pdf'), bbox_inches='tight')
            
    

        for lagr in lagrs.keys():
            plt.figure()
            plt.scatter(regs[reg], lagrs[lagr])
            #plt.scatter(test_regs[reg], lagrs[])
            for i in range(len(regs[reg])):
                plt.annotate(str(round(float(param_idx[i]),1)), xy=(regs[reg][i]+offset, lagrs[lagr][i]+offset))
            plt.savefig(os.path.join(folder, lagr+'_'+str(*reg.split('loss')[:-1])+'.pdf'), bbox_inches='tight')
    #legend = param values


def plot_loss(hist, keys = ['loss', 'val_loss'], prefix=""):
    # print "==> plotting loss function"
    plt.clf()
    for k in keys:
        if isinstance(hist, dict):
            plt.plot(hist[k], label=k)
        else:
            if k == 'loss':
                try:
                    x = [hist.history[k][i][0][0] for i in range(len(hist.history[k]))]
                except:
                    print(hist.history[k])
            else:
                x = hist.history[k]
            plt.plot(x, label=k) 
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}loss.png'.format(prefix))
    plt.close('all')

def write_loss(hist, keys = None, prefix="", full_hist = True):
    if keys == None:
        if isinstance(hist, dict):
            keys = hist.keys()
            hist_dict = hist
        else:
            keys = hist.history.keys()
            hist_dict = hist.history

    if not full_hist and keys is not None:
        with open('{}history.csv'.format(prefix), 'a+') as f:
            for k in keys:                    
                f.write('{},{}\n'.format(k, ','.join(map(str, hist_dict[k][-1]))))
    else:
        hist_df = pd.DataFrame(columns = hist_dict.keys())
        for i in range(len(hist_dict[list(hist_dict.keys())[0]])):
            interim = pd.DataFrame([[hist_dict[k][i] for k in keys]], columns = hist_dict.keys())
            hist_df.append(interim)
        print('each ', interim.shape)
        print('entire ', hist_df.shape)
        hist_df.to_csv('{}history.csv'.format(prefix))


def plot_traversals(dataset, encoder, generator, top_dims = [], top = 10, prefix = "", 
        traversals = 13, imgs = 1, stdevs = 3, z_act = None, means = None, sigmas = None): #chunk?
    ## SPECIFY TOP_DIMS or just TOP?

    # SUPPORT FOR DATASETS?
    data = dataset
    #data = dataset.data if isinstance(dataset, Dataset) else dataset

    data_points = np.random.choice(data.shape[0], imgs, replace=False)
    if z_act is None:
        if isinstance(encoder, KerasModel):
            z_act = encoder.predict(data[data_points, :])
        else:
            z_act = encoder([data[data_points, :]])[0]
    
    encoder_dim = z_act.shape[-1]
    top = min(encoder_dim, top)

    # TAKE TOP_DIMS AS GIVEN, or the top N
    if not top_dims:
        #raise Warning("Please feed index of latent dimensions to visualize, e.g. sorted by mutual information.  Proceeding with latent dims 0:top")
        top_dims = range(top)
        #raise NotImplementedError

    if means is None or sigmas is None:
        means, sigmas = get_activation_stats(encoder, data, chunk_batch = 10000)
    means = means[top_dims]
    sigmas = sigmas[top_dims]
    
    for data_pt in range(len(data_points)):
        traversal_data = np.zeros((top*traversals, data.shape[1]))
        for j in range(top):
            top_idx = top_dims[j]
            #z_act = z_act[:, top_indices]
            #print('top ', j, ' kl at index ', l)
            for trav in range(traversals):
                deviation = -stdevs + trav*(2*stdevs)/(traversals-1)
                z_perturbed = np.array(z_act[data_pt, :])
                z_perturbed[top_idx] = z_perturbed[top_idx] + deviation*sigmas[top_idx]
                
                if isinstance(generator, KerasModel):
                    output = generator.predict(z_perturbed[np.newaxis, :])
                else:
                    output = generator([z_perturbed[np.newaxis, :]])[0]

                #output = generator.predict(z_perturbed[np.newaxis, :])
                traversal_data[j*traversals + trav, :] = output
    # need sigma, mu across all data

        #if isinstance(dataset, Dataset):
        #    digit_dims = dataset.digit_dims 
        #else:
        if True:
            dim_sqrt = int(np.sqrt(data.shape[-1]))
            print(dim_sqrt)
            if dim_sqrt**2 == data.shape[-1] or (dim_sqrt +.5)**2 == data.shape[-1]: # hacky int check
                digit_dims = [dim_sqrt, dim_sqrt]
            else:
                raise ValueError("Specify dim1, dim2 in a Dataset object and feed as argument.")

        # transpose?
        figure = np.ones(((digit_dims[0])*(top), digit_dims[1] *(traversals)))
        
        row = -1

        for i in range(traversal_data.shape[0]):
            if i % traversals == 0:
                row = row + 1
                k = 0
            image = traversal_data[i, :].reshape(
                digit_dims[0], digit_dims[1])
            figure[row * digit_dims[0]: (row+1) * digit_dims[0], 
                k * digit_dims[1]: (k+1) * digit_dims[1]] = image
            k = k+1
        # sizing?    
        plt.figure(figsize=(24, 72)) 
        plt.imshow(figure)
        plt.axis('off')
        plt.savefig('{}_latent_traversals_{}.png'.format(prefix, str(data_pt)), bbox_inches='tight')
        plt.close('all')




# ADD KL DIVERGENCE CALCULATION? or get it from loss function
def get_activation_stats(encoder, data, chunk_batch = 10000):
    x2 = 0
    means = 0
    #calculating max KL from prior?
    for chunk in range(max(1, int(data.shape[0]/chunk_batch))):
        chunk_data = data[chunk*chunk_batch: min(data.shape[0], (chunk+1)*chunk_batch), :]
        
        if isinstance(encoder, KerasModel):
            z_act = encoder.predict(chunk_data)
        else:
            z_act = encoder([chunk_data])[0]
        #z_act = encoder.predict(chunk_data)
        encoder_dim = z_act.shape[-1]
        #z_act = my_predict(encoder, chunk_data, encoder.layers[-1].name)       

        #z_vars = merged_decode[:, :encoder_dim]
        #z_means = merged_decode[:, encoder_dim:]
        
        x2 = x2 + np.sum(z_act**2, axis = 0)
                #np.sqrt(np.var(all_z, axis =0))
        means = means + np.sum(z_act, axis =0)
        #merged_decode = get_z_merged([chunk_data])[0]
        #chunk_kl = kl_func(merged_decode = merged_decode)
        #kl = kl + chunk_kl

    means = 1.0/data.shape[0]*means
    sigmas = np.sqrt(1.0/data.shape[0]*x2 - means ** 2)
    #kl = np.divide(kl, data.shape[0])

    return means, sigmas


def vis_reconstruction(model, data, prefix='', noise=None, n=5, batch = 100, num_losses=1):
    # print "==> visualizing reconstructions, prefix = {}".format(prefix)
    if isinstance(dataset, Dataset):
        digit_dims = dataset.digit_dims 
    else:
        dim_sqrt = int(np.sqrt(data.shape[-1]))
        if (dim_sqrt +.5)**2 == data.shape[-1]: # hacky int sqrt check
            digit_dims = [dim_sqrt, dim_sqrt]
        else:
            raise ValueError("Specify dim1, dim2 in a Dataset object and feed as argument.")

    figure = np.ones((digit_dims[0] * 3, (digit_dims[1]+1) * n))

    # print 'DATA SHAPE.... ', data.shape
    data_dim = data.shape[1]
    # if merged:
    #    dummy = Model(input = model.input, output = model.output[:-1, :data_dim])
    #    xbars = dummy.predict(data)

    if noise is None:
        #inp = [data]*num_losses if num_losses > 1 else data
        #xbars = my_predict(model, inp, 'decoder')
        xbars = model.predict(inp, batch_size = batch)

    else:
        data_noise = noise(data)
        inp = [data_noise]*num_losses if num_losses > 1 else data_noise
        if batch is not None:
            xbars = model.predict(inp, batch_size = batch)
        else:
            xbars = model.predict(inp)

    if isinstance(xbars, list) and len(xbars) > 1:
        i = 0
        while xbars[i].shape[1] != digit_dims[0]*digit_dims[1]:
            i = i+1
        xbars = xbars[i]

    for i in range(n):
        ind = i
        # can ask for different (e.g. random) index if required 
        digit = data[ind].reshape(digit_dims[0], digit_dims[1])
        digit_decoded = xbars[ind, :data_dim].reshape(
            digit_dims[0], digit_dims[1])
        figure[0 * digit_dims[0]: (0 + 1) * digit_dims[0],
               i * digit_dims[1]: (i + 1) * digit_dims[1]] = digit
        if noise is not None:
            figure[1 * digit_dims[0]: (1 + 1) * digit_dims[0],
                   i * digit_dims[1]: (i + 1) * digit_dims[1]] = data_noise[i].reshape((digit_dims[0], digit_dims[1]))
            figure[2 * digit_dims[0]: (2 + 1) * digit_dims[0],
                   i * digit_dims[1]: (i + 1) * digit_dims[1]] = digit_decoded
        else:
            figure[1 * digit_dims[0]: (1 + 1) * digit_dims[0],
                   i * digit_dims[1]: (i + 1) * digit_dims[1]] = digit_decoded
            #i = i+1
    plt.figure(figsize=(12, 24))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.savefig('{}_reconstruction.png'.format(prefix), bbox_inches='tight')
    plt.close('all')
    

    
def manifold(activations, generator, per_dim = 50, dims = None):
    # UNTESTED
    lim_x = [np.percentile(activations[:,0], 0), np.percentile(activations[:,0], 100)]
    lim_y = [np.percentile(activations[:,1], 0), np.percentile(activations[:,1], 100)]
    grid_x = np.linspace(lim_x[0], lim_x[1], per_dim)
    grid_y = np.linspace(lim_y[0], lim_y[1], per_dim)
    
    figure = np.zeros((digit_size * n, digit_size * n))

    if dims is None:
        dim_sqrt = int(np.sqrt(generator.predict(activations[0,:]).shape[-1]))
        if dim_sqrt**2 == data.shape[-1] or (dim_sqrt +.5)**2 == data.shape[-1]: # hacky int check
            dims = [dim_sqrt, dim_sqrt]

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * dims[0]: (i + 1) * dims[0],
                   j * dims[1]: (j + 1) * dims[1]] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig(str(self.location + '_2d_latent.pdf'))
    plt.close()

def two_d_labeled(x, y, encoder, batch = 1000, prefix = ''):
    # UNTESTED
    indices = np.random.choice(x.shape[0], batch, replace=False)

    if isinstance(encoder, KerasModel):
        z_act = encoder.predict(x[indices, :])
    else:
        z_act = encoder([x[indices, :]])[0]
    

    if z_act.shape[1] > 2:
        z_act = self.pca_decompose(z_act)
    z1 = z_act[:, 0]
    z2 = z_act[:, 1]
    
    fig, ax = plt.subplots()
    scattr = ax.scatter(z1, z2, s=25, c = y[indices].astype(int), cmap = Set1)
    plt.colorbar(scattr, spacing='proportional')
    plt.grid()
    plt.savefig(str(prefix + '_2d_by_label.pdf'))
