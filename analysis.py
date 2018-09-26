"""Utilities for visualizing Keras autoencoders."""
import matplotlib
matplotlib.use('Agg')
# Force matplotlib to not use any Xwindows backend.
import sys
#sys.path.insert(0, '../three-letter-mnist/')
#import emnist_words
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
from dataset import Dataset

# if isinstance( data, Dataset):
#   data = data.data
#def rd_curve(hist, test = None, legend = None, prefix = ''):




def rd_curve(folders, beta = False, savefig = True, name = None, recon_loss = 'bce', threshold = .001):
    recons = defaultdict(lambda: defaultdict(list))
    regs = defaultdict(lambda: defaultdict(list))
    lagrs = defaultdict(lambda: defaultdict(list))
    test_recons = defaultdict(lambda: defaultdict(list))
    test_regs = defaultdict(lambda: defaultdict(list))
    test_lagrs = defaultdict(lambda: defaultdict(list))
    param_inds = defaultdict(list)
    offset = 3
    
    models = defaultdict(lambda: defaultdict(list))
    recon_over_time = defaultdict(lambda: defaultdict(list))
    #print(os.path.join(os.getcwd(), folder))
    if not isinstance(folders, list):
        folders = [folders]   
    
    for folder in folders:
        print("RD For folder ", folder)
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), folder)): #os.getcwd()):
            #print("root ", root)
            #print("dirs ", dirs)
            #print("files ", files)
            for fn in files:
            #print("File ", fn)
                if ".pickle" in fn:
                    prefix = fn.split(".pickle")[0].split("_")[:2]
                    param = fn.split(".pickle")[0]
                    #print("multiplicative? ", len(param.split("multiplicative")[-1]))
                    mult = 0 if len(param.split("multiplicative")[-1])<1 else -1
                    add = -1 if len(param.split("additive")[0]) >= len(param.split("additive")[-1]) and len(param.split("additive")[-1])>1 else 0 #
                    param = fn.split(".pickle")[0].split("additive")[add].split("multiplicative")[mult].split("_")[-1]
                    
                    #if float(param) <= threshold and float(param) > 10**-6:
                    #    continue

                    param_inds[folder].append(param)
                    

                    with open(os.path.join(os.getcwd(), folder, fn), "rb") as pkl_data:
                        try:
                            results = pickle.load(pkl_data)
                        except:
                            print("Could not open ", fn)
                            continue
                        #print(list(results.keys()))
                        if len([k for k in results.keys() if 'test' not in k and ('gaussian' in k or 'made' in k)]) >= 2:
                            for kk in [k for k in results.keys() if ('gaussian' in k or 'made' in k)]:
                                if 'test' in kk:
                                    try:
                                        results['test_mi_est_reg'] += -results[kk] if 'gaussian' in kk else results[kk]
                                    except:
                                        results['test_mi_est_reg'] = -results[kk] if 'gaussian' in kk else results[kk]
                                else:
                                    if 'gaussian' in kk: 
                                        if 'mi_est_reg' not in results:
                                            results['mi_est_reg'] = [0]*len(results[kk])
                                        if len(results['mi_est_reg']) != len(results[kk]):
                                            print()
                                        results['mi_est_reg'] = [results['mi_est_reg'][i] - results[kk][i] for i in range(len(results[kk]))]
                                    else:
                                        if 'mi_est_reg' not in results:
                                            results['mi_est_reg'] = [0]*len(results[kk])
                                        results['mi_est_reg'] = [results['mi_est_reg'][i] + results[kk][i] for i in range(len(results[kk]))]
                                    #except:  
                                    #   results['mi_est_reg'] += [-1*a for a in results[kk]] if 'gaussian' in kk else results[kk]
                                
                        for loss in results.keys():
                            k = loss.split("/")[0]
                            #print(loss, " len loss ", results[loss])
                            if 'test' not in loss:
                                print(loss)
                                if 'recon' in loss:
                                    recons[folder][k].append(results[loss][-1] if 'test' not in loss else results[loss])
                                    try:
                                        test_recons[folder][k].append(results[str('test_'+loss)])
                                    except:
                                        test_recons[folder][k].append(results[str('test_'+k+'_loss')])
                                elif 'reg' in loss or '__' in loss:
                                    if '__' in loss:
                                        key_loss = loss.split("_")
                                        for i in range(len(key_loss)):
                                            key_loss[i] = 'reg' if key_loss[i] == '' else key_loss[i]
                                        key_loss = "_".join(key_loss)
                                    else:
                                        key_loss = k 
                                    regs[folder][key_loss].append(results[loss][-1] if 'test' not in loss else results[loss])
                                    try:
                                        test_regs[folder][key_loss].append(results[str('test_'+loss)])
                                    except:
                                        test_regs[folder][key_loss].append(results[str('test_'+key_loss)])

                                elif 'lagr' in loss:
                                    if 'test' in loss:
                                        warn('*** TEST LAGRANGIAN EXISTS **** for ', fn)
                                    lagrs[folder][k].append(results[loss][-1] if 'test' not in loss else results[loss])
                                    #test_lagrs[loss].append(results[str('test_'+loss)])
                                elif loss != 'loss':
                                    print("DOING NOTHING FOR ", loss, " with fn:  ", fn)
        #write files per folder 
        csv_str=''
        print("Params ", param_inds[folder], " len ", len(param_inds[folder]))
        #print("lengths ", [(k, len(regs[folder][k])) for k in regs[folder].keys()])
        #print("lengths ", [(k, len(recons[folder][k])) for k in recons[folder].keys()])
        # rows = increasing param value
        for i in sorted(range(len(param_inds[folder])), key=lambda k: float(param_inds[folder][k])): #range(len(param_inds[folder])):
            #if ('echo' in folder and i >= len(regs[folder]['mi_echo_reg_loss'])) or ('made' in folder and i >= len(regs[folder]['made_reg_loss'])):
            #    continue
            #if ('vae' in folder and i >= len(regs[folder]['vae_reg_loss'])):
            #    continue
            # headers
            
            if csv_str=='':
                #print("len params ", len(param_inds[folder]))
                csv_str = "{} \n".format(fn)
                csv_str += 'Param \t \t'
                for k in regs[folder].keys():
                    csv_str += "{} \t \t".format(k.split('_')[0]) #k.split('loss')[:-2]
                    #csv_str += "{} \t \t".format(k.split('_')[:-2]) #k.split('loss')[:-2]
                    #print('regs ', [len(regs[folder][k]) for k in regs[folder].keys()])
                for k in recons[folder].keys():
                    csv_str += "{} \t \t".format(k.split('_')[0]) 
                    #print('rcons ', [len(recons[folder][k]) for k in recons[folder].keys()])
                for k in regs[folder].keys():
                    csv_str += "test_{} \t \t".format(k.split('_')[0])
                for k in recons[folder].keys():
                    csv_str += "test_{} \t \t".format(k.split('_')[0])
                    #csv_str += "{} \t \t".format(k.split('_')[:-2])
                    #csv_str += "test_{} \t \t".format(k.split('_')[:-2])
                for k in lagrs[folder].keys():
                    csv_str += "{} \t \t".format(k.split('_')[0])
                    #csv_str += "{} \t \t".format(k.split('_')[:-2])
                    #csv_str += "test_{} \t".format(k)
            csv_str += "\n"
            csv_str += '{} \t \t'.format(param_inds[folder][i])
            # columns
            for k in regs[folder].keys():
                csv_str += "{} \t \t".format(round(regs[folder][k][i],2))
            for k in recons[folder].keys():
                csv_str += "{} \t \t".format(round(recons[folder][k][i],2))
            for k in regs[folder].keys():
                csv_str += "{} \t \t".format(round(test_regs[folder][k][i],2))    
            for k in recons[folder].keys():
                csv_str += "{} \t \t".format(round(test_recons[folder][k][i]),2)

            for k in lagrs[folder].keys():
                csv_str += "{} \t \t".format(round(lagrs[folder][k][i]),3)

            csv_str += "\n"

            #knn_results += "Echo,\t\t{:0.4f}\n".format(knn_score)
            with open('{}/results.txt'.format(folder), 'w') as f:
                f.write(csv_str)
            #csv.write(param_inds[folder][i], regs[folder][k][i] for k in regs[folder].keys(), recons[folder][k][i] for k in recons[folder].keys())

        for reg in regs[folder].keys():
            
            for recon in recons[folder].keys():
                #print("reg ", reg, " recon ", recon)
                if 'beta' in fn or beta:
                    val, idx = min((val, idx) for (idx, val) in enumerate(param_inds[folder]))
                else: # constrainted optimization, maximal regularizer
                    val, idx = max((val, idx) for (idx, val) in enumerate(param_inds[folder]))

                #val, idx = min((val, idx) for (idx, val) in enumerate(recons[folder][recon]))
                #print('AE idx ', recon, ' : ', idx, ' , ', val, ' : test recon len ', len(test_recons[folder][recon]))
                #print('reg: ', reg, ' other keys: ', list(regs[folder].keys()))
                #print('regs len ', len(regs[folder][reg]), ' recons len ', len(recons[folder][recon]))
                
                plt.figure(figsize=(15,15))
                plt.title(str("Train/Test RD "+ folder.split('/')[-1]))
                plt.xlabel(reg)
                plt.ylabel(recon)
                # scatter except for minimum recon => dotted line

                try:
                    plt.axhline(y= recons[folder][recon][idx], color = 'b', linestyle='-', label = str('train_AE (param '+ str(round(float(val),0))+')'))
                    plt.axhline(y= test_recons[folder][recon][idx], color = 'r', linestyle='-', label= str('test_AE (param '+ str(round(float(val),0))+')'))
                except:
                    plt.axhline(y= recons[folder][recon][idx], color = 'b', linestyle='-', label = 'train_AE')
                    plt.axhline(y= test_recons[folder][recon][idx], color = 'r', linestyle='-', label= 'test_AE')
              
                plt.scatter([regs[folder][reg][i] for i in range(len(regs[folder][reg])) if i != idx], 
                    [recons[folder][recon][i] for i in range(len(recons[folder][recon])) if i != idx])
                plt.scatter([test_regs[folder][reg][i] for i in range(len(test_regs[folder][reg])) if i != idx], 
                    [test_recons[folder][recon][i] for i in range(len(test_recons[folder][recon])) if i != idx])

                plt.legend()
                for i in sorted(range(len(param_inds[folder])), key=lambda k: float(param_inds[folder][k])):
                    if i % 2 == 0:
                        plt.annotate(str(round(float(param_inds[folder][i]),1)), xy=(regs[folder][reg][i]-offset, recons[folder][recon][i]-offset), size = 'large')
                

                plt.savefig(os.path.join(folder, str(*recon.split('loss')[:-1])+'_'+str(*reg.split('loss')[:-1])+'.pdf'))#, bbox_inches='tight')
                plt.close()

            for lagr in lagrs[folder].keys():
                plt.figure()
                plt.scatter(regs[folder][reg], lagrs[folder][lagr])
                #plt.scatter(test_regs[folder][reg], lagrs[])
                for i in range(len(regs[folder][reg])):
                    plt.annotate(str(round(float(param_inds[folder][i]),1)), xy=(regs[folder][reg][i]+offset, lagrs[folder][lagr][i]+offset))
                plt.savefig(os.path.join(folder, lagr+'_'+str(*reg.split('loss')[:-1])+'.pdf'), bbox_inches='tight')
                plt.close()
            #except Exception as e:
            #    print("FAILED: reg ", reg, " recon ", recon)
            #    print(e)
        #legend = param values
    for folder in folders:
        for reg in regs[folder].keys():
            #print('reg: ', reg, end = '')
            if not ('mi_' in reg or 'vae' in reg or 'ido' in reg):
                continue
            #print("plotting ", reg)
            for recon in recons[folder].keys():
                if recon_loss in recon and 'test' not in recon:
                    if folder == folders[0]:
                        plt.figure(figsize=(15,15))
                        plt.title(str("Train/Test RD"))
                        plt.xlabel(reg)
                        plt.ylabel(recon)
                    
                    if 'beta' in fn or beta:
                        val, idx = min((val, idx) for (idx, val) in enumerate(param_inds[folder]))
                    else: # constrainted optimization, maximal regularizer
                        val, idx = max((val, idx) for (idx, val) in enumerate(param_inds[folder]))

                    plt.scatter([regs[folder][reg][i] for i in range(len(regs[folder][reg])) if i != idx], 
                        [recons[folder][recon][i] for i in range(len(recons[folder][recon])) if i != idx], label = folder.split('/')[1].split('_')[0]+'_train')
                    plt.scatter([test_regs[folder][reg][i] for i in range(len(test_regs[folder][reg])) if i != idx], 
                        [test_recons[folder][recon][i] for i in range(len(test_recons[folder][recon])) if i != idx], label = folder.split('/')[1].split('_')[0]+'_test')
                    for i in sorted(range(len(param_inds[folder])), key=lambda k: float(param_inds[folder][k])):
                        if i % 2 == 0:
                            plt.annotate(str(round(float(param_inds[folder][i]),1)), xy=(regs[folder][reg][i]-offset, recons[folder][recon][i]-offset), size = 'large')
                    

                    plt.axhline(y= recons[folder][recon][idx], color = (np.random.rand(), np.random.rand(), np.random.rand()), linestyle='--', label = 'AE_'+folder.split('/')[1].split('_')[0]+'train')
                    plt.axhline(y= test_recons[folder][recon][idx],color = (np.random.rand(), np.random.rand(), np.random.rand()), linestyle='--', label= 'AE_'+folder.split('/')[1].split('_')[0]+'test')


    plt.legend()
                #for i in range(len(regs[folder][reg])):
                #    plt.annotate(str(round(float(param_inds[folder][i]),1)), xy=(regs[folder][reg][i]-offset, recons[folder][recon][i]-offset), size = 'large')
    name_str = ("comparison_"+str(round(np.random.rand(), 2)).split('.')[0] if name is None else name+"_")+"rd"
    plt.savefig("results/"+name_str+".pdf")
    plt.close()

def plot_loss(hist, keys = ['loss', 'val_loss'], prefix=""):
    # print "==> plotting loss function"
    plt.clf()
    print(hist.keys())
    for k in keys:
        if k not in hist:
            continue 
        if isinstance(hist, dict):
            plt.plot(hist[k], label=k)
        # else:
        #     if k == 'loss':
        #         try:
        #             x = [hist.history[k][i][0][0] for i in range(len(hist.history[k]))]
        #         except:
        #             print(hist.history[k])
        #     else:
        #     x = hist.history[k]
        #     plt.plot(x, label=k) 
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

def graph_history(values, label = ""):
    f = plt.figure()
    plt.plot(list(range(len(values))), values, label = label.split['/'][-1])
    plt.title(label.split['/'][-1]+" over Training")
    plt.ylabel(label)
    plt.xlabel(epoch)
    plt.savefig(label+'_over_time.pdf')


def write_runs(recons, regs, lagrs = None, indices = None):
    for folder in recons.keys():
        for loss in recons[folder].keys():
            pass
    if indices is not None:
        for i in sorted(range(len(indices[folder])), key=lambda k: indices[folder][k]): #range(len(param_inds[folder])):
                # headers
                if csv_str=='':
                    #print("len params ", len(param_inds[folder]))
                    csv_str = "{} \n".format(fn)
                    csv_str += 'Param \t \t'
                    for k in regs[folder].keys():
                        csv_str += "{} \t \t".format(k.split('_')[:-2]) #k.split('loss')[:-2]
                    for k in recons[folder].keys():
                        csv_str += "{} \t \t".format(k.split('_')[:-2])
                        csv_str += "test_{} \t \t".format(k.split('_')[:-2])
                    for k in lagrs.keys():
                        csv_str += "{} \t \t".format(k.split('_')[:-2])
                        #csv_str += "test_{} \t".format(k)
                csv_str += "\n"
                csv_str += '{} \t \t'.format(param_inds[folder][i])
                # columns
                for k in regs[folder].keys():
                    csv_str += "{} \t \t".format(round(regs[folder][k][i],2))
                for k in recons[folder].keys():
                    csv_str += "{} \t \t".format(round(recons[folder][k][i],2))
                    csv_str += "{} \t \t".format(round(test_recons[folder][k][i]),2)
                for k in lagrs.keys():
                    csv_str += "{} \t \t".format(round(lagrs[k][i]),3)

                csv_str += "\n"


def encoder(model, x = None, layer = None):#, echo_batch = None):
    for i in model.layers:
        print(i.name)
        if layer is None or layer == 'z_act':
            if 'z_act' in i.name or 'echo' in i.name:
                final_latent = i.name
                break
        else:
            if layer in i.name:
                final_latent = i.name
    #if echo_batch is None:
    get_z = K.function([model.layers[0].get_input_at(0)], [
                    model.get_layer(final_latent).get_output_at(0)])
    #else:
    #    get_z = K.function([self.model.layers[0].get_input_at(0)], [
    #                    self.model.get_layer(final_latent).get_output_at(0)])
    if x is not None:
        print("GET Z LIST ", len(get_z([x])), get_z([x])[0].shape) 
    return get_z if x is None else get_z([x])[0]

def decoder(model, z = None):
    for i in model.layers:
        if 'z_act' in i.name or 'echo' in i.name:
            final_latent = i.name
            break

    z_inp = Input(shape = (encoder_dims[-1],))
    z_out = z_inp
    call_ = False
    for layr in model.layers:
        # only call decoder layer_list
        if call_ and not isinstance(layr, keras.layers.core.Lambda) and not isinstance(layr, IAF) and not isinstance(layr, MADE_network):#and not ('vae' in layr.name or 'noise' in layr.name or 'info_dropout' in layr.name):
            z_out = layr(z_out)
        if layr.name == final_latent:
            call_ = True
        # doesn't work with new naming convention
        #if layr.name == 'decoder' or layr.name == 'ci_decoder':
        #    call_ = False
    generator = keras.models.Model(input = [z_inp], output = [z_out])
    return generator if z is None else generator.predict(z) 

def sample_echo_reconstructions(model, dataset, n = 5, num_samples = 5, echo_batch = None, echo_dmax = 50, prefix = None):
    if prefix is None:
        prefix = 'results/'+dataset.name+'_echo'+str(echo_dmax)+str(n)
    np.random.shuffle(dataset.x_test)
    if echo_batch is not None:
        samples = dataset.x_test[:n]
    else:
        samples = dataset.x_test[:echo_batch]
        #samples = np.random.shuffle(dataset.x_test)[:echo_batch]
    mean_function = model.encoder(samples, layer = 'z_mean')
    echo_function = model.encoder(samples, layer = 'echo')
    
    generator = decoder(model)
    mean_act = mean_function([samples])[0]
    for i in range(num_samples):
        echo_output = echo_function([mean_act])[0]
        recon = generator(z)
        vis_reconstruction(recon, samples, prefix = prefix+'_'+str(n)+'_'+str(i))


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
    
    print("Z ACT SHAPE PLOT TRAVERSALS ", z_act.shape)
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
        #plt.imshow(figure)
        plt.axis('off')
        plt.savefig('{}_latent_traversals_{}.png'.format(prefix, str(data_pt)), bbox_inches='tight')
        #plt.close('all')




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


def vis_reconstruction(inp, data, model = None, prefix='', noise=None, n=5, batch = 100, num_losses=1):
    # print "==> visualizing reconstructions, prefix = {}".format(prefix)
    if isinstance(data, Dataset):
        digit_dims = data.digit_dims 
    else:
        dim_sqrt = int(np.sqrt(data.shape[-1]))
        if (dim_sqrt +.5)**2 == data.shape[-1]: # hacky int sqrt check
            digit_dims = [dim_sqrt, dim_sqrt]
        else:
            raise ValueError("Specify dim1, dim2 in a Dataset object and feed as argument.")

    figure = np.ones((digit_dims[0] * 3, (digit_dims[1]+1) * n))

    # print 'DATA SHAPE.... ', data.shape
    try:
        data_dim = data.shape[1]
    except:
        data_dim = data.dim
    # if merged:
    #    dummy = Model(input = model.input, output = model.output[:-1, :data_dim])
    #    xbars = dummy.predict(data)
    if model is None:
        xbars = inp
    else:
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
    

    
def manifold(activations, generator, per_dim = 50, dims = None, location = 'results/'):
    # UNTESTED
    lim_x = [np.percentile(activations[:,0], 0), np.percentile(activations[:,0], 100)]
    lim_y = [np.percentile(activations[:,1], 0), np.percentile(activations[:,1], 100)]
    grid_x = np.linspace(lim_x[0], lim_x[1], per_dim)
    grid_y = np.linspace(lim_y[0], lim_y[1], per_dim)
    
    figure = np.zeros((dims[0] * per_dim, dims[1] * per_dim))

    if dims is None:
        dim_sqrt = int(np.sqrt(generator.predict(activations[0,:]).shape[-1]))
        if dim_sqrt**2 == data.shape[-1] or (dim_sqrt +.5)**2 == data.shape[-1]: # hacky int check
            dims = [dim_sqrt, dim_sqrt]

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape((dims[0], dims[1]))
            figure[i * dims[0]: (i + 1) * dims[0],
                   j * dims[1]: (j + 1) * dims[1]] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig(location+'_2d_latent.pdf')
    plt.close()

def two_d_labeled(x, y, encoder, batch = 1000, prefix = ''):
    # UNTESTED
    indices = np.random.choice(x.shape[0], batch, replace=False)
    print('indices ', indices.shape)

    if isinstance(encoder, KerasModel):
        z_act = encoder.predict(x[indices, :])
    else:
        z_act = encoder([x[indices, :]])[0]
    

    if z_act.shape[1] > 2:
        z_act = self.pca_decompose(z_act)
    z1 = z_act[:, 0]
    z2 = z_act[:, 1]
    
    fig, ax = plt.subplots()
    print("axis ", ax)
    scattr = ax.scatter(z1, z2, s=25, c = y[indices].astype(int), cmap = 'Greys_r')
    plt.colorbar(scattr, spacing='proportional')
    plt.grid()
    plt.savefig(str(prefix + '_2d_by_label.pdf'))
