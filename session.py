import numpy as np
import json
import itertools
from utils import make_dir
import time
import os
import copy
import model
import subprocess
import analysis
#from tensorflow import reset_default_graph

def nested_dict(d, keys, value):
    for k in keys[:-1]:
        if isinstance(d, dict):
            d= d.setdefault(k, {})
        else:
            d = d[int(k)]
    d[keys[-1]]= value   
    

class Session(object):
    # LOOP OVER CONFIG FILES ?  or LOOP within CONFIG FILES
    def __init__(self, name = None, config = None, dataset = None, parameters = None):
        self.model_args = None

        if name is None:
            t = time.time()
            name = str('session'+str(round(t,3)-round(t,0)))

        self.name = name
        #self.configs = [self.load_config(config)]
        
        self.config = config if isinstance(config, dict) else self.load_config(config)

        self.parameters = parameters
        # HANDLE MULTIPLE LISTS : change how config initialized, change dict key to tuple
        if parameters is not None:
            self.configs = {}
            for key in parameters.keys():
                ksplit = key.split('.')
                for i in range(len(parameters[key])):
                    self.configs[parameters[key][i]] = self.load_config(config)
                    # keep tuple of param values as dict key
                    # call nested_dict for each of params
                    nested_dict(self.configs[parameters[key][i]], ksplit, parameters[key][i])

        self.run_configs(dataset)

    def run_configs(self, dataset, time ="00:59:59"):# rd_curves = True):
        histories = []
        test_results = []
        indices = []


        # mkdir
        folder = os.path.join('results', str(self.name)+'_'+str(dataset.name))
        #folder = os.path.join('results', str(self.name)+'_'+str(dataset.name)+'_'+self.config.split('.json')[0])
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(os.path.join(folder, "exec"))
            os.mkdir(os.path.join(folder, "exec"))

        exfilename = os.path.join(folder, "base_config.txt")
        with open(exfilename, 'w+') as exfile:
            #exfile.write("#!/usr/bin/env python\n\n")
            #for imp in set(imports):
            #    exfile.write("import %s\n" % imp)
            for st in self.config.keys():
                exfile.write(st+" : "+str(self.config[st])+"\n")



        count = 0
        for param, config in self.configs.items():
            print()
            print("PARAM VALUE ", param)
            
            
            config_str = json.dumps(config).replace("'", '"')
            #sess = tf.Session()
            #K.set_session(sess)
            print("config ", config_str)
            print(type(config_str))
            print()
            # write python file
            statements = [
                "#!/bin/bash",
                "cd /home/rcf-proj/gv/brekelma/autoencoders",
                "python3 test.py --filename \'{fn}\' --config \'{conf}\' ".format(
                    name = self.name, param = param, time = time, conf = config_str, fn = os.path.join(folder, str(self.name+'_'+str(param))))
            ]

            # Write executable
            exfilename = os.path.join(folder, "exec", "dummy_"+str(count)+".sh")
            with open(exfilename, 'w+') as exfile:
                #exfile.write("#!/usr/bin/env python\n\n")
                #for imp in set(imports):
                #    exfile.write("import %s\n" % imp)
                for k in self.parameters.keys():
                    exfile.write(k+" : "+str(self.parameters[k])+"\n")
                for st in statements:
                    exfile.write(st+"\n")
            os.chmod(exfilename, 0o755)


            # SBATCH + read pickle?
            cmd = 'sbatch --job-name {name}_{param} --time {time}\
            --gres=gpu:1 --ntasks=8 --wrap="./{folder}/exec/dummy_{ct}.sh"'.format(name = self.name, param = param, time = time, folder = folder, ct = str(count))
            #\" python3 test.py -filename \'{fn}\' -config \'{conf}\'  \"'.format(
             #       name = self.name, param = param, time = time, conf = config_str, fn = os.path.join(folder, str(self.name+'_'+str(param))))

            # del python file
            count = count + 1
            print("Command ", cmd)
            #!/bin/bash                                                                                                                                                       
            #SBATCH --ntasks=8                                                                                                                                                
            #SBATCH --time=00:59:00                                                                                                                                           
            #SBATCH --gres=gpu:1 
            subprocess.call([cmd], shell=True)
            #os.remove("dummy.sh")


            # m = model.NoiseModel(dataset, config = config, filename = str(self.name+'_'+str(param)))
            # m.fit(dataset.x_train)
            # histories.append(m.hist)
            # test_results.append(m.test_results)
            # indices.append(param)
            # reset_default_graph()

        #if rd_curves:
       #     analysis.rd_curve(folder)
                #hist = histories, test = test_results, legend = indices)


    def load_config(self, config):
        # save all files from session in one folder
        
        #self.folder = config.split('.')[:-1]
        #make_dir(self.folder)

        with open(os.path.join(os.getcwd(), config)) as conf:
            config_dict = json.load(conf)

        #self.model_args = self._parse_config(config_dict)
        #print(model_args)

        return config_dict


    def run_from_config(self, config):
        self.load_config(config)    
        self.run_all()


    def run_all(self):
        if self.model_args is None:
            raise ValueError("Please enter model args")
        else:
            pass
            #for job in self.model_args:
            #   arg_dump = json.dumps(self.model_args)
            # run_dummy script with args_dict as argument (inside, have json.loads(argv[1]))
            # bash run_model.py arg_dump .... but how to get new script?
            # yield args_dict


    def _parse_config(self, config):
        # for each dictionary item which is a list
        product_list = []
        key_list = []
        for arg_key, arg_value in config.items():
            # list of lists should be appended as such
            if isinstance(arg_value, list) and isinstance(arg_value[0], list):
                #EXCEPTIONS? all taken care of by above???
                # if arg_key == 'latent_dims' and not isinstance(arg_value[0], list):
                #   # latent_dims is a list, so don't want to take product over individual entries
                #   product_list.append([arg_value])
                # elif arg_key == 'anneal_sched':
                #   pass
                # else:
                product_list.append(arg_value)  
            else:
                product_list.append([arg_value])
            key_list.append(arg_key)
            arg_lists = itertools.product(*product_list)
            for model_args in range(len(arg_lists)):
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
