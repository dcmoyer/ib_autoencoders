import os


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config):
        # save all files from session in one folder
        # folder = split(config, '.')[:-1]
        # make_dir(self.folder)

        with open(os.path.join(os.getcwd(), config)) as conf:
            config_dict = json.load(conf)

        model_args = parse_config(config_dict)
        print(model_args)
        return model_args

def parse_config(config):
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
            for model_args in range(len(argument_lists)):
                arg_dicts.append(dict(zip(key_list, model_args)))
        
        print(arg_dicts[0])
        print(arg_dicts[1])
        return arg_dicts # list of argument dictionaries