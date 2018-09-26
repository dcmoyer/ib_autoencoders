#!/bin/bash
cd /home/rcf-proj/gv/brekelma/autoencoders
python test.py --filename 'results/echoA_500_400_fmnist/echoA_500_400_0.3' --config '{"layers": [{"layer": 0, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 1, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 2, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 3, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 4, "layer_kwargs": {"padding": "valid", "strides": 1, "kernel_size": 7}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 5, "layer_kwargs": {"d_max": 50, "trainable": true, "init": -2.0, "noise": "additive"}, "type": "echo", "k": 1, "encoder": true}, {"layer": 0, "layer_kwargs": {"padding": "valid", "strides": 1, "kernel_size": 7}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 1, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 2, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 3, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 4, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 5, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 4}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 6, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "keras.activations.sigmoid", "encoder": false}], "output_activation": "sigmoid", "optimizer": "Adam", "recon": "bce", "optimizer_params": {"beta_1": 0.5}, "losses": [{"layer": -1, "type": "echo", "weight": 0.3, "encoder": true}], "initializer": "glorot_uniform", "batch": 100, "decoder_dims": [64, 64, 64, 32, 32, 32, 1], "input_shape": [28, 28, 1], "epochs": 400, "beta": 1.0, "lr": "alemi_400", "anneal_schedule": null, "latent_dims": [32, 32, 64, 64, 256, 64], "activation": {"decoder": "softplus", "encoder": "softplus"}, "anneal_function": null}' --verbose 1 --dataset fmnist --per_label 500
