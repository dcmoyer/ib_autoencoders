#!/bin/bash
cd /home/rcf-proj/gv/brekelma/autoencoders
python test.py --filename 'results/vae_1250_400_binary_mnist/vae_1250_400_0.05' --config '{"layers": [{"layer": 0, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 1, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 2, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 3, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 4, "layer_kwargs": {"padding": "valid", "strides": 1, "kernel_size": 7}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 5, "type": "additive", "k": 1, "encoder": true}, {"layer": 0, "layer_kwargs": {"padding": "valid", "strides": 1, "kernel_size": 7}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 1, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 2, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 3, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 4, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 5, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 4}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 6, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "keras.activations.sigmoid", "encoder": false}], "output_activation": "sigmoid", "optimizer": "Adam", "recon": "bce", "optimizer_params": {"beta_1": 0.5}, "losses": [{"output": -1, "layer": -1, "type": "vae", "weight": 0.05, "encoder": true}], "initializer": "glorot_uniform", "batch": 100, "decoder_dims": [64, 64, 64, 32, 32, 32, 1], "input_shape": [28, 28, 1], "epochs": 400, "beta": 1.0, "lr": "alemi_400", "anneal_schedule": null, "latent_dims": [32, 32, 64, 64, 256, 64], "activation": {"decoder": "softplus", "encoder": "softplus"}, "anneal_function": null}' --verbose 1 --dataset binary_mnist --per_label 1250
