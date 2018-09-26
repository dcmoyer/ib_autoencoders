#!/bin/bash
cd /home/rcf-proj/gv/brekelma/autoencoders
python test.py --filename 'results/made_1250_400_binary_mnist/made_1250_400_0.2' --config '{"layers": [{"layer": 0, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 1, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 2, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 3, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": 4, "layer_kwargs": {"padding": "valid", "strides": 1, "kernel_size": 7}, "type": "Conv2D", "activation": "custom_functions.activations.gated_linear", "encoder": true}, {"layer": -1, "k": 1, "density_estimator": {"layer_kwargs": {"layers": [640, 640, 640, 640], "activation": "custom_functions.activations.gated_linear", "steps": 4, "mean_only": true}, "type": "maf"}, "encoder": true, "layer_kwargs": {"layers": [640, 640, 640, 640], "activation": "custom_functions.activations.gated_linear", "steps": 4, "mean_only": true}, "type": "iaf"}, {"layer": 0, "layer_kwargs": {"padding": "valid", "strides": 1, "kernel_size": 7}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 1, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 2, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 3, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 4, "layer_kwargs": {"padding": "same", "strides": 2, "kernel_size": 5}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 5, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 4}, "type": "Conv2DTranspose", "activation": "custom_functions.activations.gated_linear", "encoder": false}, {"layer": 6, "layer_kwargs": {"padding": "same", "strides": 1, "kernel_size": 5}, "type": "Conv2D", "activation": "keras.activations.sigmoid", "encoder": false}], "output_activation": "sigmoid", "optimizer": "Adam", "recon": "bce", "optimizer_params": {"beta_1": 0.5}, "losses": [{"layer": -1, "layer_kwargs": {"activation": "custom_functions.activations.gated_linear"}, "type": "made_density", "weight": 0.2, "encoder": true}, {"layer": -1, "type": "iaf", "weight": 0.2, "encoder": true}], "batch": 100, "decoder_dims": [64, 64, 64, 32, 32, 32, 1], "initializer": "glorot_uniform", "epochs": 400, "beta": 1.0, "lr": "alemi_400", "anneal_schedule": null, "latent_dims": [32, 32, 64, 64, 256, 64], "input_shape": [28, 28, 1], "anneal_function": null}' --verbose 1 --dataset binary_mnist --per_label 1250
