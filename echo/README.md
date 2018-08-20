# CorEx with Echo Noise

Usually, we assume that encoded latent factors are independent conditioned on the input. That might be overly 
restrictive. What happens if we relax that assumption? 

This code implements a special case where it is easy to relax the assumption. We can have noise that is correlated
in a manner that resembles correlations in the encoded state. Then the capacity has a simple, 
analytic expression. 

Some sample experiments in the experiments folder: 
```bash
python run_mnist.py  # Run unsupervised autoencoder
python run_supervised.py  # Specify bottleneck on cifar/mnist
python run_celeba.py  # unsupervised autoencoder for celeba
```
