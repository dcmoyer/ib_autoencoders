#!/bin/bash                                                                                                                                                       
#SBATCH --ntasks=8                                                                                                                                                
#SBATCH --time=00:59:00                                                                                                                                           
#SBATCH --gres=gpu:1                                                                                                                                              

cd /home/rcf-proj/gv/brekelma/autoencoders

python3 test.py --echo_init $ECHO --noise 'multiplicative'
