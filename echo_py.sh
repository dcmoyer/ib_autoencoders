#!/bin/bash                                                                                                                                                       
#SBATCH --ntasks=8                                                                                                                                                
#SBATCH --time=00:59:00                                                                                                                                           
#SBATCH --gres=gpu:1                                                                                                                                              

cd /home/rcf-proj/gv/brekelma/autoencoders

python3 test.py --noise 'additive' --echo_init $ECHO
#python3 test.py --constraint $ECHO
