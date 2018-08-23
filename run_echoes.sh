#!/bin/bash 
NOISE = "multiplicative"      
export NOISE                                                                                                                                            
for ECHO in -.0804 -.15212 -.20397 -.25182 -.28543 -.31527 -.34873 -.37777 -.40088 -.42320 -.68384 -.72252
do
        echo $ECHO
        export ECHO
        sbatch echo_py.sh
        sleep 1
done