#!/bin/bash

FOLDER_NAME="$1"
if [ "$1" = "" ]
then
    echo "No folder name given"
    exit
else
    mkdir "$FOLDER_NAME"
fi


module load python_gpu/3.7.1

bsub -oo "$FOLDER_NAME"/all_run_1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_sTriplet_Visual.py --leonhard 1 --n_epochs 50
bsub -oo "$FOLDER_NAME"/all_run_2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50
bsub -oo "$FOLDER_NAME"/all_run_3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50
bsub -oo "$FOLDER_NAME"/all_run_4 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50

#bsub -oo "$FOLDER_NAME"/words_run_1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 -only_words
#bsub -oo "$FOLDER_NAME"/words_run_2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 -only_words
#bsub -oo "$FOLDER_NAME"/words_run_3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 -only_words
#bsub -oo "$FOLDER_NAME"/words_run_4 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 -only_words



bsub -oo "$FOLDER_NAME"/surj_0 -n 4 -R    "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 0 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_3e-4 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 3e-4 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_6e-4 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 6e-4 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_1e-3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas  100 1e-3 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_3e-3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 3e-3 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_6e-3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas  100 6e-3 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_1e-2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 1e-2 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_3e-2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 3e-2 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_6e-2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 6e-2  1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_1e-1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 1e-1 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_3e-1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 1e-1 1e-3 --learning_rate 1e-4 --margin 50 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_6e-1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 6e-1 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9

bsub -oo "$FOLDER_NAME"/surj_1 -n 4 -R    "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python ZSL_All_Triplet_Visual.py --leonhard 1 --n_epochs 50 --alphas 100 1 1e-2 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9






echo -e 'all jobs successfully submitted.'