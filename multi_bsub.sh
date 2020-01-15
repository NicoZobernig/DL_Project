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

bsub -oo "$FOLDER_NAME"/all_run_1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50
bsub -oo "$FOLDER_NAME"/all_run_2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50
bsub -oo "$FOLDER_NAME"/all_run_3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50
bsub -oo "$FOLDER_NAME"/all_run_4 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50

bsub -oo "$FOLDER_NAME"/words_run_1 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50 -only_words
bsub -oo "$FOLDER_NAME"/words_run_2 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50 -only_words
bsub -oo "$FOLDER_NAME"/words_run_3 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50 -only_words
bsub -oo "$FOLDER_NAME"/words_run_4 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080]" python zsl_triplet_test.py --leonhard 1 --n_epochs 50 -only_words


echo -e 'all jobs successfully submitted.'