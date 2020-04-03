#!/bin/bash
#SBATCH --job-name=lstm


export CUDA_VISIBLE_DEVICES=$(scontrol show job=$SLURM_JOBID --details | grep GRES_IDX | awk -F "IDX:" '{print $2}' | awk -F ")" '{print $1}' | sed "s/-/,/g")

docker run --rm \
	-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
        -v /datasets/TIMIT:/TIMIT \
        -v /home/xinchi:/SAVE/ \
        --gpus all titouan/kaldi_with_pytorch:latest bash -c \
        /SAVE/train_beam_lstm.sh


