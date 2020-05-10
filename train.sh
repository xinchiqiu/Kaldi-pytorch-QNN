#!/bin/bash
cd egs
rm -r timit
cp /SAVE/timit.tar .
tar xvf timit.tar

cd timit/s5
./utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph
cd ..

cp -r /SAVE/Kaldi-pytorch-QNN .
cd Kaldi-pytorch-QNN/Kaldi_baseline

python run_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_cudnn_fbank_beam.cfg
cd exp
cp -r LSTM_cudnn_beam_512_true_6000 /SAVE/results

