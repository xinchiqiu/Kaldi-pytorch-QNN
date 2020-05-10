set samples 1000
set xrange [0.000000:1.000000]
set autoscale y
set size 0.78, 1.0
set nogrid
set ylabel 'Counts'
set xlabel 'Confidence Measure'
set title  'Confidence scores for /home/xinchi/Kaldi-pytorch-QNN/Kaldi_baseline/exp/QLSTM_mfcc_singlecopy_3500/decode_TIMIT_test_out_dnn2/score_8/ctm_39phn.filt'
plot '/home/xinchi/Kaldi-pytorch-QNN/Kaldi_baseline/exp/QLSTM_mfcc_singlecopy_3500/decode_TIMIT_test_out_dnn2/score_8/ctm_39phn.filt.hist.dat' using 1:2 '%f%f' title 'All Conf.' with lines, \
     '/home/xinchi/Kaldi-pytorch-QNN/Kaldi_baseline/exp/QLSTM_mfcc_singlecopy_3500/decode_TIMIT_test_out_dnn2/score_8/ctm_39phn.filt.hist.dat' using 1:2 '%f%*s%f' title 'Correct Conf.' with lines, \
     '/home/xinchi/Kaldi-pytorch-QNN/Kaldi_baseline/exp/QLSTM_mfcc_singlecopy_3500/decode_TIMIT_test_out_dnn2/score_8/ctm_39phn.filt.hist.dat' using 1:2 '%f%*s%*s%f' title 'Incorrect Conf.' with lines
set size 1.0, 1.0
