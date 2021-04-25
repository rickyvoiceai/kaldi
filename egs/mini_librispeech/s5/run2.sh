#!/usr/bin/env bash

# Change this location to somewhere where you want to put the data.
data=./corpus/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh

set -euo pipefail

mkdir -p $data

for part in dev-clean-2 train-clean-5; do
  local/download_and_untar.sh $data $data_url $part
done

if [ $stage -le 0 ]; then
  local/download_lm.sh $lm_url $data data/local/lm
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean-2 train-clean-5; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  for part in dev_clean_2 train_clean_5; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short
fi

# train a monophone system
if [ $stage -le 3 ]; then
  # TODO(galv): Is this too many jobs for a smaller dataset?
  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_500short data/lang_nosp exp/mono

  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_clean_5 data/lang_nosp exp/mono exp/mono_ali_train_clean_5
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_clean_5 data/lang_nosp exp/mono_ali_train_clean_5 exp/tri1

  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
    data/train_clean_5 data/lang_nosp exp/tri1 exp/tri1_ali_train_clean_5
fi

# train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_clean_5 data/lang_nosp exp/tri1_ali_train_clean_5 exp/tri2b

  # Align utts using the tri2b model
  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train_clean_5 data/lang_nosp exp/tri2b exp/tri2b_ali_train_clean_5
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_clean_5 data/lang_nosp exp/tri2b_ali_train_clean_5 exp/tri3b
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_clean_5 data/lang_nosp exp/tri3b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train_clean_5 data/lang exp/tri3b exp/tri3b_ali_train_clean_5
fi


if [ $stage -le 8 ]; then
  # Test the tri3b system with the silprobs and pron-probs.

  # decode using the tri3b model
  utils/mkgraph.sh data/lang_test_tgsmall \
                   exp/tri3b exp/tri3b/graph_tgsmall
  for test in dev_clean_2; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
                          exp/tri3b/graph_tgsmall data/$test \
                          exp/tri3b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                       data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test
  done
fi

# Train a chain model
if [ $stage -le 9 ]; then
./local/chain/tuning/run_cnn_tdnn_1a.sh # cnn tdnnf
./local/chain/tuning/run_cnn_tdnn_1c.sh # cnn tdnnf, attention, multiple stream 
./local/chain/tuning/run_cnn_tdnn_1d.sh # cnn tdnn lstm, spectrum augmentation, segment cmvn
./local/chain/tuning/run_cnn_tdnn_1e.sh # cnn tdnn lstm, smaller model
./local/chain/tuning/run_cnn_tdnn_1f.sh # cnn tdnn lstm, spectrum augmentation, online cmvn
./local/chain/tuning/run_cnn_tdnn_1g.sh # cnn tdnnf, attention, multiple stream, spectrum augmentation, online cmvn
./local/chain/tuning/run_cnn_tdnn_1h.sh # cnn tdnnf, attention, multiple stream, spectrum augmentation
./local/chain/tuning/run_cnn_tdnn_1j.sh # cnn tdnnf, attention, multiple stream, spectrum augmentation, bigger model
fi

# grep WER exp/chain/cnn_tdnn1a_sp/decode_tgsmall_dev_clean_2/wer_* | ./utils/best_wer.sh 
# %WER 10.98 [ 2211 / 20138, 213 ins, 298 del, 1700 sub ] exp/chain/cnn_tdnn1a_sp/decode_tgsmall_dev_clean_2/wer_13_0.0
# grep WER exp/chain/cnn_tdnn1a_sp/decode_tglarge_dev_clean_2/wer_* | ./utils/best_wer.sh 
# %WER 7.94 [ 1598 / 20138, 174 ins, 217 del, 1207 sub ] exp/chain/cnn_tdnn1a_sp/decode_tglarge_dev_clean_2/wer_12_1.0
# grep WER exp/chain/cnn_tdnn1a_sp_online/decode_tgsmall_dev_clean_2/wer_* | ./utils/best_wer.sh 
# %WER 10.95 [ 2206 / 20138, 212 ins, 302 del, 1692 sub ] exp/chain/cnn_tdnn1a_sp_online/decode_tgsmall_dev_clean_2/wer_13_0.0
# grep WER exp/chain/cnn_tdnn1a_sp_online/decode_tglarge_dev_clean_2/wer_* | ./utils/best_wer.sh 
# %WER 7.97 [ 1604 / 20138, 175 ins, 209 del, 1220 sub ] exp/chain/cnn_tdnn1a_sp_online/decode_tglarge_dev_clean_2/wer_12_1.0


# %WER 11.03 [ 2221 / 20138, 232 ins, 274 del, 1715 sub ] exp/chain/cnn_tdnn1c_sp/decode_tgsmall_dev_clean_2/wer_12_0.0
# %WER 7.98 [ 1607 / 20138, 177 ins, 211 del, 1219 sub ] exp/chain/cnn_tdnn1c_sp/decode_tglarge_dev_clean_2/wer_11_1.0
# %WER 11.05 [ 2226 / 20138, 234 ins, 277 del, 1715 sub ] exp/chain/cnn_tdnn1c_sp_online/decode_tgsmall_dev_clean_2/wer_12_0.0
# %WER 7.97 [ 1605 / 20138, 179 ins, 209 del, 1217 sub ] exp/chain/cnn_tdnn1c_sp_online/decode_tglarge_dev_clean_2/wer_11_1.0

# %WER 10.79 [ 2173 / 20138, 232 ins, 269 del, 1672 sub ] exp/chain/cnn_tdnn1d_sp/decode_tgsmall_dev_clean_2/wer_13_0.5
# %WER 7.72 [ 1555 / 20138, 222 ins, 150 del, 1183 sub ] exp/chain/cnn_tdnn1d_sp/decode_tglarge_dev_clean_2/wer_14_0.5

# %WER 12.84 [ 2585 / 20138, 298 ins, 303 del, 1984 sub ] exp/chain/cnn_tdnn1e_sp/decode_tgsmall_dev_clean_2/wer_15_0.0
# %WER 9.20 [ 1853 / 20138, 256 ins, 205 del, 1392 sub ] exp/chain/cnn_tdnn1e_sp/decode_tglarge_dev_clean_2/wer_17_0.5
# %WER 12.78 [ 2574 / 20138, 306 ins, 306 del, 1962 sub ] exp/chain/cnn_tdnn1e_sp_online/decode_tgsmall_dev_clean_2/wer_15_0.0
# %WER 9.37 [ 1886 / 20138, 262 ins, 203 del, 1421 sub ] exp/chain/cnn_tdnn1e_sp_online/decode_tglarge_dev_clean_2/wer_17_0.5

# %WER 10.85 [ 2185 / 20138, 263 ins, 254 del, 1668 sub ] exp/chain/cnn_tdnn1f_sp/decode_tgsmall_dev_clean_2/wer_14_0.0
# %WER 7.89 [ 1588 / 20138, 216 ins, 177 del, 1195 sub ] exp/chain/cnn_tdnn1f_sp/decode_tglarge_dev_clean_2/wer_13_1.0
# %WER 10.72 [ 2158 / 20138, 272 ins, 233 del, 1653 sub ] exp/chain/cnn_tdnn1f_sp_online/decode_tgsmall_dev_clean_2/wer_13_0.0
# %WER 7.81 [ 1572 / 20138, 202 ins, 185 del, 1185 sub ] exp/chain/cnn_tdnn1f_sp_online/decode_tglarge_dev_clean_2/wer_14_1.0

# %WER 10.49 [ 2113 / 20138, 233 ins, 218 del, 1662 sub ] exp/chain/cnn_tdnn1g_sp/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 7.29 [ 1469 / 20138, 172 ins, 159 del, 1138 sub ] exp/chain/cnn_tdnn1g_sp/decode_tglarge_dev_clean_2/wer_10_1.0
# %WER 10.51 [ 2116 / 20138, 231 ins, 221 del, 1664 sub ] exp/chain/cnn_tdnn1g_sp_online/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 7.25 [ 1459 / 20138, 152 ins, 177 del, 1130 sub ] exp/chain/cnn_tdnn1g_sp_online/decode_tglarge_dev_clean_2/wer_11_1.0

# %WER 10.32 [ 2078 / 20138, 237 ins, 218 del, 1623 sub ] exp/chain/cnn_tdnn1h_sp/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 7.18 [ 1446 / 20138, 156 ins, 167 del, 1123 sub ] exp/chain/cnn_tdnn1h_sp/decode_tglarge_dev_clean_2/wer_11_1.0
# %WER 10.32 [ 2079 / 20138, 251 ins, 190 del, 1638 sub ] exp/chain/cnn_tdnn1h_sp_online/decode_tgsmall_dev_clean_2/wer_10_0.0
# %WER 7.20 [ 1450 / 20138, 153 ins, 168 del, 1129 sub ] exp/chain/cnn_tdnn1h_sp_online/decode_tglarge_dev_clean_2/wer_11_1.0

# %WER 10.04 [ 2022 / 20138, 219 ins, 229 del, 1574 sub ] exp/chain/cnn_tdnn1i_sp/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 6.93 [ 1396 / 20138, 154 ins, 155 del, 1087 sub ] exp/chain/cnn_tdnn1i_sp/decode_tglarge_dev_clean_2/wer_11_1.0
# %WER 10.02 [ 2018 / 20138, 222 ins, 225 del, 1571 sub ] exp/chain/cnn_tdnn1i_sp_online/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 6.91 [ 1391 / 20138, 151 ins, 151 del, 1089 sub ] exp/chain/cnn_tdnn1i_sp_online/decode_tglarge_dev_clean_2/wer_11_1.0

# %WER 9.81 [ 1975 / 20138, 223 ins, 205 del, 1547 sub ] exp/chain/cnn_tdnn1j_sp/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 6.84 [ 1378 / 20138, 186 ins, 118 del, 1074 sub ] exp/chain/cnn_tdnn1j_sp/decode_tglarge_dev_clean_2/wer_11_0.5
# %WER 9.83 [ 1980 / 20138, 225 ins, 201 del, 1554 sub ] exp/chain/cnn_tdnn1j_sp_online/decode_tgsmall_dev_clean_2/wer_11_0.0
# %WER 6.87 [ 1383 / 20138, 186 ins, 117 del, 1080 sub ] exp/chain/cnn_tdnn1j_sp_online/decode_tglarge_dev_clean_2/wer_11_0.5

