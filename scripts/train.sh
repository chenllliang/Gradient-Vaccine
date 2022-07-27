#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


root_dir=../data


lang_list=$root_dir/lang_list.txt  # <path to a file which contains a list of languages separted by new lines>
lang_pairs=ro_RO-en_XX,fr_XX-en_XX  # translation directions

BIN=$root_dir/multilingual_bin
SAVE=$root_dir/checkpoints
tensorboard_dir=$root_dir/tensorboards

sp=1
# sampling temperature

export CUDA_VISIBLE_DEVICES=0
TASK=gradient_vaccine_alpha_0.5_ema0.01
# alpha is the predefine cos similarity threshold, ema is the beta value of ema
mkdir grads_cos/${TASK}

fairseq-train "${BIN}" \
  --save-dir ${SAVE}/${TASK} \
  --arch transformer \
  --share-all-embeddings \
  --task translation_multi_simple_epoch_vaccine \
  --sampling-method "temperature" \
  --sampling-temperature ${sp} \
  --encoder-langtok "tgt" \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --best-checkpoint-metric ppl \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-04 --warmup-updates 4000 --max-update 200000 \
  --max-tokens 20000 --update-freq 1 \
  --left-pad-source False \
  --skip-invalid-size-inputs-valid-test \
  --save-interval 1 --save-interval-updates 50000 \
  --no-epoch-checkpoints \
  --seed 888 --log-format json --log-interval 100 \
  --tensorboard-logdir ${tensorboard_dir}/0720/${TASK} \
  --grad-dir grads_cos/${TASK} \
  --grad-vaccine \
  --grad-vaccine-alpha 0.5 \
  --grad-vaccine-ema-beta 0.01 \
