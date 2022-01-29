#!/bin/bash

python train.py \
            --dataFile "../data/FewAsp(multi)" \
            --fileVocab "/bert_base_uncased/bert-base-uncased-vocab.txt" \
            --fileModelConfig "/bert_base_uncased/bert-base-uncased-config.json" \
            --fileModel "/bert_base_uncased/bert-base-uncased-pytorch_model.bin" \
            --fileModelSave "./model/model_n5k5" \
            --numDevice 0 \
            --learning_rate 1e-5 \
            --epochs 20 \
            --numNWay 5 \
            --numKShot 5 \
            --numQShot 5 \
            --count 5 \
            --episodeTrain 100 \
            --episodeTest 600 \
            --numFreeze 6 \
            --warmup_steps 100 \
            --dropout_rate 0.1 \
            --weight_decay 0.2 \
            --MFB_DROPOUT_RATIO 0.1 \
            --MFB_OUT_DIM 1 \
            --MFB_FACTOR_NUM 100 \
            --temperature 0.1
