# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFile',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--fileVocab',
                        type=str,
                        help='path to pretrained model vocab')
        
    parser.add_argument('--fileModelConfig',
                        type=str,
                        help='path to pretrained model config')

    parser.add_argument('--fileModel',
                        type=str,
                        help='path to pretrained model')

    parser.add_argument('--fileModelSave',
                        type=str,
                        help='path to save model')
    

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=20)
    parser.add_argument('--numNWay',
                        type=int,
                        help='number of classes per episode',
                        default=5)
    parser.add_argument('--numKShot',
                        type=int,
                        help='number of instances per class',
                        default=5)

    parser.add_argument('--numQShot',
                        type=int,
                        help='number of querys per class',
                        default=5)
    
    parser.add_argument('--count',
                        type=int,
                        help='number of counts',
                        default=5)
    
    parser.add_argument('--episodeTrain',
                        type=int,
                        help='number of tasks per epoch in training process',
                        default=100)

    parser.add_argument('--episodeTest',
                        type=int,
                        help='number of tasks per epoch in testing process',
                        default=600)

    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=1e-5)

    parser.add_argument('--numFreeze',
                        type=int,
                        help='number of freezed layers in pretrained model',
                        default=6)

    parser.add_argument('--numDevice',
                        type=int,
                        help='id of gpu ',
                        default=0)

    parser.add_argument('--warmup_steps',
                        type=int,
                        help='num of warmup_steps',
                        default=100)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='ratio of decay',
                        default=0.2)

    parser.add_argument('--dropout_rate',
                        type=float,
                        help='ratio of dropout',
                        default=0.1)

    parser.add_argument('--MFB_DROPOUT_RATIO',
                        type=float,
                        help='ratio of MFB dropout',
                        default=0.1)

    parser.add_argument('--MFB_OUT_DIM',
                        type=int,
                        help='dim of mfb outputs',
                        default=1)

    parser.add_argument('--MFB_FACTOR_NUM',
                        type=int,
                        help='num of mfb factor',
                        default=100)                   
    
    parser.add_argument('--temperature',
                        type=float,
                        help='scl temperature',
                        default=0.1)
    
    
    return parser
