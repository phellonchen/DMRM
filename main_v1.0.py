import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# from dataset import Dictionary, VQAFeatureDataset
import base_model as base_model
from train import train
import utils
import misc.dataLoader_v1 as dl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='baseline0_newatt2')
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')


    parser.add_argument('--input_img_h5', default='data/features_faster_rcnn_x101_train.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_img_val_h5', default='data/features_faster_rcnn_x101_val.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_ques_h5', default='data/visdial_data_v1.0.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_json', default='data/visdial_params_v1.0.json',
                        help='path to dataset, now hdf5 file')

    parser.add_argument('--img_feat_size', type=int, default=512, help='input batch size')
    parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
    parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
    parser.add_argument('--neg_batch_sample', type=int, default=30,
                        help='folder to output images and model checkpoints')
    parser.add_argument('--num_val', default=1000, help='number of image split out as validation set.')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
    args = parser.parse_args()

    args.input_encoding_size = args.ninp
    args.rnn_size = args.nhid
    args.num_layers = args.nlayers
    args.drop_prob_lm = args.dropout
    args.fc_feat_size = args.img_feat_size
    args.att_feat_size = args.img_feat_size
    args.att_hid_size = args.img_feat_size
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    batch_size = args.batch_size

    train_dset = dl.train(input_img_h5=args.input_img_h5, input_ques_h5=args.input_ques_h5,
                       input_json=args.input_json, negative_sample=args.negative_sample,
                       num_val=args.num_val, data_split='train')

    eval_dset = dl.validate(input_img_h5=args.input_img_val_h5, input_ques_h5=args.input_ques_h5,
                              input_json=args.input_json, negative_sample=args.negative_sample,
                              num_val=args.num_val, data_split='val')

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=int(args.workers))

    eval_loader = torch.utils.data.DataLoader(eval_dset, batch_size=5,
                                                 shuffle=False, num_workers=int(args.workers))

    args.vocab_size = train_dset.vocab_size
    args.ques_length = train_dset.ques_length
    args.ans_length = train_dset.ans_length + 1
    args.his_length = train_dset.ques_length + train_dset.ans_length
    args.seq_length = args.ans_length
    constructor = 'build_%s' % args.model
    vocab_size = train_dset.vocab_size
    model = getattr(base_model, constructor)(args, args.nhid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d_v1.0.npy')

    model = nn.DataParallel(model).cuda()
    model = train(model, train_loader, eval_loader, args)


    # dis_model, model = train_D(model, train_loader, args)


    # train_RL(model,  train_loader, eval_loader, args)
