import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import evaluate
import utils
import misc.dataLoader as dl
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='baseline0_newatt2')
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--input_img_h5', default='data/features_faster_rcnn_x101_train.h5.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_ques_h5', default='data/visdial_data_v1.0.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_json', default='data/visdial_params_v1.0.json',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--model_path', default='',
                        help='path to model, now pth file')
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
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate for, default=0.00005')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    batch_size = args.batch_size

    eval_dset = dl.validate(input_img_h5=args.input_img_h5, input_ques_h5=args.input_ques_h5,
                input_json=args.input_json, negative_sample=args.negative_sample,
                num_val=args.num_val, data_split='test')

    eval_loader = torch.utils.data.DataLoader(eval_dset, batch_size=5,
                                                 shuffle=False, num_workers=int(args.workers))

    args.vocab_size = eval_dset.vocab_size
    args.ques_length = eval_dset.ques_length
    args.ans_length = eval_dset.ans_length + 1
    args.his_length = eval_dset.ques_length + eval_dset.ans_length
    args.seq_length = args.ques_length
    constructor = 'build_%s' % args.model
    vocab_size = eval_dset.vocab_size
    model = getattr(base_model, constructor)(args, args.nhid).cuda()
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model_path)

    # model_dict = model.state_dict()
    # keys = []
    # for k, v in checkpoint['model'].items():
    #     keys.append(k)
    # i = 0
    # for k, v in model_dict.items():
    #     #if v.size() == checkpoint['model'][keys[i]].size():
    #     # print(k, ',', keys[i])
    #     model_dict[k] = checkpoint['model'][keys[i]]
    #     i = i + 1
    # model.load_state_dict(model_dict)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('Evaluating ... ')
    start_time = time.time()
    rank_all = evaluate(model, eval_loader, args, True)
    R1 = np.sum(np.array(rank_all) == 1) / float(len(rank_all))
    R5 = np.sum(np.array(rank_all) <= 5) / float(len(rank_all))
    R10 = np.sum(np.array(rank_all) <= 10) / float(len(rank_all))
    ave = np.sum(np.array(rank_all)) / float(len(rank_all))
    mrr = np.sum(1 / (np.array(rank_all, dtype='float'))) / float(len(rank_all))

    print('mrr: %f R1: %f R5 %f R10 %f Mean %f time: %.2f' % (mrr, R1, R5, R10, ave, time.time() - start_time))
