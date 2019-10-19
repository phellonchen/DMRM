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
    parser.add_argument('--input_img_h5', default='/home/user/chenfeilong/dataset/visdial0.9/data/vdl_img_vgg.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_ques_h5', default='/home/user/chenfeilong/aaai/tdAtten3-2-1/data/visdial_data.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_json', default='/home/user/chenfeilong/aaai/tdAtten3-2-1/data/visdial_params.json',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--model_path', default='/home/user/chenfeilong/aaai/tdAtten3-2-1-v0.9/saved_models/2019-8-28-14-53/model_epoch_17.pth',
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
    #model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model_path)
    # print(model.state_dict().keys())
    # print(checkpoint['model'].keys())
    model_dict = model.state_dict()
    keys = []
    for k, v in checkpoint['model'].items():
        keys.append(k)
    i = 0
    for k, v in model_dict.items():
        #if v.size() == checkpoint['model'][keys[i]].size():
        # print(k, ',', keys[i])
        model_dict[k] = checkpoint['model'][keys[i]]
        i = i + 1
    model.load_state_dict(model_dict)
    #model.load_state_dict(checkpoint['model'])
    model.eval()
    print('Evaluating ... ')
    start_time = time.time()
    rank_all = evaluate(model, eval_loader, args, True)
    R1 = np.sum(np.array(rank_all) == 1) / float(len(rank_all))
    R5 = np.sum(np.array(rank_all) <= 5) / float(len(rank_all))
    R10 = np.sum(np.array(rank_all) <= 10) / float(len(rank_all))
    ave = np.sum(np.array(rank_all)) / float(len(rank_all))
    mrr = np.sum(1 / (np.array(rank_all, dtype='float'))) / float(len(rank_all))
    #save_path = checkpoint['args'].save_path
    #logger = utils.Logger(os.path.join(save_path, 'eval-log.txt'))
    #logger.write('mrr: %f R1: %f R5 %f R10 %f Mean %f time: %.2f' % (mrr, R1, R5, R10, ave, time.time() - start_time))
    print('mrr: %f R1: %f R5 %f R10 %f Mean %f time: %.2f' % (mrr, R1, R5, R10, ave, time.time() - start_time))
