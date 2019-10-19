import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import base_model as base_model
from train import train
import misc.dataLoader as dl
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='baseline0_newatt2')
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    parser.add_argument('--input_img_h5', default='data/features_faster_rcnn_x101_train.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_imgid', default='data/features_faster_rcnn_x101_train_v0.9_imgid.json',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_ques_h5', default='data/visdial_data.h5',
                        help='path to dataset, now hdf5 file')
    parser.add_argument('--input_json', default='data/visdial_params.json',
                        help='path to dataset, now hdf5 file')

    parser.add_argument('--img_feat_size', type=int, default=2048, help='input batch size')
    parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
    parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
    parser.add_argument('--neg_batch_sample', type=int, default=30,
                        help='folder to output images and model checkpoints')
    parser.add_argument('--num_val', default=1000, help='number of image split out as validation set.')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
    parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    args.seed = random.randint(1, 10000)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    batch_size = args.batch_size

    train_dataset = dl.train(input_img_h5=args.input_img_h5, input_imgid=args.input_imgid, input_ques_h5=args.input_ques_h5,
                             input_json=args.input_json, negative_sample=args.negative_sample,
                             num_val=args.num_val, data_split='train')

    eval_dateset = dl.validate(input_img_h5=args.input_img_h5, input_imgid=args.input_imgid, input_ques_h5=args.input_ques_h5,
                               input_json=args.input_json, negative_sample=args.negative_sample,
                               num_val=args.num_val, data_split='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=int(args.workers))

    eval_loader = torch.utils.data.DataLoader(eval_dateset, batch_size=5,
                                              shuffle=False, num_workers=int(args.workers))

    args.vocab_size = train_dataset.vocab_size
    args.ques_length = train_dataset.ques_length
    args.ans_length = train_dataset.ans_length + 1
    args.his_length = train_dataset.ques_length + train_dataset.ans_length
    args.seq_length = args.ans_length
    constructor = 'build_%s' % args.model
    vocab_size = train_dataset.vocab_size
    model = getattr(base_model, constructor)(args, args.nhid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Training params: ", num_params)
    model = nn.DataParallel(model).cuda()
    model = train(model, train_loader, eval_loader, args)

