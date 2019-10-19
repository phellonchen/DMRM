from __future__ import print_function
import os
import sys
import json
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='../data/visdial_params_v1.0.json',
                        help='path to dataset, now hdf5 file')
    args = parser.parse_args()

    f = json.load(open(args.input_json, 'r'))
    itow = f['itow']
    emb_dim = 300
    glove_file = '../data/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(itow, glove_file)
    np.save('../data/glove6b_init_%dd_v1.0.npy' % emb_dim, weights)
