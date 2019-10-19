import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from collections import OrderedDict
import numpy as np
#from ciderD.ciderD import CiderD
"""
Some utility Functions.
"""

# CiderD_scorer = None
#CiderD_scorer = CiderD(df='corpus')


def repackage_hidden_volatile(h):
    if type(h) == Variable:
        return Variable(h.data, volatile=True)
    else:
        return tuple(repackage_hidden_volatile(v) for v in h)

def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)

def clip_gradient(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data.clamp_(-5, 5)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    if epoch < 20:
        lr = lr * (0.5 ** (epoch // 5))
    if epoch < 30 and epoch >= 20:
        lr = 0.0001
    if epoch >= 30:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def decode_txt(itow, x):
    """Function to show decode the text."""
    out = []
    for b in range(x.size(1)):
        txt = ''
        for t in range(x.size(0)):
            idx = x[t,b]
            if idx == 0 or idx == 3:
                break
            txt += itow[str(int(idx))]
            txt += ' '
        out.append(txt)

    return out

def decode_txt_ques(itow, x):
    """Function to show decode the text."""
    out = []
    for b in range(x.size(1)):
        txt = ''
        for t in range(x.size(0)):
            idx = x[t,b]
            if idx == 3:
                break
            if idx == 0:
                continue
            txt += itow[str(int(idx))]
            if t != (x.size(0) -1):
                txt += ' '
        out.append(txt)

    return out

def l2_norm(input):
    """
    input: feature that need to normalize.
    output: normalziaed feature.
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def sample_batch_neg(answerIdx, negAnswerIdx, sample_idx, num_sample):
    """
    input:
    answerIdx: batch_size
    negAnswerIdx: batch_size x opt.negative_sample

    output:
    sample_idx = batch_size x num_sample
    """

    batch_size = answerIdx.size(0)
    num_neg = negAnswerIdx.size(0) * negAnswerIdx.size(1)
    negAnswerIdx = negAnswerIdx.clone().view(-1)
    for b in range(batch_size):
        gt_idx = answerIdx[b]
        for n in range(num_sample):
            while True:
                rand = int(random.random() * num_neg)
                neg_idx = negAnswerIdx[rand]
                if gt_idx != neg_idx:
                    sample_idx.data[b, n] = rand
                    break


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        #print(input[0])
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output

def RLreward(input, target, reward):
    batch_size = input.size(0)
    seq_len = input.size(1)
    inp = input.permute(1, 0, 2)  # seq_len x batch_size
    target = target.permute(1, 0)  # seq_len x batch_size

    loss = 0
    for i in range(seq_len):
        # TODO: should h be detached from graph (.detach())?
        for j in range(batch_size):
            loss += -inp[i][j][target.data[i][j]] * reward[j]  # log(P(y_t|Y_1:Y_{t-1})) * Q

    return loss / batch_size

def array_to_str(arr):
    out = []
    for i in range(len(arr)):
        out.append(str(arr[i]))
        if arr[i] == 0:
            break
    return out



def init_cider_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)

def get_self_critical_reward(netG, netW, sample_ans_input, ques_hidden, gen_result, ans_input, itows):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    #seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    greedy_res, _ = netG.sample(netW, sample_ans_input, ques_hidden)
    ans_sample_txt = decode_txt(itows, greedy_res.t())
    #print('greedy_ans:  %s' % (ans_sample_txt))
    res1 = OrderedDict()
    res2 = OrderedDict()
    #
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    ans_input = ans_input.cpu().numpy()
    for i in range(batch_size):
        res1[i] = array_to_str(gen_result[i])
    for i in range(batch_size):
        res2[i] = array_to_str(greedy_res[i])
    #
    gts = OrderedDict()
    for i in range(len(ans_input)):
        gts[i] = array_to_str(ans_input[i])
    #
    # # _, scores = Bleu(4).compute_score(gts, res)
    # # scores = np.array(scores[3])
    # res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    # gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}


    from nltk.translate import bleu
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4
    scores = []
    for i in range(len(gen_result)):
        score = bleu(gts[i], res1[i], weights=(0.5, 0.5))
        # if score != 0:
        #     print i , ': ' , score
        scores.append(score)
    for i in range(len(greedy_res)):
        score = bleu(gts[i], res2[i], weights=(0.5, 0.5))
        scores.append(score)
#    scores = bleu(gts, res, smoothing_function=smoothie)

    # _, scores = CiderD_scorer.compute_score(gts, res)
    # print('Cider scores:', _)
    scores = np.array(scores)
    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

class LayerNorm(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta