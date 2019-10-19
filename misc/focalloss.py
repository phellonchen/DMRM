import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):

        target = target.view(-1,1)

        # logpt = F.log_softmax(input, 1)
        logpt = input
        mask = target.data.gt(0)
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)
        logpt = logpt.gather(1,target)
        logpt = torch.masked_select(logpt, mask)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class nPairLoss(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong, fake=None, fake_diff_mask=None):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)

        wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)),1) \
                + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)),1)

        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = right.norm() + feat.norm() + wrong.norm() + batch_wrong.norm()

        if fake:
            fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
            fake_score = torch.masked_select(torch.exp(fake_dis - right_dis), fake_diff_mask)

            margin_score = F.relu(torch.log(fake_score + 1) - self.margin)
            loss_fake = torch.sum(margin_score)
            loss_dis += loss_fake
            loss_norm += fake.norm()

        loss = (loss_dis + 0.1 * loss_norm) / batch_size
        if fake:
            return loss, loss_fake.data[0] / batch_size
        else:
            return loss

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

class G_loss(nn.Module):
    """
    Generator loss:
    minimize right feature and fake feature L2 norm.
    maximinze the fake feature and wrong feature.
    """
    def __init__(self, ninp):
        super(G_loss, self).__init__()
        self.ninp = ninp

    def forward(self, feat, right, fake):

        batch_size = feat.size(0)
        fake = fake.view(batch_size, -1, self.ninp)
        num_fake = fake.size(1)
        feat = feat.view(batch_size, 1, self.ninp).repeat(1, num_fake, 1)
        right = right.view(batch_size, 1, self.ninp).repeat(1, num_fake, 1)

        feat = feat.view(-1, self.ninp, 1)
        #wrong_dis = torch.bmm(wrong, feat)
        #batch_wrong_dis = torch.bmm(batch_wrong, feat)
        fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)

        fake_score = torch.exp(right_dis - fake_dis)
        loss_fake = torch.sum(torch.log(fake_score + 1))

        loss_norm = feat.norm() + fake.norm() + right.norm()
        loss = (loss_fake  + 0.1 * loss_norm ) / batch_size / num_fake

        return loss, loss_fake.data[0]/batch_size