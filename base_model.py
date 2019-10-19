import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding, QuestionEmbedding2
from classifier import SimpleClassifier
from fc import FCNet
from Decoders.decoder1 import _netG as netG
import torch.nn.functional as F
from torch.autograd import Variable
from misc.utils import LayerNorm
class BaseModel2(nn.Module):
    def __init__(self, w_emb, q_emb, h_emb, v_att, h_att, q_net, v_net, h_net, qih_att, qhi_att, qih_net, qhi_net,
                 decoder, args, qhih_att, qihi_att):
        super(BaseModel2, self).__init__()
        self.ninp = args.ninp
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.h_emb = h_emb
        self.decoder = decoder
        self.img_embed = nn.Linear(args.img_feat_size, 2 * args.nhid)
        self.w1 = nn.Linear(args.nhid*2, args.nhid*2)
        self.w2 = nn.Linear(args.nhid*2, args.nhid*2)
        self.track_1 = v_att
        self.locate_1 = h_att
        self.locate_2 = qih_att
        self.track_2 = qhi_att
        self.locate_3 = qhih_att
        self.track_3 = qihi_att
        self.q_net = q_net
        self.v_net = v_net
        self.h_net = h_net
        self.qih_net = qih_net
        self.qhi_net = qhi_net
        self.fc1 = nn.Linear(args.nhid * 4, self.ninp)
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        # self.fch = FCNet([args.nhid * 2, args.nhid * 2])
        # self.layernorm = LayerNorm(args.nhid*2)

    def forward(self, image, question, history, answer, tans, rnd, Training=True, sampling=False):

        # prepare I, Q, H
        image = self.img_embed(image)
        w_emb = self.w_emb(question)
        q_emb, ques_hidden = self.q_emb(w_emb)  # [batch, q_dim]

        hw_emb = self.w_emb(history)
        h_emb, _ = self.h_emb(hw_emb)  # [batch * rnd, h_dim]
        h_emb = h_emb.view(-1, rnd, h_emb.size(1))

        # cap & image
        # qc_att = self.v_att(image, h_emb[:, 0, :])
        # qc_emb = (qc_att * image).sum(1)
        # qc_emb = self.fch(qc_emb * q_emb)

        # question & image --> qi
        qv_att = self.track_1(image, q_emb)
        qv_emb = (qv_att * image).sum(1)  # [batch, v_dim]

        # question & history --> qh
        qh_att = self.locate_1(h_emb, q_emb)
        qh_emb = (qh_att * h_emb).sum(1)  # [batch, h_dim]
        # qh_emb = self.fch(qh_emb+q_emb)
        # qh_emb = self.layernorm(qh_emb+h_emb[:,0,:])

        # qh & image --> qhi
        qhi_att = self.track_2(image, qh_emb)
        qhi_emb = (qhi_att * image).sum(1)  # [batch, v_dim]

        # qi & history --> qih
        qih_att = self.locate_2(h_emb, qv_emb)
        qih_emb = (qih_att * h_emb).sum(1)  # [batch, h_dim]
        
        q_re = self.q_net(q_emb)
        qih_emb = self.h_net(qih_emb)
        qih_emb = q_re * qih_emb

        qhi_emb = self.v_net(qhi_emb)
        qhi_emb = q_re * qhi_emb

        # qih & i --> qihi
        qihi_att = self.track_3(image, qih_emb)
        qihi_emb = (qihi_att * image).sum(1)

        # qhi & his --> qhih
        qhih_att = self.locate_3(h_emb, qhi_emb)
        qhih_emb = (qhih_att * h_emb).sum(1)

        q_repr = self.q_net(q_emb)
        qhi_repr = self.qhi_net(qihi_emb)
        qqhi_joint_repr = q_repr * qhi_repr

        qih_repr = self.qih_net(qhih_emb)
        qqih_joint_repr = q_repr * qih_repr

        joint_repr = torch.cat([self.w1(qqhi_joint_repr), self.w2(qqih_joint_repr)], 1)  # [batch, h_dim * 2
        joint_repr = F.tanh(self.fc1(F.dropout(joint_repr, self.dropout, training=self.training)))

        _, ques_hidden = self.decoder(joint_repr.view(-1, 1, self.ninp), ques_hidden)

        if sampling:
            batch_size, _, _ = image.size()
            sample_ans_input = Variable(torch.LongTensor(batch_size, 1).fill_(2).cuda())
            sample_opt = {'beam_size': 1}
            seq, seqLogprobs = self.decoder.sample(self.w_emb, sample_ans_input, ques_hidden, sample_opt)
            sample_ans = self.w_emb(Variable(seq))
            ans_emb = self.w_emb(tans)
            sample_ans = torch.cat([w_emb, joint_repr.view(batch_size, -1, self.ninp),sample_ans], 1)
            ans_emb = torch.cat([w_emb, joint_repr.view(batch_size, -1, self.ninp), ans_emb], 1)
            return sample_ans, ans_emb

        if not Training:
            batch_size, _, hid_size = image.size()
            hid_size = int(hid_size / 2)
            hidden_replicated = []
            for hid in ques_hidden:
                hidden_replicated.append(hid.view(2, batch_size, 1,hid_size).expand(2,
                    batch_size, 100, hid_size).clone().view(2, -1, hid_size))
            hidden_replicated = tuple(hidden_replicated)
            ques_hidden = hidden_replicated

        emb = self.w_emb(answer)
        pred, _ = self.decoder(emb, ques_hidden)
        return pred


def build_baseline0_newatt2(args, num_hid):
    w_emb = WordEmbedding(args.vocab_size, args.ninp, 0.0)
    q_emb = QuestionEmbedding2(args.ninp, num_hid, args.nlayers, True, 0.0)
    h_emb = QuestionEmbedding2(args.ninp, num_hid, args.nlayers, True, 0.0)
    v_att = NewAttention(args.nhid*2, q_emb.num_hid*2, num_hid*2)
    h_att = NewAttention(args.nhid*2, q_emb.num_hid*2, num_hid*2)
    qih_att = NewAttention(args.nhid*2, q_emb.num_hid*2, num_hid*2)
    qhi_att = NewAttention(args.nhid*2, q_emb.num_hid*2, num_hid*2)
    q_net = FCNet([q_emb.num_hid*2, num_hid*2])
    v_net = FCNet([args.nhid*2, num_hid*2])
    h_net = FCNet([args.nhid*2, num_hid*2])
    qih_net = FCNet([args.nhid*2, num_hid*2])
    qhi_net = FCNet([args.nhid*2, num_hid*2])
    qhih_att = NewAttention(args.nhid*2, q_emb.num_hid*2, num_hid*2)
    qihi_att = NewAttention(args.nhid*2, q_emb.num_hid*2, num_hid*2)

    decoder = netG(args)
    return BaseModel2(w_emb, q_emb, h_emb, v_att, h_att, q_net, v_net, h_net, qih_att, qhi_att, qih_net, qhi_net,
                     decoder, args, qhih_att, qihi_att)

class attflat(nn.Module):
    def __init__(self, args):
        super(attflat, self).__init__()
        self.mlp = FCNet([args.nhid * 2, args.nhid, 1])
        self.fc = nn.Linear(args.nhid*2, args.nhid*2)

    def forward(self, x):
        batch_size, q_len, nhid = x.size()
        att = self.mlp(x.view(-1, nhid))
        att = F.softmax(att, dim=1)
        x_atted = (att.view(batch_size, q_len, -1) * x.view(batch_size, q_len, -1)).sum(1)
        x_atted = self.fc(x_atted)

        return x_atted
