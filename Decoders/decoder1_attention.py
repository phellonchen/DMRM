import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from classifier import SimpleClassifier

class _netG(nn.Module):
    def __init__(self, args):
        super(_netG, self).__init__()

        self.ninp = args.ninp
        self.nhid = args.nhid
        self.nlayers = args.nlayers
        self.dropout = args.dropout
        self.rnn = getattr(nn, 'LSTM')(self.ninp, self.nhid, self.nlayers, bidirectional=False, dropout=self.dropout, batch_first=True)
        self.rnn_type = 'LSTM'

        self.decoder =SimpleClassifier(self.nhid*2, self.nhid*4, args.vocab_size, self.dropout)
        self.d = args.dropout
        self.beta = 3
        self.vocab_size = args.vocab_size
        # self.init_weights()
        self.w_q = nn.Linear(self.nhid*2, self.nhid)
        self.ans_q = nn.Linear(self.nhid, self.nhid)
        self.Wa_q = nn.Linear(self.nhid, 1)

        self.w_h = nn.Linear(self.nhid*2, self.nhid)
        self.ans_h = nn.Linear(self.nhid, self.nhid)
        self.Wa_h = nn.Linear(self.nhid, 1)

        self.w_i = nn.Linear(self.nhid*2, self.nhid)
        self.ans_i = nn.Linear(self.nhid, self.nhid)
        self.Wa_i = nn.Linear(self.nhid, 1)

        self.concat = nn.Linear(self.nhid*3, self.nhid)
        # self.fusion = nn.Linear(self.nhid*2, self.nhid*2)

    def init_weights(self):
        self.decoder.weight = nn.init.xavier_uniform(self.decoder.weight)
        self.decoder.bias.data.fill_(0)

    def forward(self, emb, question, history, image, hidden):
        ques_length = question.size(1)
        his_length = history.size(1)
        img_length = image.size(1)
        batch_size, ans_length, _  = emb.size()
        question = question.contiguous()
        seqLogprobs = []
        for index in range(ans_length):
            input_ans = emb[:, index, :].unsqueeze(1)
            output, hidden = self.rnn(input_ans, hidden)
            input_ans = output.squeeze(1)
            ques_emb = self.w_q(question.view(-1, 2*self.nhid)).view(-1, ques_length, self.nhid)
            input_ans_q = self.ans_q(input_ans).view(-1, 1, self.nhid)
            atten_emb_q = F.tanh(ques_emb + input_ans_q.expand_as(ques_emb))
            ques_atten_weight = F.softmax(self.Wa_q(F.dropout(atten_emb_q, self.d, training=self.training).view(-1, self.nhid)).view(-1, ques_length), 1)
            ques_attn_feat = torch.bmm(ques_atten_weight.view(-1, 1, ques_length), ques_emb.view(-1,ques_length, self.nhid))
            
            input_ans_h = self.ans_h(input_ans).view(-1, 1, self.nhid)
            his_emb = self.w_h(history.view(-1, 2* self.nhid)).view(-1, his_length, self.nhid)
            atten_emb_h = F.tanh(his_emb + input_ans_h.expand_as(his_emb))
            his_atten_weight = F.softmax(self.Wa_h(F.dropout(atten_emb_h, self.d, training=self.training).view(-1, self.nhid)).view(-1, his_length), 1)
            his_attn_feat = torch.bmm(his_atten_weight.view(-1, 1, his_length), his_emb.view(-1, his_length, self.nhid))
            
            input_ans_i = self.ans_i(input_ans).view(-1, 1, self.nhid)
            img_emb = self.w_i(image.view(-1, 2* self.nhid)).view(-1, img_length, self.nhid)
            atten_emb_i = F.tanh(img_emb + input_ans_i.expand_as(img_emb))
            img_atten_weight = F.softmax(self.Wa_i(F.dropout(atten_emb_i, self.d, training=self.training).view(-1, self.nhid)).view(-1, img_length), 1)
            img_attn_feat = torch.bmm(img_atten_weight.view(-1, 1, img_length), img_emb.view(-1, img_length, self.nhid))
            
            concat_feat = torch.cat((ques_attn_feat.view(-1, self.nhid), his_attn_feat.view(-1, self.nhid), img_attn_feat.view(-1, self.nhid)),1)
            concat_feat = F.tanh(self.concat(F.dropout(concat_feat, self.d, training=self.training)))
            fusion_feat = torch.cat((output.squeeze(1), concat_feat),1)

            fusion_feat = F.dropout(fusion_feat, self.d, training=self.training)
            decoded = self.decoder(fusion_feat.view(-1, 2*self.nhid))
            logprob = F.log_softmax(self.beta * decoded, 1)
            seqLogprobs.append(logprob)

        return torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1).contiguous(), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def sample_beam(self, netW, input, hidden_state, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = input.size(1)

        # assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq_all = torch.LongTensor(self.seq_length, batch_size, beam_size).zero_()
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            # copy the hidden state for beam_size time.
            state = []
            for state_tmp in hidden_state:
                state.append(state_tmp[:, k, :].view(1, 1, -1).expand(1, beam_size, self.nhid).clone())

            state = tuple(state)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
            for t in range(self.seq_length + 1):
                if t == 0:  # input <bos>
                    it = input.data.resize_(1, beam_size).fill_(self.vocab_size)
                    xt = netW(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    ys, ix = torch.sort(logprobsf, 1,
                                        True)  # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1
                    for cc in range(cols):  # for each column (word, essentially)
                        for qq in range(rows):  # for each beam expansion
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[qq, cc]
                            if beam_seq[t - 2, qq] == self.vocab_size:
                                local_logprob.data.fill_(-9999)

                            candidate_logprob = beam_logprobs_sum[qq] + local_logprob
                            candidates.append({'c': ix.data[qq, cc], 'q': qq, 'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})

                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == self.vocab_size or t == self.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix]
                                                       })

                    # encode as vectors
                    it = beam_seq[t - 1].view(1, -1)
                    xt = netW(Variable(it.cuda()))

                if t >= 1:
                    state = new_state

                output, state = self.rnn(xt, state)

                output = F.dropout(output, self.d, training=self.training)
                decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
                logprobs = F.log_softmax(self.beta * decoded)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            for ii in range(beam_size):
                seq_all[:, k, ii] = self.done_beams[k][ii]['seq']

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, netW, input, state, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        seq_length = opt.get('seq_length', 9)
        self.seq_length = seq_length

        if beam_size > 1:
            return self.sample_beam(netW, input, state, opt)

        batch_size = input.size(1)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = input.data
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it,
                                                             requires_grad=False))  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            xt = netW(Variable(it.view(-1, 1), requires_grad=False))

            if t >= 1:
                seq.append(it)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))
                it = torch.unsqueeze(it, 0)

            output, state = self.rnn(xt, state)
            output = F.dropout(output, self.d, training=self.training)
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            logprobs = F.log_softmax(self.beta * decoded, 1)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)







