import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import datetime
import numpy as np
import sys

def LangMCriterion(input, target):
    target = target.view(-1, 1)
    logprob_select = torch.gather(input, 1, target)

    mask = target.data.gt(0)  # generate the mask
    if isinstance(input, Variable):
        mask = Variable(mask, volatile=input.volatile)
    out = torch.masked_select(logprob_select, mask)
    loss = -torch.sum(out)  # get the average loss.
    return loss

def train(model, train_loader, eval_loader, args):
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s-%s-%s' % (t.year, t.month, t.day, t.hour, t.minute)
    save_path = os.path.join(args.output, cur_time)
    args.save_path = save_path
    utils.create_dir(save_path)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    lr_default = args.lr if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10, 30, lr_decay_step) if eval_loader is not None else range(10, 20, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    
    logger = utils.Logger(os.path.join(save_path, 'log.txt'))
    for arg in vars(args):
        logger.write('{:<20}: {}'.format(arg, getattr(args, arg)))
    best_eval_score = 0
    model.train()
    start_time = time.time()
    best_cnt = 0

    print('Training ... ')
    for epoch in range(args.epochs):
       
        total_loss = 0
        count = 0
        train_score = 0
        t = time.time()
        train_iter = iter(train_loader)

        # TODO: get learning rate
        # lr = adjust_learning_rate(optim, epoch, args.lr)
        if epoch < 4:
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            lr = optim.param_groups[0]['lr']
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            lr =  optim.param_groups[0]['lr']
        else:
            lr = optim.param_groups[0]['lr']
        iter_step = 0
        for i in range(len(train_loader)):
            average_loss_tmp = 0
            count_tmp = 0
            train_data = next(train_iter)
            image, image_id, history, question, answer, answerT, ans_len, ans_idx, ques_ori, opt, opt_len, opt_idx = train_data
            batch_size = question.size(0)
            image = image.view(image.size(0), -1, args.img_feat_size)
            img_input = Variable(image).cuda()
            for rnd in range(10):
                ques = question[:, rnd, :]
                his = history[:, :rnd + 1, :].clone().view(-1, args.his_length)
                ans = answer[:, rnd, :]
                tans = answerT[:, rnd, :]
                opt_ans = opt[:, rnd, :].clone().view(-1, args.ans_length)
                
                ques = Variable(ques).cuda().long()
                his = Variable(his).cuda().long()
                ans = Variable(ans).cuda().long()
                tans = Variable(tans).cuda().long()
                opt_ans = Variable(opt_ans).cuda().long()

                pred = model(img_input, ques, his, ans, tans, rnd + 1)
                loss = LangMCriterion(pred.view(-1, args.vocab_size), tans)
                loss = loss / torch.sum(tans.data.gt(0))

                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)
                optim.step()
                model.zero_grad()

                average_loss_tmp += loss.data[0]
                total_loss += loss.data[0]
                count += 1
                count_tmp += 1
            sys.stdout.write('Training: Epoch {:d} Step {:d}/{:d}  \r'.format(epoch + 1, i + 1, len(train_loader)))
            if (i+1) % 50 == 0:
                average_loss_tmp /= count_tmp
                print("step {} / {} (epoch {}), g_loss {:.3f}, lr = {:.6f}".format(i + 1, len(train_loader), epoch + 1, average_loss_tmp, lr))
        iter_step += 1
        total_loss /= count
        logger.write('Epoch %d : learningRate %4f train loss %4f Time: %3f' % (epoch + 1, lr, total_loss, time.time() - start_time))
        model.eval()
        print('Evaluating ... ')
        start_time = time.time()
        rank_all = evaluate(model, eval_loader, args)
        R1 = np.sum(np.array(rank_all) == 1) / float(len(rank_all))
        R5 = np.sum(np.array(rank_all) <= 5) / float(len(rank_all))
        R10 = np.sum(np.array(rank_all) <= 10) / float(len(rank_all))
        ave = np.sum(np.array(rank_all)) / float(len(rank_all))
        mrr = np.sum(1 / (np.array(rank_all, dtype='float'))) / float(len(rank_all))
        logger.write('Epoch %d: mrr: %f R1: %f R5 %f R10 %f Mean %f time: %.2f' % (epoch + 1, mrr, R1, R5, R10, ave, time.time()-start_time))

        eval_score = mrr

        model_path = os.path.join(save_path, 'model_epoch_%d.pth' % (epoch + 1))
        torch.save({'epoch': epoch,
                    'args': args,
                    'model': model.state_dict()}, model_path)

        if eval_score > best_eval_score:
            model_path = os.path.join(save_path, 'best_model.pth')
            torch.save({'epoch': epoch,
                        'args': args,
                        'model': model.state_dict()}, model_path)
            best_eval_score = eval_score
            best_cnt = 0
        else:
            best_cnt = best_cnt + 1
            if best_cnt > 10:
                break
    return model


def evaluate(model, eval_loader, args, Eval=False):
    rank_all_tmp = []
    eval_iter = iter(eval_loader)
    step = 0
    for i in range(len(eval_loader)):
        eval_data = next(eval_iter)
        image, image_id, history, question, answer, answerT, questionL, opt_answer, \
        opt_answerT, answer_ids, answerLen, opt_answerLen = eval_data

        image = image.view(image.size(0), -1, args.img_feat_size)
        img_input = Variable(image).cuda()
        batch_size = question.size(0)
        for rnd in range(10):
            ques, tans = question[:, rnd, :], opt_answerT[:, rnd, :].clone().view(-1, args.ans_length)
            his = history[:, :rnd + 1, :].clone().view(-1, args.his_length)
            ans = opt_answer[:, rnd, :, :].clone().view(-1, args.ans_length)
            gt_id = answer_ids[:, rnd]

            ques = Variable(ques).cuda().long()
            tans = Variable(tans).cuda().long()
            his = Variable(his).cuda().long()
            ans = Variable(ans).cuda().long()
            gt_index = Variable(gt_id).cuda().long()

            pred = model(img_input, ques, his, ans, tans, rnd + 1, Training=False)
            logprob = - pred.permute(1, 0, 2).contiguous().view(-1, args.vocab_size)

            logprob_select = torch.gather(logprob, 1, tans.t().contiguous().view(-1, 1))

            mask = tans.t().data.eq(0)  # generate the mask
            if isinstance(logprob, Variable):
                mask = Variable(mask, volatile=logprob.volatile)
            logprob_select.masked_fill_(mask.view_as(logprob_select), 0)

            prob = logprob_select.view(args.ans_length, -1, 100).sum(0).view(-1, 100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b * 100

            gt_score = prob.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(prob, 1)

            count = sort_score.lt(gt_score.view(-1, 1).expand_as(sort_score))
            rank = count.sum(1) + 1

            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())
        step += 1
        if Eval:
            sys.stdout.write('Evaluating: {:d}/{:d}  \r'.format(i, len(eval_loader)))
            if (i+1) % 50 == 0:
                R1 = np.sum(np.array(rank_all_tmp) == 1) / float(len(rank_all_tmp))
                R5 = np.sum(np.array(rank_all_tmp) <= 5) / float(len(rank_all_tmp))
                R10 = np.sum(np.array(rank_all_tmp) <= 10) / float(len(rank_all_tmp))
                ave = np.sum(np.array(rank_all_tmp)) / float(len(rank_all_tmp))
                mrr = np.sum(1 / (np.array(rank_all_tmp, dtype='float'))) / float(len(rank_all_tmp))
                print('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' % (i+1, len(eval_loader), mrr, R1, R5, R10, ave))

    return rank_all_tmp

