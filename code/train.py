# -*- coding: utf-8 -*-
"""
    train a neural network for medical diagnosis classification
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

#import torchtext
#from torchtext.data.utils import get_tokenizer
import os
import torch
#from torchtext.legacy.data import BucketIterator,Iterator
import argparse
import math
import time
from tqdm import tqdm
import numpy as np
import random
from model.transformer import TransformerModel
from model.fastTextModel import FastTextModel
from model.textCNN import TextCNNModel,TextCNNConfig
from model.DPCNN import DPCNNConfig,DPCNNModel
from model.TextRNN import TextRNNConfig,TextRNNModel
from model.TextRCNN import TextRCNNConfig,TextRCNNModel
from model.TextRNN_Att import TextRNNAttConfig,TextRNNAttModel
from model.transformer import TransformerConfig,TransformerModel
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  # ignore warnings due to update in pytorch version
from utils.eval import cal_accuracy
from utils.kFoldData import KFoldDataLoader
import gensim
from gensim.models import Word2Vec
import pandas as pd
import pickle

def train_epoch(model, training_data, optimizer, opt, device):
    """
    train a epoch
    """
    model.train()
    losses = []
    acces = []

    desc = '  - (Training)   '
    for report,label in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        report = report.type(torch.LongTensor).to(device)
        label = label.to(device).float()
        # print(report.shape,label.shape)

        optimizer.zero_grad()
        pred = model(report)    #prediction
        loss = F.binary_cross_entropy(pred,label)
        acc = cal_accuracy(pred,label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        acces.append(acc)

    # print(losses)
    loss_per_seq = sum(losses)/len(losses)
    acc_per_seq = sum(acces)/len(acces)
    return loss_per_seq,acc_per_seq

def eval_epoch(model, validation_data, opt,device):
    """
    eval a epoch
    """
    model.eval()
    losses = []
    acces = []

    desc = '  - (Validation)   '
    for report,label in tqdm(validation_data, mininterval=1, desc=desc, leave=False):
        report = report.type(torch.LongTensor).to(device)
        label = label.to(device).float()
        pred = model(report)    #prediction, we do not use mask now!
        loss = F.binary_cross_entropy(pred,label)
        losses.append(loss.item())
        acc = cal_accuracy(pred,label)
        acces.append(acc)

    loss_per_seq = sum(losses)/len(losses)
    acc_per_seq = sum(acces)/len(acces)
    return loss_per_seq,acc_per_seq


def train(model, training_data, validation_data, optimizer, device, opt):
    """
    train the model
    """

    def print_performances(header,start_time,loss,acc):
        print('  - {header:12} loss : {loss:5.5f} , acc : {acc:0.3f} , elapse : {elapse:5.5f} min'.format(
              header=header,loss=loss, acc=acc, elapse=(time.time()-start_time)/60))

    valid_losses = []
    valid_accs = []
    train_losses = []
    train_accs = []
    for epoch_i in range(opt.epoch):

        print('[ Epoch', epoch_i, ']')

        # train in training dataset
        start = time.time()
        train_loss,train_acc = train_epoch(model, training_data,
            optimizer, opt,device)
        print_performances('Training',start,train_loss,train_acc)
        train_losses += [train_loss]
        train_accs += [train_acc]

        # eval in validation dataset
        start = time.time()
        valid_loss,valid_acc = eval_epoch(model, validation_data,
            opt,device)
        print_performances('Validation',start,valid_loss,valid_acc)

        valid_losses += [valid_loss]
        valid_accs += [valid_acc]

        # save the model
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict(),'valid_loss':valid_loss,'valid_acc':valid_acc}
        if opt.save_mode == 'all':
            model_name = '{model_name}_loss_{loss:3.3f}.chkpt'.format(model_name=opt.model,loss=train_loss)
            torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
        elif opt.save_mode == 'best':
            model_name = '{model_name}.chkpt'.format(model_name=opt.model)
            if valid_loss <= min(valid_losses) and valid_acc >= max(valid_accs):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')
        loss_over_shit = 100*(min(valid_losses) - train_loss)/min(valid_losses)
        acc_over_shit = 100*(train_acc > max(valid_accs))/max(valid_accs)
        if loss_over_shit > 10 and acc_over_shit > 0.0:
            break

    return train_losses,train_accs,valid_losses,valid_accs

def load_model(opt,device):
    m_model = None
    if "DPCNN" in opt.model:
        model_config = DPCNNConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.padding_idx = opt.pad_token
        m_model = DPCNNModel(model_config).to(device)
    # TextRNN
    elif "TextRNN" in opt.model:
        model_config = TextRNNConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.padding_idx = opt.pad_token
        m_model = TextRNNModel(model_config).to(device)
    elif "TextAttRNN" in opt.model:
        model_config = TextRNNAttConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.padding_idx = opt.pad_token
        m_model = TextRNNAttModel(model_config).to(device)
    elif  "TextRCNN" in opt.model:
        model_config = TextRCNNConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.padding_idx = opt.pad_token
        m_model = TextRCNNModel(model_config).to(device)
    elif  "Transformer" in opt.model:
        model_config = TransformerConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.device = device
        model_config.padding_idx = opt.pad_token
        m_model = TransformerModel(model_config).to(device)
    else:
        # default TextCNN
        model_config = TextCNNConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.padding_idx = opt.pad_token
        m_model = TextCNNModel(model_config).to(device)
    return m_model

def main():
    parser = argparse.ArgumentParser()
    # parameters of training
    parser.add_argument('-epoch', type=int, default=13)
    parser.add_argument('-max_len',type=int,default=70)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-fold_k', type=int, default=5)
    parser.add_argument('-label_smoothing', type=float, default=0.000,help="The possibility of flip train label")

    # ntokens will be changed according to train data
    parser.add_argument('-model',default="TextCNN",choices=["TextCNN","DPCNN","TextRNN","TextAttRNN","TextRCNN","Transformer"],help="Net work for learning")
    parser.add_argument('-ntokens', type = int, default= 858)
    parser.add_argument('-nemb', type=int, default=500)
    parser.add_argument('-nclass', type=int, default=17)

    # parameters of optimizer
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    # word2vector pretrain model
    parser.add_argument('-no_word2vec_pretrain',action='store_true',help="if we use pretrain word2vector model")
    parser.add_argument('-word2vec_path',default=os.path.join(os.getenv('PROJTOP'),'user_data/word_pretrain/word2vector.model'),
                        help="path of word2vector model")

    # parameters of saving data
    parser.add_argument('-output_dir', type=str,
        default=os.path.join(os.getenv('PROJTOP'),'user_data/model_data'))
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    opt = parser.parse_args()

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if opt.output_dir and not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Start getting data...")
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/medical_nlp_round1_data/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
    #train_df = train_df.sample(frac=1).reset_index(drop=True)
    k_fold_data_loader = KFoldDataLoader(
                                df=train_df,
                                batch_size = opt.batch_size,
                                k = opt.fold_k,
                                nclass = opt.nclass,
                                max_len = opt.max_len,
                                label_smoothing = opt.label_smoothing
                                )
    opt.pad_token = k_fold_data_loader.pad_idx
    opt.ntokens = k_fold_data_loader.pad_idx + 1
    print("Finish getting data ~ v ~")

    # create the log file to save training performance
    log_train_file = os.path.join(opt.output_dir, 'train.csv')
    print('[Info] Training performance will be written to file: {}'.format(
        log_train_file))
    train_info = {}
    min_valid_losses = []
    max_valid_accs = []

    print(opt)
    model_name = opt.model
    stacking_data = {"pred":[],"label":[]}
    for k_index in range(0,opt.fold_k):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('{0}th fold (total : {1})'.format(k_index,opt.fold_k))
        train_iterator, val_iterator ,train_dataset,valid_dataset= k_fold_data_loader.get_ith_data(k_index)

        print("Create model...")
        m_model = load_model(opt,device)
        opt.model = "%s_fold%d"%(model_name,k_index+1)

        word_vec_file = open(opt.word2vec_path,'rb')
        word2vec_model = pickle.load(word_vec_file)
        if not opt.no_word2vec_pretrain:
            print("use pretrained word2vector model")
            m_model.use_pretrain_word2vec(word2vec_model)
        word_vec_file.close()
        print(m_model)
        print("Finish model producing ~ v ~")

        optimizer = optim.Adam(m_model.parameters(), betas=(0.9, 0.98), eps=1e-05)

        print("Start training...")
        train_losses,train_accs,valid_losses,valid_accs = train(m_model, train_iterator, val_iterator, optimizer, device, opt)
        train_info["{}th_fold_train_loss".format(k_index)] = train_losses
        train_info["{}th_fold_train_acc".format(k_index)] = train_accs
        train_info["{}th_fold_valid_loss".format(k_index)] = valid_losses
        train_info["{}th_fold_valid_ass".format(k_index)] = valid_accs
        min_valid_losses.append(min(valid_losses))
        max_valid_accs.append(max(valid_accs))
        print("Finish training ~ v ~")

        for j in range(len(valid_dataset)):
            report,label = valid_dataset[j]
            pred = m_model(torch.LongTensor([report]).to(device)).squeeze()
            res_report = ""
            res_label = ""
            for m,n in zip(pred,label):
                res_report += "%.8f "%(m)
                res_label += "%d "%(n)

            stacking_data["pred"].append(res_report)
            stacking_data["label"].append(res_label)

    # save the train info
    train_info_df = pd.DataFrame(train_info)
    train_info_df.to_csv(log_train_file)
    print('min losses:',min_valid_losses,'mean losses:',sum(min_valid_losses)/len(min_valid_losses))
    print('max accs:',max_valid_accs,'mean accs:',sum(max_valid_accs)/len(max_valid_accs))

    # save the train info
    stack_df = pd.DataFrame(stacking_data)
    print(stack_df)
    stack_df.to_csv(os.path.join(os.getenv('PROJTOP'),'user_data/tmp_data/%s_pred.csv'%(model_name)))

if __name__ == "__main__":
    main()
