# -*- coding: utf-8 -*-
"""
    train a neural network for medical diagnosis classification
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟),Fu Yangsheng,Huang Zheng
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
from model.TextMRCNN import TextMRCNNConfig,TextMRCNNModel
from model.TextRNN_Att import TextRNNAttConfig,TextRNNAttModel
from model.transformer import TransformerConfig,TransformerModel
from model.bert import BERTConfig,BERTModel
from transformers import RobertaTokenizerFast
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  # ignore warnings due to update in pytorch version
from utils.eval import cal_accuracy
import gensim
from gensim.models import Word2Vec
import pandas as pd
import pickle

def train_epoch(model, training_data, optimizer, opt, device,scheduler=None):
    """
    train a epoch
    """
    model.train()
    losses = []
    acces = []

    desc = '  - (Training)   '
    for report,label in training_data:
        # modify the data format
        if "BERT" in opt.model:
            report = report.to(device)
        else:
            report = report.type(torch.LongTensor).to(device)
        label = label.to(device).float()

        # predict and calculate the loss, accuracy
        optimizer.zero_grad()
        pred = model(report)    #prediction
        loss = 0.6*F.binary_cross_entropy(pred[:17],label[:17]) +  0.4*F.binary_cross_entropy(pred[17:],label[17:])
        acc = cal_accuracy(pred,label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

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
    for report,label in validation_data:
        # modify the data format
        if "BERT" in opt.model:
            report = report.to(device)
        else:
            report = report.type(torch.LongTensor).to(device)
        label = label.to(device).float()

        # predict and calculate the loss, accuracy
        pred = model(report)    #prediction, we do not use mask now!
        loss = 0.6*F.binary_cross_entropy(pred[:17],label[:17]) +  0.4*F.binary_cross_entropy(pred[17:],label[17:])
        losses.append(loss.item())
        acc = cal_accuracy(pred,label)
        acces.append(acc)

    loss_per_seq = sum(losses)/len(losses)
    acc_per_seq = sum(acces)/len(acces)
    return loss_per_seq,acc_per_seq


def train(model, training_data, validation_data, optimizer, device, opt,scheduler=None):
    """
    train the model
    """

    def performance_info(header,start_time,loss,acc):
        return '  - {header:12} loss : {loss:5.5f} , acc : {acc:0.3f} , elapse : {elapse:5.5f} min'.format(
                header=header,loss=loss, acc=acc, elapse=(time.time()-start_time)/60)

    valid_losses = []
    valid_accs = []
    train_losses = []
    train_accs = []
    min_valid_loss_index = 0
    for epoch_i in range(opt.epoch):

        epoch_info = '[ Epoch %d ]'%(epoch_i)

        # train in training dataset
        start = time.time()
        train_loss,train_acc = train_epoch(model, training_data,
            optimizer, opt,device,scheduler=scheduler)
        epoch_info += performance_info('Training',start,train_loss,train_acc)
        train_losses += [train_loss]
        train_accs += [train_acc]

        # eval in validation dataset
        start = time.time()
        valid_loss,valid_acc = eval_epoch(model, validation_data,
            opt,device)
        epoch_info += performance_info('Validation',start,valid_loss,valid_acc)

        valid_losses += [valid_loss]
        valid_accs += [valid_acc]

        # save the model
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict(),'valid_loss':valid_loss,'valid_acc':valid_acc}
        if opt.save_mode == 'all':
            if (epoch_i + 1 - opt.save_start_epoch)%(opt.save_per_epoch) == 0:
                model_name = '{model_name}_epoch_{epoch:d}.chkpt'.format(model_name=opt.model,epoch=epoch_i+1,loss=valid_loss)
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
        elif opt.save_mode == 'best':
            model_name = '{model_name}.chkpt'.format(model_name=opt.model)
            if valid_loss <= min(valid_losses) and valid_acc >= max(valid_accs):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                epoch_info += '    - [Info] The checkpoint file has been updated.'
        if opt.stop_early:
            loss_over_shit = 100*(min(valid_losses) - train_loss)/min(valid_losses)
            acc_over_shit = 100*(train_acc > max(valid_accs))/max(valid_accs)
            if loss_over_shit > 0 and acc_over_shit > 0:
                break
        print(epoch_info)
    return train_losses,train_accs,valid_losses,valid_accs

def load_model(opt,device):
    """
    load the model
    args:
        opt : option for model, like name of model and setting of a model
            there are several model which you can choose, like TextRCNN,TextCNN,DPCNN,BERT.
            we advice the model name like "BERT_fold1_PLoss0.24_balabala"
        device: which device you want to run the model
    return:
        the model
    """
    m_model = None
    if "DPCNN" in opt.model:
        model_config = DPCNNConfig(n_vocab=opt.ntokens,embedding=opt.nemb,num_class=opt.nclass,max_seq_len=opt.max_len,dropout=opt.dropout)
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
        model_config = TextRCNNConfig(n_vocab=opt.ntokens,embedding=opt.nemb,
                max_seq_len=opt.max_len,num_class=opt.nclass,
                dropout=opt.dropout,lstm_layer=opt.lstm_layer,
                hidden_size=opt.hidden_size,lstm_dropout=opt.lstm_dropout
                )
        model_config.padding_idx = opt.pad_token
        m_model = TextRCNNModel(model_config).to(device)
    elif  "TextMRCNN" in opt.model:
        model_config = TextMRCNNConfig(n_vocab=opt.ntokens,embedding=opt.nemb,
                max_seq_len=opt.max_len,num_class=opt.nclass,
                dropout=opt.dropout,lstm_layer=opt.lstm_layer,
                hidden_size=opt.hidden_size,lstm_dropout=opt.lstm_dropout
                )
        model_config.padding_idx = opt.pad_token
        m_model = TextMRCNNModel(model_config).to(device)
    elif  "Transformer" in opt.model:
        model_config = TransformerConfig(opt.ntokens,opt.nemb,opt.nclass)
        model_config.device = device
        model_config.padding_idx = opt.pad_token
        m_model = TransformerModel(model_config).to(device)
    elif  "BERT" in opt.model:
        model_config = BERTConfig(num_class = opt.nclass,dropout=opt.dropout,
            frazing_encode=opt.frazing_bert_encode,pre_train_path=opt.bert_path)
        m_model = BERTModel(model_config).to(device)
    else:
        # default TextCNN
        model_config = TextCNNConfig(n_vocab=opt.ntokens,embedding=opt.nemb,num_class=opt.nclass,max_seq_len=opt.max_len,dropout=opt.dropout)
        model_config.padding_idx = opt.pad_token
        m_model = TextCNNModel(model_config).to(device)
    return m_model

def main():
    parser = argparse.ArgumentParser()
    # parameters of training
    parser.add_argument('-epoch', type=int, default=100,help="epochs you want to run")
    parser.add_argument('-max_len',type=int,default=100,help="mac length of the sentence")
    parser.add_argument('-batch_size', type=int, default=128,help="size of batch")
    parser.add_argument('-stop_early',action="store_true", help="stop early")
    parser.add_argument('-fold_k', type=int, default=5,help="number of fold")
    parser.add_argument('-fold_index', type=int, default=-1,help="define which fold we will run. Run all fold if fold_index < 0")
    parser.add_argument('-label_smoothing', type=float, default=0.000,help="The possibility of flip train label")

    # ntokens will be changed according to train data
    parser.add_argument('-model',default="TextCNN",
        # choices=["BERT","TextCNN","DPCNN","TextRNN","TextAttRNN","TextRCNN","Transformer"],
        help="Net work for learning")
    parser.add_argument('-ntokens', type = int, default= 858,help="number of tokens")
    parser.add_argument('-nemb', type=int, default=100,help="embeding size")
    parser.add_argument('-hidden_size', type=int, default=256,help="hidden size")
    parser.add_argument('-lstm_dropout', type=float, default=0.1,help="dropout rate of lstm layer")
    parser.add_argument('-nclass', type=int, default=29,help="number of class")
    parser.add_argument('-dropout', type=float, default=0.5,help="dropout rate of end layer")
    parser.add_argument('-lstm_layer', type=int, default=2,help="number of lstm layer")
    parser.add_argument('-frazing_bert_encode',action="store_true", help="frazing bert encode when train")
    parser.add_argument('-bert_path',default=os.path.join(os.getenv('PROJTOP'),"user_data/bert"),help="path of pretrained BERT")
    parser.add_argument('-tokenizer_path',default=os.path.join(os.getenv('PROJTOP'),"user_data/bert"),help="path of tokenizer")

    # data enhancement, not do this defaultly
    parser.add_argument('-eda_alpha',type=float,default=0.0,help="alpha of eda")
    parser.add_argument('-n_aug',type=float,default=0.0,help="n of aug")

    # parameters of optimizer
    parser.add_argument('-lr', type=float, default=1.e-3,help="learning rate, advice: 1.e-5 for bert and 1.e-3 for others")
    parser.add_argument('-seed', type=int, default=None,help="random seed")

    # word2vector pretrain model
    parser.add_argument('-no_word2vec_pretrain',action='store_true',help="if we use pretrain word2vector model")
    parser.add_argument('-word2vec_path',default=os.path.join(os.getenv('PROJTOP'),'user_data/word_pretrain/word2vector.model'),
                        help="path of word2vector model")

    # parameters of saving data
    parser.add_argument('-input', type=str,
        default=os.path.join(os.getenv('PROJTOP'),'tcdata/train.csv'),help="path for train.csv")
    parser.add_argument('-output_dir', type=str,
        default=os.path.join(os.getenv('PROJTOP'),'user_data/model_data'),help="the path to save the trained model")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-save_start_epoch', type=int, default = 5,help = "the epoch after which we start save the model")
    parser.add_argument('-save_per_epoch', type=int,default=5, help = "save the model every save_per_epoch after save_start_epoch")
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

    # device to run the training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("If the cuda is avaliable ?")
    print(torch.cuda.is_available())

    print("Start getting data...")
    train_df = pd.read_csv(opt.input,sep="\|,\|",names=["id","report","label"],index_col=0)
    #train_df = train_df.sample(frac=1).reset_index(drop=True)
    # The dataset format of BERT model and others are quite different, so we write the dataset class respectively
    if "BERT" in opt.model:
        from utils.bert_kFoldData import KFoldDataLoader
        tokenizer = RobertaTokenizerFast.from_pretrained(opt.tokenizer_path, max_len=opt.max_len)
        tokens = [str(i) for i in range(857,-1,-1)]
        tokenizer.add_tokens(tokens)
        k_fold_data_loader = KFoldDataLoader(
                                df=train_df,
                                tokenizer = tokenizer,
                                batch_size = opt.batch_size,
                                k = opt.fold_k,
                                nclass = opt.nclass,
                                max_len = opt.max_len,
                                label_smoothing = opt.label_smoothing,
                                eda_alpha = opt.eda_alpha,
                                n_aug = opt.n_aug
                                )
        opt.pad_token = tokenizer.vocab["<pad>"]
        opt.ntokens = len(tokenizer.vocab)
    else:
        from utils.kFoldData import KFoldDataLoader
        k_fold_data_loader = KFoldDataLoader(
                                df=train_df,
                                batch_size = opt.batch_size,
                                k = opt.fold_k,
                                nclass = opt.nclass,
                                max_len = opt.max_len,
                                label_smoothing = opt.label_smoothing,
                                eda_alpha = opt.eda_alpha,
                                n_aug = opt.n_aug
                                )
        opt.pad_token = k_fold_data_loader.pad_idx
        opt.ntokens = k_fold_data_loader.pad_idx + 1
    print("Finish getting data ~ v ~")

    # create the log file to save training performance
    if opt.fold_index > 0:
        log_train_file = os.path.join(opt.output_dir, '%s_fold%d_train.csv'%(opt.model,opt.fold_index))
    else:
        log_train_file = os.path.join(opt.output_dir, '%s_train.csv'%(opt.model))
    print('[Info] Training performance will be written to file: {}'.format(
        log_train_file))
    train_info = {}
    min_valid_losses = []
    max_valid_accs = []

    print(opt)
    model_name = opt.model
    stacking_data = {"pred":[],"label":[]}
    for k_index in range(0,opt.fold_k):

        # if opt.fold_index is set to a integer bigger than zero,
        # we only a definite fold
        if opt.fold_index < 0:
            pass
        else:
            if (k_index+1) < opt.fold_index:
                continue
            elif (k_index+1) == opt.fold_index:
                pass
            else:
                break

        # get the dataset and dataloader
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('{0}th fold (total : {1})'.format(k_index,opt.fold_k))
        train_iterator, val_iterator ,train_dataset,valid_dataset= k_fold_data_loader.get_ith_data(k_index)

        # create the model
        print("Create model...")
        m_model = load_model(opt,device)
        opt.model = "%s_fold%d"%(model_name,k_index+1)

        # load the pretrained word vector or not
        if not opt.no_word2vec_pretrain and ("BERT" not in model_name):
            word_vec_file = open(opt.word2vec_path,'rb')
            word2vec_model = pickle.load(word_vec_file)
            print("use pretrained word2vector model")
            m_model.use_pretrain_word2vec(word2vec_model)
            word_vec_file.close()
        if k_index == 0:
            print(m_model)

        # the optimizer
        optimizer = optim.AdamW(m_model.parameters(),lr=opt.lr,betas=(0.9, 0.98), eps=1e-05)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=1.e-6,last_epoch=-1)

        # train the model
        print("Start training...")
        train_losses,train_accs,valid_losses,valid_accs = train(m_model, train_iterator, val_iterator, optimizer, device, opt,scheduler=lr_scheduler)
        train_info["{}th_fold_train_loss".format(k_index)] = train_losses
        train_info["{}th_fold_train_acc".format(k_index)] = train_accs
        train_info["{}th_fold_valid_loss".format(k_index)] = valid_losses
        train_info["{}th_fold_valid_ass".format(k_index)] = valid_accs
        min_valid_losses.append(min(valid_losses))
        max_valid_accs.append(max(valid_accs))
        print("Finish training ~ v ~")

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
