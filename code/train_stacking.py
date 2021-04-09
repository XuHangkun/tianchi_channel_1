# -*- coding: utf-8 -*-
"""
    predict according to trained model
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟),Fu Yangsheng,Huang Zheng
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

import torch
import argparse
import torchtext
import os
import pandas as pd
from tqdm import tqdm
from model.Stacking import StackingConfig,StackingModel
from utils.stacking_dataset import StackingDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import torch.nn.functional as F
from utils.eval import cal_accuracy


def print_performances(header,start_time,loss,acc):
        print('  - {header:12} loss : {loss:5.5f} , acc : {acc:0.3f} , elapse : {elapse:5.5f} min'.format(
            header=header,loss=loss, acc=acc, elapse=(time.time()-start_time)/60))

def train_epoch(model,training_data,optimizer,opt,device):
    """
    train a epoch
    """
    model.train()
    losses = []
    acces = []

    desc = '  - (Training)   '
    for report,label in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        report = report.to(device)
        label = label.to(device)

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
        report = report.to(device)
        label = label.to(device)
        pred = model(report)    #prediction, we do not use mask now!
        loss = F.binary_cross_entropy(pred,label)
        losses.append(loss.item())
        acc = cal_accuracy(pred,label)
        acces.append(acc)

    loss_per_seq = sum(losses)/len(losses)
    acc_per_seq = sum(acces)/len(acces)
    return loss_per_seq,acc_per_seq


def main():
    parser = argparse.ArgumentParser(description="Stacking")
    parser.add_argument('-models',default=["TextCNN","TextRCNN","DPCNN"],nargs="+",help="Net workS for learning")
    parser.add_argument('-save_mode',default="best",help="save mode")

    parser.add_argument('-input', type = str,
                        default=os.path.join(os.getenv('PROJTOP'),'user_data/tmp_data'),
                        help='input directory')
    parser.add_argument('-n_class',type=int,default=17,help="number of class")
    parser.add_argument('-epoch',type=int,default=20,help="train epoch")
    parser.add_argument('-batch_size',type=int,default=256,help="batch size")
    # parameters of saving data
    parser.add_argument('-output_dir', type=str,
        default=os.path.join(os.getenv('PROJTOP'),'user_data/model_data'))

    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the train dataset
    data_frames = [pd.read_csv(os.path.join(opt.input,"%s_pred.csv"%model_name),index_col=0) for model_name in opt.models]
    length = len(data_frames[0])
    train_dfs = [df.iloc[:int(0.8*length)].reset_index(drop=True) for df in data_frames]
    valid_dfs = [df.iloc[int(0.8*length):].reset_index(drop=True) for df in data_frames]
    train_dataset = StackingDataset(train_dfs)
    train_dataloader =  DataLoader(train_dataset,batch_size=256)
    valid_dataset = StackingDataset(valid_dfs)
    valid_dataloader =  DataLoader(valid_dataset,batch_size=256)

    # build the model
    model_config = StackingConfig(len(opt.models),num_calss=opt.n_class)
    m_model = StackingModel(model_config).to(device)

    # build the optimizer
    optimizer = optim.Adam(m_model.parameters(), betas=(0.9, 0.98), eps=1e-05,lr=5.e-2)

    valid_losses = []
    valid_accs = []
    train_losses = []
    train_accs = []
    #train the model
    for epoch_i in range(opt.epoch):
        print('[ Epoch',epoch_i, ']')
        # train
        start = time.time()
        train_loss,train_acc = train_epoch(m_model,train_dataloader,
                optimizer, opt,device)
        print_performances('Training',start,train_loss,train_acc)
        train_losses += [train_loss]
        train_accs += [train_acc]

        # valid
        start = time.time()
        valid_loss,valid_acc = eval_epoch(m_model,valid_dataloader,opt,device)
        print_performances('Validation',start,valid_loss,valid_acc)
        valid_losses += [valid_loss]
        valid_accs += [valid_acc]

        # save the model
        model_name = "Stacking"
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': m_model.state_dict(),'valid_loss':valid_loss,'valid_acc':valid_acc}
        if opt.save_mode == 'all':
            model_name = '{model_name}_loss_{loss:3.3f}.chkpt'.format(model_name,loss=train_loss)
            torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
        elif opt.save_mode == 'best':
            model_name = '{model_name}.chkpt'.format(model_name=model_name)
            if valid_loss <= min(valid_losses) and valid_acc >= max(valid_accs):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

if __name__ == "__main__":
    main()
