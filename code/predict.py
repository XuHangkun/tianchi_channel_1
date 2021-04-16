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
import os
import pandas as pd
from tqdm import tqdm
from model.fastTextModel import FastTextModel
from model.textCNN import TextCNNModel,TextCNNConfig
from model.DPCNN import DPCNNConfig,DPCNNModel
from model.TextRNN import TextRNNConfig,TextRNNModel
from model.TextRCNN import TextRCNNConfig,TextRCNNModel
from model.TextRNN_Att import TextRNNAttConfig,TextRNNAttModel
from model.transformer import TransformerConfig,TransformerModel
from model.bert import BERTConfig,BERTModel
from utils.bert_kFoldData import KFoldDataLoader
from transformers import RobertaTokenizerFast

def load_model(opt,device):
    m_models = []
    for index in range(len(opt.models)):
        # load model setting
        model_weight_path = os.path.join(opt.model_path,'{model_name}.chkpt'.format(model_name=opt.models[index]))
        checkpoint = torch.load(model_weight_path,map_location=device)
        model_setting = checkpoint["settings"]

        if "DPCNN_" in opt.models[index]:
            model_config = DPCNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
            model_config.padding_idx = model_setting.pad_token
            m_model = DPCNNModel(model_config).to(device)
        # TextRNN
        elif "TextRNN_" in opt.models[index]:
            model_config = TextRNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
            model_config.padding_idx = model_setting.pad_token
            m_model = TextRNNModel(model_config).to(device)
        elif "TextRCNN_" in opt.models[index]:
            model_config = TextRCNNConfig(n_vocab=model_setting.ntokens,embedding=model_setting.nemb,max_seq_len=60,num_class=model_setting.nclass)
            model_config.padding_idx = model_setting.pad_token
            m_model = TextRCNNModel(model_config).to(device)
        elif "TextRNNAtt_" in opt.models[index]:
            model_config = TextRNNAttConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
            model_config.padding_idx = model_setting.pad_token
            m_model = TextRNNAttModel(model_config).to(device)
        elif "Transformer_" in opt.models[index]:
            model_config = TransformerConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
            model_config.device = device
            model_config.padding_idx = model_setting.pad_token
            m_model = TransformerModel(model_config).to(device)
        elif  "BERT" in opt.models[index]:
            model_config = BERTConfig(num_class = model_setting.nclass,pre_train_path=model_setting.bert_path)
            m_model = BERTModel(model_config).to(device)
        else:
            # default TextCNN
            model_config = TextCNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
            model_config.padding_idx = model_setting.pad_token
            m_model = TextCNNModel(model_config).to(device)

        m_model.load_state_dict(checkpoint['model'])
        m_model.eval()
        print('[Info] Trained model %s state loaded.'%(opt.models[index]))
        m_models.append(m_model)

    return m_models

def main():
    """
    prediction according to trained model
    """
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('-models',default=["TextCNN_fold1","TextCNN_fold2","TextCNN_fold3","TextCNN_fold4","TextCNN_fold5",
                            "TextRCNN_fold1","TextRCNN_fold2","TextRCNN_fold3","TextRCNN_fold4","TextRCNN_fold5",
                            "DPCNN_fold1","DPCNN_fold2","DPCNN_fold3","DPCNN_fold4","DPCNN_fold5"],
                            nargs="+",help="Net workS for learning")
    parser.add_argument('-model_path', type = str,
                        default=os.path.join(os.getenv('PROJTOP'),'user_data/model_data'),
                        help='Path to model weight file')
    parser.add_argument('-tokenizer_path',default=os.path.join(os.getenv('PROJTOP'),"user_data/bert"),help="path of tokenizer")
    parser.add_argument('-input', type=str,
        default=os.path.join(os.getenv('PROJTOP'),'tcdata/medical_nlp_round1_data/test.csv'),help="path for test.csv")
    parser.add_argument('-output', type = str,
                        default=os.path.join(os.getenv('PROJTOP'),'prediction_result/result.csv'),
                        help='output csv file')
    parser.add_argument('-is_test', action="store_true", default=True, help="do test or validation" )

    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    models = load_model(opt,device)

    # tokenizer for bert model
    tokenizer = RobertaTokenizerFast.from_pretrained(opt.tokenizer_path, max_len=70)
    tokens = [str(i) for i in range(857,-1,-1)]
    tokenizer.add_tokens(tokens)

    # load data
    if opt.is_test:
        results = {"report_ID":[],"Prediction":[]}
    else:
        results = {"report_ID":[],"Prediction":[],"label":[]}
    data = pd.read_csv(opt.input,index_col=0,sep="\|,\|",names=["id","report"])
    print(data)


    # prediction here
    desc = '  - (Prediction)   '
    for index in tqdm(data.index, mininterval=1, desc=desc, leave=False):

        results["report_ID"].append(index)
        if not opt.is_test:
            results["label"].append(data["label"][index])
        preds = []
        for model,model_name in zip(models,opt.models):

            if "BERT" in model_name:
                report = [data["report"][index]]
                report = tokenizer(report,padding=True, truncation=True, return_tensors="pt")
                report.to(device)
            else:
                report = [[int(x) for x in (data["report"][index]).split()]]
                if len(report[0]) > 60:
                    new_report = [[report[0][i] for i in range(60)]]
                    report = new_report
                else:
                    report=[report[0]+[858 for i in range(60-len(report[0]))]]
                report = torch.LongTensor(report).to(device)
                #print(report.shape)

            pred = model(report).squeeze()  # shape [17,1] --> [5]
            preds.append(pred)
        mean_pred = torch.zeros(preds[0].shape).to(device)
        for pred in preds:
            mean_pred += pred/len(preds)

        res = ""
        for value in mean_pred:
            res += "%.8f "%(value)
        # print(res)
        results["Prediction"].append(res.strip())

    #pred_results = pd.DataFrame(results)
    #pred_results.to_csv(opt.output,sep=sep,index=False)
    with open(opt.output,"w") as f:
        for index,report in zip(results["report_ID"],results["Prediction"]):
            f.write("%d|,|%s\n"%(index,report))

if __name__ == '__main__':
    main()
