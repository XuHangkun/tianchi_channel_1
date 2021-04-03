# -*- coding: utf-8 -*-
"""
    predict according to trained model
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

import torch
import argparse
import torchtext
import os
import pandas as pd
from tqdm import tqdm
from model.transformer import TransformerModel
from model.fastTextModel import FastTextModel
from model.textCNN import TextCNNModel,TextCNNConfig
from model.DPCNN import DPCNNConfig,DPCNNModel
from model.TextRNN import TextRNNConfig,TextRNNModel
from model.TextRCNN import TextRCNNConfig,TextRCNNModel
from model.TextRNN_Att import TextRNNAttConfig,TextRNNAttModel
from model.Stacking import StackingConfig,StackingModel

def load_model(opt,device):
    m_models = []
    for index in range(len(opt.models)):
        # load model setting
        a_classes = []
        for i in range(opt.fold_k):
            model_weight_path = os.path.join(opt.model_path,'{model_name}_fold{fold_k}.chkpt'.format(model_name=opt.models[index],fold_k=i+1))
            checkpoint = torch.load(model_weight_path,map_location=device)
            model_setting = checkpoint["settings"]

            if "DPCNN" in opt.models[index]:
                model_config = DPCNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
                model_config.padding_idx = model_setting.pad_token
                m_model = DPCNNModel(model_config).to(device)
            # TextRNN
            elif "TextRNN" in opt.models[index]:
                model_config = TextRNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
                model_config.padding_idx = model_setting.pad_token
                m_model = TextRNNModel(model_config).to(device)
            elif "TextRCNN" in opt.models[index]:
                model_config = TextRCNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
                model_config.padding_idx = model_setting.pad_token
                m_model = TextRCNNModel(model_config).to(device)
            elif "TextRNNAtt" in opt.models[index]:
                model_config = TextRNNAttConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
                model_config.padding_idx = model_setting.pad_token
                m_model = TextRNNAttModel(model_config).to(device)
            else:
                # default TextCNN
                model_config = TextCNNConfig(model_setting.ntokens,model_setting.nemb,model_setting.nclass)
                model_config.padding_idx = model_setting.pad_token
                m_model = TextCNNModel(model_config).to(device)

            m_model.load_state_dict(checkpoint['model'])
            m_model.eval()
            print('[Info] Trained model %s state loaded.'%(opt.models[index]))
            a_classes.append(m_model)
        m_models.append(a_classes)

    return m_models

def main():
    """
    prediction according to trained model
    """
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('-models',default=["TextCNN","TextRCNN","DPCNN"],
                            nargs="+",help="Net workS for learning")
    parser.add_argument('-model_path', type = str,
                        default=os.path.join(os.getenv('PROJTOP'),'user_data/model_data'),
                        help='Path to model weight file')
    parser.add_argument('-input', type = str,
                        default=os.path.join(os.getenv('PROJTOP'),'user_data/tmp_data/test.csv'),
                        help='input csv file')
    parser.add_argument('-fold_k',type=int,default=5,help="number of fold")
    parser.add_argument('-output', type = str,
                        default=os.path.join(os.getenv('PROJTOP'),'prediction_result/result.csv'),
                        help='output csv file')
    parser.add_argument('-is_test', action="store_true", default=True, help="do test or validation" )

    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    models = load_model(opt,device)
    staching_config = StackingConfig(len(models))
    staching_model = StackingModel(staching_config).to(device)
    stack_model_weight_path = os.path.join(opt.model_path,'Stacking.chkpt')
    stack_checkpoint = torch.load(stack_model_weight_path,map_location=device)
    staching_model.load_state_dict(stack_checkpoint['model'])

    # load data
    if opt.is_test:
        names = ['id','report']
        results = {"report_ID":[],"Prediction":[]}
        sep = ","
    else:
        name = ['id','label']
        results = {"report_ID":[],"Prediction":[],"label":[]}
        sep=","
    data = pd.read_csv(opt.input,index_col=0)
    print(data)


    # prediction here
    desc = '  - (Prediction)   '
    for index in tqdm(data.index, mininterval=1, desc=desc, leave=False):
        report = [[int(x) for x in (data["report"][index]).split()]]
        if len(report[0]) > 70:
            new_report = [[report[0][i] for i in range(70)]]
            report = new_report
        else:
            report=[report[0]+[858 for i in range(70-len(report[0]))]]
        report = torch.LongTensor(report).to(device)
        #print(report.shape)

        results["report_ID"].append(index)
        if not opt.is_test:
            results["label"].append(data["label"][index])
        preds = []
        for type_models in models:
            total_preds = []
            for model in type_models:
                pred = model(report).squeeze()  # shape [17,1] --> [17]
                total_preds.append(pred)
            preds.append(sum(total_preds)/len(total_preds))
        # item in preds should be [17]
        stack_input = torch.stack(preds,-1).to(device)
        mean_pred = staching_model(stack_input)
        res = ""
        for value in mean_pred:
            res += "%.8f "%(value)
        # print(res)
        results["Prediction"].append(res.strip())

    pred_results = pd.DataFrame(results)
    pred_results.to_csv(opt.output,sep=sep,index=False)

if __name__ == '__main__':
    main()