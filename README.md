# **tc_medical_nlp**
Medical imaging report anomaly detection

## **解决方案**
### 数据预处理
- 报告长度处理。对于TextRCNN和DPCNN,在将报告输入到模型之前，我们会将报告处理成长度为60,多裁少补。BERT模型不采取这个处理
- 数据增广。我们采取了两种操作来增广数据。随机对调一个报告中的两个词或者随机删去单词。对于一个长度为L的句子，我们会改变其L*eda_alpha个单词。对于一个报告来说，我们会用每种方法变化出n_aug个报告。如果n_aug是属于(0,1),那就按照概率决定要不要对这个报告去增广

### 模型
我们的基本方法是用五折交叉训练三种神经网络。然后对15个模型的预测结果取平均。
- BERT。BERT需要进行MLM预训练。我们每50个epoch保存预训练模型，最后采用loss为0.64左右的预训练模型。在进行我们的任务时，在BERT的pooler_output后面接全连接输出后进行训练。最终我们取epoch=13的模型去做预测。
- TextRCNN。具体参数设置可以见code/model/TextRCNN.py。TextRCNN训练30个epoch。
- DPCNN。具体参数设置可以见code/model/DPCNN.py。DPCNN训练7-8个epoch左右

### 基本参数设置
- batch size 128
- 优化器采用Adam, BERT训练的learning rate(lr) = 1.e-5。TextRCNN和DPCNN训练learning rate(lr)=1.e-3。具体如下
```python
optimizer = optim.Adam(m_model.parameters(),lr=opt.lr,betas=(0.9, 0.98), eps=1e-05)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=1.e-6,  last_epoch=-1)
```
- BERT和TextRCNN不采取数据增广(eda_alpha=0,n_aug=0)。DPCNN采取数据增广(eda_alpha=0.1,n_aug=0.5)

## **训练模型**
### 设置环境变量
```bash
# 环境变量比较重要，要在medical_nlp_project项目的根目录下面设置
$ cd /Project/Top/Path
$ source setup.sh
# 此时要切换到code目录下
$ cd code
```
### 预训练BERT模型
```bash
# 1000 epoch totally, save the model per 50 epoch
$ python bert_pretrain.py
```
### 训练三个模型
```bash
# You have many model choices, like DPCNN,TextCNN,TextRCNN,BERT. You are free to change the model name, for eg, BERT_lalala is also a fine model name.
python train.py -model DPCNN -epoch 8
source train.sh
```

## **预测**
```bash
source run.sh
```

## **系统依赖**
- 操作系统
```bash
LSB Version:    :core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch
Distributor ID: CentOS
Description:    CentOS Linux release 7.8.2003 (Core)
Release:        7.8.2003
Codename:       Core
```
- python 3.6