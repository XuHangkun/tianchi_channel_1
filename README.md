# tc_medical_nlp
Medical imaging report anomaly detection

## set the enviroment
```bash
$ cd /Project/Top/Path
$ source setup.sh
$ cd code
```
## train word vector
```bash
$ python train_word_vector.py
```

## pretrain the BERT
```bash
# 1000 epoch totally, save the model per 50 epoch
$ python bert_pretrain.py
```
## Train the model
```bash
# You have many model choices, like DPCNN,TextCNN,TextRCNN,BERT. You are free to change the model name, for eg, BERT_lalala is also a fine model name.
python train.py -model DPCNN -epoch 8
```
## Predict
```bash
python predict.py -models DPCNN_fold1 TextCNN_fold2 TextRCNN_fold3 BERT_fold4
```
