from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import argparse
from utils.EDA import RandomDelete,RandomSwap
import numpy as np

def easy_data_augmentation(texts,eda_alpha=0.1,n_aug=4):
    """
    Data Enhancement, randomly delete partial words or swap the words
    For evergy sentence, we need to change eda_alpha*sentence_len words.
    """
    def concat_words(words):
        sentence = ""
        for word in words:
            sentence += "%s "%(word)
        return sentence

    enhanced_texts = []
    if n_aug == 0:
        return
    for i in range(len(texts)):
        true_aug = 0
        if n_aug >1:
            true_aug = int(n_aug)
        elif n_aug >= 0:
            if np.random.random() < n_aug:
                true_aug = 1
        for j in range(true_aug):
            # randomly delete some words
            enhanced_texts.append(concat_words(RandomDelete(texts[i].split(),eda_alpha)))
            # randomly swap some words
            enhanced_texts.append(concat_words(RandomSwap(texts[i].split(),eda_alpha)))
    texts += enhanced_texts
    # randomly break up the data
    for i in range(len(texts)):
        text_1_index = int(np.random.random()*len(texts))
        text_2_index = int(np.random.random()*len(texts))
        x = texts[text_1_index]
        texts[text_1_index] = texts[text_2_index]
        texts[text_2_index] = x

    return texts


# Define the
parser = argparse.ArgumentParser(description="Bert Pretrain")
parser.add_argument('-epoch',type=int,default=100,help="epoch")
parser.add_argument('-hidden_size',type=int,default=768,help="hidden size")
parser.add_argument('-lr',type=float,default=7.e-4,help="hidden size")
parser.add_argument('-eda_alpha',type=float,default=0.15,help="hidden size")
parser.add_argument('-n_aug',type=float,default=4.0,help="hidden size")
parser.add_argument('-batch_size',type=int,default=1024,help="epoch")
parser.add_argument('-corpus_dir',default=os.path.join(os.getenv('PROJTOP'),'tcdata'),help="dir of corpus")
parser.add_argument('-corpus_file',nargs="+",default=["track1_round1_train_20210222.csv","track1_round1_testA_20210222.csv","track1_round1_testB.csv","train.csv"],help="file of corpus")
parser.add_argument('-out_dir',default=os.path.join(os.getenv('PROJTOP'),'user_data/bert'),help="out dir of tokenizer and pretrained model")
parser.add_argument('-not_do_eda',action="store_true",help="out dir of tokenizer and pretrained model")
parser.add_argument('-pre_train_path',default=os.path.join(os.getenv('PROJTOP'),"user_data/bert"),help="path of pretrained BERT")
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate the corpus
# Read train.csv and save the report
reports_path = os.path.join(args.out_dir,'reports.txt')
all_reports = []
for corpus_file in args.corpus_file:
    if "test" in corpus_file :
        train_df = pd.read_csv(os.path.join(args.corpus_dir,corpus_file),sep="\|,\|",names=["id","report"],index_col=0)
    elif "train" in corpus_file:
        train_df = pd.read_csv(os.path.join(args.corpus_dir,corpus_file),sep="\|,\|",names=["id","report","label"],index_col=0)

    for i in range(len(train_df)):
        report = train_df["report"][i]
        all_reports.append(report)
# Do data angumentation here
if args.not_do_eda:
    pass
else:
    all_reports = easy_data_augmentation(all_reports,args.eda_alpha,args.n_aug)

# write the data
reports_f = open(reports_path,"w")
for report in all_reports:
    reports_f.write("%s\n"%(report))
reports_f.close()

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
tokens = [str(i) for i in range(857,-1,-1)]
tokenizer.add_tokens(tokens)
# Customize training
#tokenizer.train(files=reports_path,vocab_size=658+5, min_frequency=2)
tokenizer.save_model(args.out_dir)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(args.out_dir, max_len=100)
tokens = [str(i) for i in range(857,-1,-1)]
tokenizer.add_tokens(tokens)
#print(tokenizer.vocab)
vocab_size = len(tokenizer.vocab)
print("<pad>:%d,<mask>:%d,<s>:%d,</s>:%d,<unk>:%d"%(tokenizer.vocab["<pad>"],
    tokenizer.vocab["<mask>"],
    tokenizer.vocab["<s>"],
    tokenizer.vocab["</s>"],
    tokenizer.vocab["<unk>"]))
print("total tokens: %d"%(vocab_size))

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=reports_path,
    block_size=args.batch_size
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import RobertaConfig,RobertaForMaskedLM
from transformers import AutoConfig,AutoModel

config = RobertaConfig(
    vocab_size=vocab_size,
    hidden_size=args.hidden_size,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
    pad_token_id=tokenizer.vocab["<pad>"],
    bos_token_id=tokenizer.vocab["<s>"],
    eos_token_id=tokenizer.vocab["</s>"]
)

if args.pre_train_path:
    model = RobertaForMaskedLM.from_pretrained(args.pre_train_path).to(device)
else:
    model = RobertaForMaskedLM(config=config).to(device)
print(model)

from transformers import Trainer, TrainingArguments
#counts = 1
training_args = TrainingArguments(
    output_dir=args.out_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    save_steps=1000,
    save_total_limit=10,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(args.out_dir)
