from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import argparse

# Define the
parser = argparse.ArgumentParser(description="Bert Pretrain")
parser.add_argument('-epoch',type=int,default=100,help="epoch")
parser.add_argument('-batch_size',type=int,default=128,help="epoch")
parser.add_argument('-corpus_dir',default=os.path.join(os.getenv('PROJTOP'),'tcdata'),help="dir of corpus")
parser.add_argument('-out_dir',default=os.path.join(os.getenv('PROJTOP'),'user_data/bert'),help="out dir of tokenizer and pretrained model")
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate the corpus
# Read train.csv and save the report
reports_path = os.path.join(os.getenv('PROJTOP'),'user_data/bert/reports.txt')
reports_f = open(reports_path,"w")
corpus_input = ["track1_round1_train_20210222.csv","track1_round1_testA_20210222.csv","track1_round1_testB.csv","train.csv"]
corpus_input_tag = [0,1,1,0]
for corpus_file,tag in zip(corpus_input,corpus_input_tag):
    if tag:
        train_df = pd.read_csv(os.path.join(args.corpus_dir,corpus_file),sep="\|,\|",names=["id","report"],index_col=0)
    else:
        train_df = pd.read_csv(os.path.join(args.corpus_dir,corpus_file),sep="\|,\|",names=["id","report","label"],index_col=0)

    for i in range(len(train_df)):
        report = train_df["report"][i]
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
tokenizer = RobertaTokenizerFast.from_pretrained(args.out_dir, max_len=70)
tokens = [str(i) for i in range(857,-1,-1)]
tokenizer.add_tokens(tokens)
#print(tokenizer.vocab)
vocab_size = len(tokenizer.vocab)

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=reports_path,
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import RobertaConfig,RobertaForMaskedLM

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config).to(device)

from transformers import Trainer, TrainingArguments
#counts = 1
training_args = TrainingArguments(
    output_dir=args.out_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    save_steps=5600,
    save_total_limit=30,
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
