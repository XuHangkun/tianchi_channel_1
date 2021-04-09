from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os
import torch
import pandas as pd
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Read train.csv and save the report
reports_path = os.path.join(os.getenv('PROJTOP'),'user_data/Bert/reports.txt')
reports_f = open(reports_path,"w")
train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/medical_nlp_round1_data/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
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
tokenizer.save_model(os.path.join(os.getenv('PROJTOP'),'user_data/Bert'))

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(os.getenv('PROJTOP'),'user_data/Bert'), max_len=70)
tokens = [str(i) for i in range(857,-1,-1)]
tokenizer.add_tokens(tokens)
print(tokenizer.vocab)
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
    output_dir=os.path.join(os.getenv('PROJTOP'),'user_data/Bert'),
    overwrite_output_dir=True,
    num_train_epochs=1000,
    per_device_train_batch_size=128,
    save_steps=3950,
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
