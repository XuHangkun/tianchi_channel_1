import os
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(os.getenv('PROJTOP'),'user_data/bert'), max_len=70)
tokens = [str(i) for i in range(857,-1,-1)]
tokenizer.add_tokens(tokens)
print(tokenizer.vocab["<s>"])
print(tokenizer.vocab["</s>"])
print(tokenizer.vocab["<pad>"])
print(tokenizer.vocab["<mask>"])


from transformers import RobertaConfig,RobertaForMaskedLM
a = "623 656 293 851 636 842 <mask> 493 338 266 369 691 693 380 136 363 399 556 698 66 432 449 177 830 381 332 290 380 26 343  28 177 415 832 14"
b = "693 380 136 363 399 556 698 66 432 449 177 830 381 332 290 380 26 343  28 177 415 832 14"
c = tokenizer([a,b])
print(c)
model = RobertaForMaskedLM.from_pretrained((os.getenv('PROJTOP'),'user_data/bert'))
