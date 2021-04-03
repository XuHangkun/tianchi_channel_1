# tc_medical_nlp
Medical imaging report anomaly detection

## set the enviroment
```bash
$ cd /Proj/Top/Path
$ source setup.sh
$ cd code
```
## train word vector
```bash
$ python train_word_vector.py -word_size 256
```
## Train the model
```bash
# now you are at code directory
python train.py -model DPCNN -epoch 8
```

## Predict
```bash
python predict.py -model DPCNN
```
