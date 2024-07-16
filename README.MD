# NEURAL MACHINE TRANSLATION

This project focus principally on Neural Machine translation. It aim to compare various implementaion from scratch, the architecture used are:
* Vanilla RNN
* GRU
* LSTM
* GRU + Attention
* GRU + Transformer?

## Depemdencies

* Python 3
* Pytorch
* Torcheval
* Numpy

## Folder structure
```{shell}
.
├── data --> Datasets
│   ├── train --> Training data
│   │   ├── raw_source.txt
│   │   ├── raw_target.txt
│   │   ├── norm  --> Normalized data
│   │   │   ├── norm_source.txt
│   │   │   └── norm_target.txt
│   └── val --> Validation data
│   ├── train
│   │   ├── raw_source.txt
│   │   ├── raw_target.txt
│   │   ├── norm --> Normalized Data
│   │   │   ├── norm_source.txt
│   │   │   └── norm_target.txt
└── src --> Modules
│   ├── GRU.py
│   ├── LSTM.py
│   ├── NeuralMachineTranslation.py
│   ├── Normalizer.py
│   ├── RNN.py
│   ├── Tokenizer.py
│   └── utils.py
├── GRU.ipynb
├── LSTM.ipynb
├── README.MD
├── RNN.ipynb
├── config.json
```
