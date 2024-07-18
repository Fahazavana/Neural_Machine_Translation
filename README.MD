# NEURAL MACHINE TRANSLATION

This project focusu primaly in translating English to Afrikaans. It also aim to compare the following arhitecture:

* Vanilla:
	* RNN
	* GRU
	* LSTM
* With Scaled Dot product Attention:
	* RNN
	* GRU
	* LSTM

These model are tarined on a private dataset conatining Engeinnering Assement corpus (Stellenboch Univeristy) and augmentend with some data from tatoeba projects.


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
