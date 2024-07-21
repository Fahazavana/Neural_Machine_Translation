# Neural Machine Translation

## Project Description

This project focuses on translating English to Afrikaans using various neural network architectures. 

* Vanilla Architectures:
  * Recurrent Neural Networks (RNN)
  * Gated Recurrent Unit (GRU)
  * Long Short-Term Memory (LSTM)
* Architectures with Scaled Dot-Product Attention:
  * RNN with Attention
  * GRU with Attention
  * LSTM with Attention

These models are trained on a educational corpus from Stellenbosch University and augmented with data from the Tatoeba project. The project also explores different decoding strategies, such as greedy search and beam search, to optimize translation quality.

## Dependencies

* Python 3
* PyTorch
* NumPy
* torchinfo
* evaluate
* tqdm

Ensure all dependencies are installed before running the project.

## Folder Structure and description

```
.
├── data
│   ├── train
│   │   ├── norm
│   │   │   ├── afrikaans.txt
│   │   │   └── english.txt
│   │   ├── Sentence pairs in English-Afrikaans - 2024-07-16.tsv
│   │   ├── *.af.txt
│   │   └── *.en.txt
│   └── val
│       ├── norm
│       │   ├── *afrikaans.txt
│       │   └── *english.txt
│       ├── *.af.txt
│       └── *.en.txt
├── report
│   ├── figures
│   ├── *.pdf
│   └── *.tex
├── src
│   └── *.py
├── *.ipynb
├── *.json
├── *.npy
└── *.MD
```

- **data/**: Contains the datasets used for training and validation.
  - **train/**: Training data.
    - **norm/**: Normalized training data.
    - Raw training data
  - **val/**: Validation data.
    - **norm/**: Normalized validation data.
    - Raw validation data

- **report/**: Contains the report documents and figures.
  - **figures/**: Figures used in the report.
  - **content.tex**: LaTeX content file for the report.
  - **main.pdf**: Compiled PDF of the report.
  - **main.tex**: Main LaTeX file for the report.

- **src/**: Source code for the project.
  - **LSTM.py**: Implementation of the LSTM model.
  - **NeuralMachineTranslation.py**: Main script for neural machine translation.
  - **Normalizer.py**: Script for data normalization.
  - **RNN_GRU.py**: Implementation of RNN and GRU models.
  - **RNN_GRUAttention.py**: Implementation of RNN and GRU models with attention.
  - **Tokenizer.py**: Script for tokenizing input data.
  - **Translator.py**: Script for translation without attention.
  - **TranslatorAtt.py**: Script for translation with attention.
  - **utils.py**: Utility functions.

- **00-Data_Prepartion.ipynb**: Jupyter Notebook for preparing, cleaning and normalizing data.
- **01-RNN.ipynb**: Jupyter Notebook for RNN experiments.
- **02-GRU.ipynb**: Jupyter Notebook for GRU experiments.
- **03-LSTM.ipynb**: Jupyter Notebook for LSTM experiments.
- **04-RNNAttention.ipynb**: Jupyter Notebook for RNN with attention experiments.
- **05-GRUAttention.ipynb**: Jupyter Notebook for GRU with attention experiments.
- **06-LSTMAttention.ipynb**: Jupyter Notebook for LSTM with attention experiments.
- **07-Fine-tuning.ipynb**: Jupyter Notebook for Fine tunning experiments (This notebook have been runned on Kaggle).
- **08-Loss_visualization.ipynb**: Jupyter Notebook for Vilsualizing all the loss during training.

- **README.md**: This README file.
- **config.json**: Configuration file for the project.
- **config_val_sun_only.json**: Configuration file for validation using the SUN dataset only.

## Usage

1. **Data Preparation**: Ensure the datasets are correctly placed in the `data` directory, and run ```00-Data_Preparation.ipynb```.
2. **Training**: Use the provided Jupyter Notebooks (e.g., `01-RNN.ipynb`, `02-GRU.ipynb`) to train the models.
3. **Evaluation**: Conatained in each notebook.

**PS** There might be some icomabatibilty with ```cuda``` as all the code was tested pricipally on Apple Silicon M3 (but it sould work). The choice of the device (```mps, cuda, cpu```) is automatically selected based on the ``os` and ```GPU``` availability.

## References

* Herman Kamper, NLP817, [https://www.kamperh.com/nlp817]()
* Aladdin Persson, Pytorch Seq2Seq Tutorial for Machine Translation, [https://www.youtube.com/watch?v=EoGUlvhRYpk]()
* Hugging Face [https://huggingface.co/]()
* Tatoeba [https://tatoeba.org/en]()
* Open Parallel Corpora [https://opus.nlpl.eu/]()
* PyTorch: AdamW [https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html]()
* Google Sentencepiece [https://github.com/google/sentencepiece]()
* Helsinki NLP [https://blogs.helsinki.fi/language-technology/](), [https://huggingface.co/Helsinki-NLP]()
* Junczys-Dowmunt, M., Grundkiewicz, R., Dwojak, T., Hoang, H., Heafield, K., Neckermann, T., Seide, F., Germann, U., Aji, A.F., Bogoychev, N. and Martins, A.F., 2018. Marian: Fast neural machine translation in C++. \textit{arXiv preprint arXiv:1804.00344.}
* Jörg Tiedemann and Santhosh Thottingal. 2020. OPUS-MT - Building open translation services for the World. \textit{In Proceedings of the 22nd Annual Conference of the European Association for Machine Translation, pages 479–480, Lisboa, Portugal. European Association for Machine Translation.}