# Neural Machine Translation

## Project Description

This project focuses on translating English to Afrikaans using various neural network architectures. The primary aim is to compare the performance of different models, including:

* Vanilla Architectures:
  * Recurrent Neural Networks (RNN)
  * Gated Recurrent Unit (GRU)
  * Long Short-Term Memory (LSTM)
* Architectures with Scaled Dot-Product Attention:
  * RNN with Attention
  * GRU with Attention
  * LSTM with Attention

These models are trained on a private dataset containing an Engineering Assessment corpus from Stellenbosch University and augmented with data from the Tatoeba project. The project also explores different decoding strategies, such as greedy search and beam search, to optimize translation quality.

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
  - **figures/**: Visual aids and figures used in the report.
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

- **README.md**: This README file.
- **config.json**: Configuration file for the project.
- **config_val_sun_only.json**: Configuration file for validation using the SUN dataset only.

## Usage

1. **Data Preparation**: Ensure the datasets are correctly placed in the `data` directory.
2. **Training**: Use the provided Jupyter Notebooks (e.g., `00-RNN.ipynb`, `01-GRU.ipynb`) to train the models.
3. **Evaluation**: Evaluate the models using the validation data and compare performance.
4. **Report Generation**: Use the LaTeX files in the `report` directory to generate the final report.

## License


## Acknowledgments

- Stellenbosch University for the Engineering Assessment corpus.
- Tatoeba project for additional data.