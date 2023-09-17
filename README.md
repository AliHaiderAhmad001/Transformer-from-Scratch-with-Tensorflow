# Attention is All You Need - Transformer Model for Machine Translation

This repository contains an implementation of the Transformer model, as described in the paper "Attention is All You Need" by Vaswani et al., for the task of machine translation from English to French.

## Overview

The Transformer model is a state-of-the-art neural network architecture designed for sequence-to-sequence tasks, such as machine translation. It utilizes the self-attention mechanism to capture long-range dependencies and improve parallelization, making it highly efficient and effective for processing sequential data.

## Features

- Implementation of the Transformer model from scratch in TensorFlow 2.
- Training the model on a dataset of English-French sentence pairs.
- Custom positional encodings, attention mechanisms and etc.
- Learning rate schedule with warmup for training.
- Monitoring and early stopping using custom callbacks.
- Inferance.

## Credits
The implementation of the Transformer model is based on the paper "Attention is All You Need" by Vaswani et al. The code in this repository is developed by Ali Haidar Ahmad and is for educational purposes.

## Requirements

- Python 3.10.6
- TensorFlow 2.12.0

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/AliHaiderAhmad001/Neural-Machine-Translator.git
cd Neural-Machine-Translator
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Project Structure
```
├── config.py               # Configuration class for model hyperparameters
├── data_processing.py      # Functions for data processing and normalization
├── encoder.py              # Encoder classes and positional encodings
├── decoder.py              # Decoder classes with attention mechanisms
├── positional_embeddings.py# Implementation for positional_embeddings layer and SinusoidalPositionalEncoding
├── embeddings.py           # Embeddings layer for inputs
├── attention.py            # Attention head and multi-head attention mechanisms
├── feed_forward.py         # Feed-forward layer in the transformer
├── lr_schedule.py          # Custom learning rate schedule with warmup
├── load_model.py           # to load the model
├── requirements.txt        # List of required dependencies
├── train.py                # Script to train the Transformer model
├── inferance.py            # Script to translate English sentences to French
├── loss_functions.py       # Loss functions for the model (CCE)
├── metrics.py              # Masked accuracy and BLUE
├── transformer_callbacks.py# Custom monitoring and early stopping using.
├── Demo/README.md          # Project documentation
├── tmp                     # In it we put all the files resulting from the processing and training processes, including the final model.
├── dataset/fra.txt         # The dataset
└── README.md
```

## Usage

1. Prepare the Data:
* Place your English-French sentence pairs in a text file, with each pair separated by a tab character.
* Normalize, Vectorize and make tf.data format using the data_processing.py script.

2. Train the Model:
* Set the hyperparameters in the Config class in the config.py script.
* Create the Transformer model and custom learning rate schedule using the Transformer and LrSchedule classes in the model.py script.
* Compile the model with the appropriate loss function and optimizer.
* Train the model using the fit method and monitor the training progress.

3. Translate Sentences:
* Load the trained model using the load_model function in the load_model.py script.
* Use the trained model to generate translations for English sentences using the translate function in the model.py script.

## Contributing

If you find some bug or typo, please let me know or fix it and push it to be analyzed.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/AliHaiderAhmad001/Neural-Machine-Translator/blob/main/LICENSE.txt) file for details.

