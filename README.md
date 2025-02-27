# Word2Vec CBOW Implementation

## Overview

This project implements the **Continuous Bag of Words (CBOW)** variant of **Word2Vec** using **Python and NumPy**. The primary objective is to learn word embeddings that capture semantic relationships between words, making them useful for Natural Language Processing (NLP) tasks.

## Features

- Implements CBOW from scratch without deep learning libraries
- Trains word embeddings using **stochastic gradient descent (SGD)**
- Includes **negative log-likelihood loss** and **softmax output layer**
- Visualizes word embeddings with **PCA, t-SNE, and UMAP**
- Requires only **NumPy and Matplotlib**

## How CBOW Works

CBOW predicts a **target word** based on its **surrounding context words**. It follows these steps:

1. **Input**: Takes context words around a target word.
2. **Embedding Layer**: Maps words to dense vector representations.
3. **Concatenation**: Combines context word embeddings into a single vector.
4. **Linear Layer**: Projects the context vector to the vocabulary space.
5. **Softmax Layer**: Converts logits into probabilities.
6. **Loss Function**: Uses **negative log-likelihood loss** for optimization.
7. **Backpropagation**: Updates embeddings via gradient descent.

## Installation

This project requires **Python 3.7+** and a few dependencies.

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Model

```
python main.py
```

## Implementation Details

### Neural Network Layers

- **Linear Layer**: Implements a simple matrix multiplication (`XW`).
- **Embedding Layer**: Converts word indices to dense vectors.
- **Softmax Layer**: Converts scores into probabilities.

### Loss Function

- Uses **Negative Log Likelihood (NLL)** loss

### Optimizer

- Implements **Stochastic Gradient Descent (SGD)** for weight updates:
  $$
  \theta_{new} = \theta_{old} - \eta \frac{\partial L}{\partial \theta}
  $$

### Training Pipeline

- Reads input text, tokenizes words, and builds a vocabulary.
- Constructs **context-target** training pairs.
- Trains the CBOW model over multiple epochs.
- Saves learned word embeddings.

## How to Use

Clone the repository and train your own **CBOW Word2Vec model** from scratch:

```
git clone https://github.com/your-username/word2vec-cbow.git
cd word2vec-cbow
python main.py
```
