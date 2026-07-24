# Multilayer Perceptron (MLP) on MNIST - in C
_Made by Colin Cauderlier (intra 42: ccauderl)_

[![CodeFactor](https://www.codefactor.io/repository/github/adrir42/mnist-perceptron/badge/main)](https://www.codefactor.io/repository/github/adrir42/mnist-perceptron/overview/main)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4d51dae29e4646dc89b3b1501dcf4ba8)](https://app.codacy.com/gh/Adrir42/MNIST-Perceptron/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Quality gate status](https://sonarcloud.io/api/project_badges/measure?project=Adrir42_MNIST-Perceptron&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Adrir42_MNIST-Perceptron)

From scratch implementation, in C, of a multilayer perceptron (MLP) for handwritten digit recognition on the MNIST dataset. No deep learning library is used: the forward pass, the backpropagation and the gradient descent are all hand-coded.
This is my first project of a neural network, so this README also shows what I understood about machine learning so far.

## Result
**~ 95% accuracy** on the test set (10 000 images), after training on the 60 000 images of the training set.

## Architecture

```
Input (784 = 28×28 pixels) → Hidden layer (128 neurons, ReLU) → Output layer (10 neurons, Softmax)
```
- **Hidden layer**: 784 → 128, ReLU activation
- **Output layer**: 128 → 10, Softmax activation
- **Loss function**: Cross-entropy
- **Optimization**: Stochastic gradient descent (SGD)
- **Weight initialization**: He for the hidden layer, Xavier/Glorot for the output layer

## Requirements

- GCC compiler
- The MNIST dataset in IDX format (see Installation section)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Adrir42/MNIST-Perceptron.git
```

### 2. Download the MNIST dataset

The dataset is **not included** in this repository (see the Dataset License section below). Download the 4 files from the official source (see at the end):

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

Decompress them (`gunzip`) and place them in the directory expected by the program (You can change it in the ocr.h file).

### 3. Compile

```bash
make all
```

### 4. Run

```bash
./perceptron
```

## Example output

```
Epoch 1 : 54515/60000 (90.86)
Epoch 2 : 56323/60000 (93.87)
Epoch 3 : 56966/60000 (94.94)
Epoch 4 : 57145/60000 (95.24)
Epoch 5 : 57429/60000 (95.71)
...
Epoch 16 : 58351/60000 (97.25)
Epoch 17 : 58416/60000 (97.36)
Epoch 18 : 58471/60000 (97.45)
Epoch 19 : 58559/60000 (97.60)
Epoch 20 : 58476/60000 (97.46)
TEST: 9592/10000 (95.92)
```
## How it works

### Overview
The program:
1. Loads and parses the MNIST files in binary IDX format
2. Initializes the network's weights and biases
3. Trains the network over several epochs (forward pass → loss computation → backward pass → weight update)
4. Evaluates the final accuracy on the test set

### How the neural network works

A MLP (Multilayer Perceptron) learns to map an input image to a class (here, a handwritten digit from 0 to 9) by progressively adjusting thousands of numerical weights, based on already-labeled examples. Learning happens by a four-step cycle on each image: forward pass->loss computation->backward pass (=backpropagation)->weight update.

### 1. Forward pass

Each image (28*28 pixels, normalized between 0 and 1) id=s fed into the network as a vector of 784 values.

- **Hidden layer (784 → 128)**: each neuron computes a linear regression : a weighted sum of all input pixels, plus a bias. It then applies the **ReLU** activation function (`max(0, x)`), which introduces the non-linearity needed for the network to learn complex patterns.
- **Output layer (128 → 10)**: each neuron in turn computes a weighted sum of the 128 hidden layer activations like before. The **Softmax** function then turns these 10 raw scores into probabilities (summing to 1), representing the network's confidence for each of the 10 possible digits.
### 2. Loss computation
The **cross-entropy** loss measures the gap between the network's prediction and the true label: it is close to 0 when the network assigns a high probability to the correct digit, and grows very large when it is confidently wrong. This is the valie that training seeks to minimize and on which is based the final accuracy.
### 3. Backward pass
Starting from the error made at the output, the network computes, while moving back toward the input layer by layer, how much each weight contributed to that error: this is the gradient. This computation relies on the chain rule, and reuses the intermediate values computed during the forward pass.
### 4. Weight update
Each weight is corrected in the direction that reduces the loss, proportionnally to its gradient and a learning rate. The learning rate is a hard-coded constant, based on similar projects and personnal testing. In this project, my learning rate is 0.05, which is high (just like my learning rate 😉)

```
weight = weight - learning_rate × gradient
```
This cycle is repeated for each of the 60 000 training images, over several epochs (full passes over the dataset), until the network reaches a satisfactory accuracy.

What is impressing with machine learning is that no recognition rule has ever been explicitly programmed, **everything is learned directly from the data.**

## Project structure

| File | Role |
|---|---|
| `main.c` | Main training and evaluation loop |
| `fds.c` | Opening/closing data files and parsing IDX headers |
| `init.c` | Network weight and bias initialization |
| `forward_pass.c` | Forward propagation (dense + ReLU + softmax) |
| `backward_pass.c` | Gradient backpropagation |
| `display.c` / `display_files.c` | Display and debugging functions |
| `ocr.h` | Data structures and declarations |

## AI usage: 100% humans code

Claude AI has been used for this project as a step-by-step tutorial, it provided me guidance on the mathematical notions and the general structure of the program. **No code has ever been generated for this project, **

## License

This project is distributed under the MIT license.

## MNIST dataset

The MNIST dataset is **not included** in this repository and must be downloaded separately from [the official source](http://yann.lecun.com/exdb/mnist/).

Yann LeCun and Corinna Cortes hold the copyright of the MNIST dataset, which is a derivative work from the original NIST datasets. The dataset is made available under the terms of the **Creative Commons Attribution-Share Alike 3.0** license.

Here is another source if the first one is not working:
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## Acknowledgements

- Yann LeCun, Corinna Cortes, and Christopher J.C. Burges for creating and making the MNIST dataset available
    - Yann Lecun: https://github.com/ylecun, https://www.researchgate.net/profile/Yann-Lecun-2
    - Corinna Cortes : https://www.researchgate.net/profile/Corinna-Cortes
    - Christopher J.C. Burges : https://www.researchgate.net/profile/Christopher-Burges

- My brother Albin Cauderlier for his welcomed support and advices.