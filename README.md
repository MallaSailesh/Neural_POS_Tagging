# POS Tagging with Feed Forward and Recurrent Neural Networks

This project involves designing and implementing models for Part-of-Speech (POS) Tagging using Feed Forward Neural Networks (FFNN) and Recurrent Neural Networks (RNN). 

## 1. Models

### 1.1 Feed Forward Neural Network POS Tagging

The FFNN model takes embeddings for the token, along with `p` previous and `s` successive tokens as input, where `p` and `s` are fixed for a given network, and outputs the POS tag.

**Example:**
For the sentence "An apple a day keeps the doctor away", to get the POS tag for the word "a" with `p = 2` and `s = 3`, the network would take the embeddings for ["An", "apple", "a", "day", "keeps", "the"] and output the POS tag "DET".

### 1.2 Recurrent Neural Network POS Tagging

The RNN model (Vanilla RNN, LSTM, or GRU) takes embeddings for all tokens in a sentence and outputs the corresponding POS tags in sequence.

**Example:**
For the sentence "An apple a day keeps the doctor away", the model takes the embeddings for ["An", "apple", "a", "day", "keeps", "the", "doctor", "away"] and outputs the POS tags for all the words in the sentence ["DET", "NOUN", "DET", "NOUN", "VERB", "DET", "NOUN", "ADV"].

## 2. Hyperparameter Tuning

### 2.1 Feed Forward Neural Network POS Tagger

Experimented with different hyperparameters like hidden layer sizes, embedding sizes, and activation functions. 

- Reported dev set evaluation metrics for three different configurations.
- Reported test set evaluation metrics for the best performing configuration on the dev set.
- Provided graphs for context_window ∈ {0...4} vs dev set accuracy, where `p = s = context_window` for one such configuration.

### 2.2 Recurrent Neural Network POS Tagger

Experimented with different hyperparameters like number of stacks, hidden state sizes, and embedding sizes. 

- Reported dev set evaluation metrics for three different configurations.
- Reported test set evaluation metrics for the best performing configuration on the dev set.
- Provided epoch vs dev set accuracy graphs for the three configurations being evaluated.

## 3. Evaluation Metrics

Used various metrics like accuracy, recall, and F1 scores to evaluate the POS Taggers on the dev and test sets. Generated Confusion Matrices for both sets to better evaluate tag-based performance.

## 4. Dataset

The files are located at `ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-{train,dev,test}.conllu`. Only the first, second, and fourth columns (word index, lowercase word, and POS tag) were used. The `conllu` library was used for parsing these files.

**Note:** The UD dataset does not include punctuation. Generally, when working with other datasets, punctuation may be filtered out before tagging.

## 5. Running the Models

Use the following commands to run the models:

- For Feed Forward Neural Network:
  ```sh
  python3 pos_tagger.py -f
  ```
- For Recurrent Neural Network:
  ```
  python3 pos_tagger.py -r
  ```



## 6. Tuning and Plots

`FFNN_tuning` and `RNN_tuning` are used to check the best hyperparameters for better accuracy. They print the classification report and confusion matrix of the dev set for all configurations and for the test set of the best configuration.

`FFNN_plots` and `RNN_plots` are used to print accuracy vs different hyperparameter plots.

## 7. Model File Naming Conventions

- `FFNN_200_50_relu_1_1.pt`:
  - 200: embedding size
  - 50: hidden neuron size
  - relu: activation function after hidden layer
  - 1: `p` value
  - 1: `s` value

- `RNN_200_50_1.pt`:
  - 200: embedding size
  - 50: hidden LSTM dimension
  - 1: number of LSTM layers

## Note

The training set is small, so many words or sentences may not be present in the training set, leading to incorrect POS tag predictions. This project is a simple demonstration of POS Tagging. In reality, larger training sets, heavy training, and high compute power are required for accurate POS tagging.
