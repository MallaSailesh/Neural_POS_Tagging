Run the command :- 
1. python3 pos_tagger.py -f for FFNN (feed forward neural network) for predicting POS_Tags of a given sentence 
2. python3 pos_tagger.py -r for RNN (recurrent  neural network) for predicting POS_Tags of a given sentence. 

FFNN_tuning and RNN_tuning is used to check the best hyperparameters required to get better accuracy. It prints the classification report and confusion matrix of dev set of all the hyperparameter configurations and for test set of only best configuration . <br>


FFNN_plots and RNN_plots are used to print the plots of accuracy vs different hyperparameters. <br>

FFNN_200_50_relu_1_1.pt ==> 
- 200 - embedding size
- 50 - hidden neuron size
- relu - activation function after hidden layer
- 1 - p value
- 1 - s value

RNN_200_50_1.pt ==> 
- 200 - embedding size
- 50 - hidden LSTM dimension
- 1 - number of LSTM layers 

Note :- 
- Here as the training set is small , most of the words are unkown or those kind of sentences which you enter may not be there in train set , so the pos_tags predicted can be wrong.  <br>
- What i have done  is simple case of how to do POS_Tagging. But in reality ,  big train set , heavy training, high compute power etc is needed to predict accurate pos_tags.

