import sys
import string
import torch
from FFNN import POS_Tagging_FFNN
from RNN import POS_Tagging_RNN

from conllu import parse

def get_data(type):
    with open(f'./UD_English-Atis/en_atis-ud-{type}.conllu', 'r', encoding='utf-8') as f:
        data = f.read()
    total_data  = parse(data)
    final_data = []
    for sentence in total_data:
        data = []
        for token in sentence:
            data.append([token["id"], token["form"], token["upos"]])
        final_data.append(data)
    return final_data

def clean_text(sentence):
    translator = str.maketrans('', '', string.punctuation)  # Create removal mapping
    sentence = sentence.translate(translator).lower()  # Remove punctuation & lowercase
    tokens = sentence.split()
    return tokens

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python pos_tagger.py <model_flag>")
        sys.exit(1)

    train_data = get_data("train")
    dev_data = get_data("dev")
    test_data = get_data("test")
    
    type = sys.argv[1]
    if type == '-f':
        model = torch.load('FFNN_200_50_relu_1_1.pt')
    elif type == '-r':
        model = torch.load('RNN_200_50_1.pt')
    else:
        print("Usage: model_flag must be -f or -r.")
        sys.exit(1)
    
    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "quit":
            break
        tokens = clean_text(sentence)
        predicted_tags = model.predict(tokens)
        initial_tokens = clean_text(sentence)
        for i in range(len(initial_tokens)):
            print(initial_tokens[i], " ", predicted_tags[i])

