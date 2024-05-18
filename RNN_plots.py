from RNN import POS_Tagging_RNN
from conllu import parse
import matplotlib.pyplot as plt 

def get_data(type):
    with open(f'./UD_English-Atis/en_atis-ud-{type}.conllu', 'r', encoding='utf-8') as f:
        data = f.read()
    total_data = parse(data)
    final_data = []
    for sentence in total_data:
        data = []
        for token in sentence:
            data.append([token["id"], token["form"], token["upos"]])
        final_data.append(data)
    return final_data

if __name__ == "__main__":

    train_data = get_data("train")
    dev_data = get_data("dev")
    test_data = get_data("test")

    for hidden_size in [50, 100]:
        for embedding_size in [100, 200]:
            for num_layers in [1, 2]:
                accuracies = []
                for epcohs in [2, 4]:
                    model = POS_Tagging_RNN(train_data, dev_data, test_data, embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
                    accuracies.append(model.train(epochs=epcohs))
                plt.plot([2, 4], accuracies, marker='o', linestyle='-')
                plt.xlabel("Number of training epochs")
                plt.ylabel("Accuracies")
                plt.title("Accuracy change with change in number of epochs")
                plt.show()
    
