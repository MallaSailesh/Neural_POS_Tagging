from FFNN import POS_Tagging_FFNN
from conllu import parse

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

    for activation_fnc in ["relu", "tanh"]:
        for hidden_size in [50, 100]:
            for embedding_size in [100, 200]:
                model = POS_Tagging_FFNN(train_data, dev_data, test_data, embedding_size=embedding_size, hidden_dim=hidden_size, activation=activation_fnc, p=2, s=2)
                print(f"Activation Function: {activation_fnc}, Hidden Layer Size: {hidden_size}, Embedding Size: {embedding_size}\n")
                model.train(epochs=5)
    
    print("Best paramters: ")
    print("Activation Function: relu")
    print("Hidden Layer Size: 50")
    print("Embedding Size: 200\n")

    model = POS_Tagging_FFNN(train_data, dev_data, test_data, embedding_size=embedding_size, hidden_dim=hidden_size, activation=activation_fnc, p=2, s=2)
    model.train(epochs=5)
    model.test()