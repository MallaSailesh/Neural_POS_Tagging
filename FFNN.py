import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders

from sklearn.metrics import classification_report, confusion_matrix

class POS_Tagging_FFNN(nn.Module):
  def __init__(self, train_data, dev_data, test_data, embedding_size, hidden_dim = 50, activation="relu" , p=2, s=2, lr=0.005, batch_size=32):
    '''
    train_data :- list of lists where each element has idx, word, pos_tag
    embedding_size :- vector size to represent the word
    p :- number of previous taken we take while predicting the tag for the current word.
    s :- number of successive token we take while predicting the tag for the current word.
    p + s >= 1 because if we dont know anything about previous and successive tokens it very tough to classify
    '''
    super().__init__()
    self.input_size = (p+s+1)*embedding_size
    self.p = p
    self.s = s
    self.embedding_size = embedding_size
    self.batch_size=batch_size

    self.all_tags = [sub_data[2] for data in train_data for sub_data in data]
    self.all_words = [sub_data[1] for data in train_data for sub_data in data]

    self.unique_tags = list(set(self.all_tags))
    self.unique_tags.append("<UNK>")
    self.num_of_tags = len(self.unique_tags)
    # tags_dict is used for getting the index of the tag
    self.tags_dict = {}
    for i in range(self.num_of_tags):
      self.tags_dict[self.unique_tags[i]] = i;

    self.vocab = list(set(self.all_words))
    self.vocab.extend(["<s>", "<e>", "<unk>"])
    self.vocab_size = len(self.vocab)
    self.words_dict = {}
    for i in range(self.vocab_size):
      self.words_dict[self.vocab[i]] = i;

    self.word_embeddings = nn.Embedding(self.vocab_size, embedding_size)

    self.train_data = self.data_process(self.get_words_and_tags(train_data))
    self.dev_data = self.data_process(self.get_words_and_tags(dev_data))
    self.test_data = self.data_process(self.get_words_and_tags(test_data))

    self.loss_fnc = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    self.main = nn.Sequential(
        nn.Linear(self.input_size, hidden_dim),
        nn.ReLU() if activation == 'relu' else nn.Tanh(),
        nn.Linear(hidden_dim, self.num_of_tags)
    )


  def get_words_and_tags(self, data):
    extracted_data = []
    for sent in data:
      modified_data = []
      modified_data.extend([(self.words_dict["<s>"], self.tags_dict["<UNK>"])]*(self.p))
      for token in sent:
        word_map = self.words_dict["<unk>"] if token[1] not in self.words_dict.keys() else self.words_dict[token[1]]
        tag_map = self.tags_dict["<UNK>"] if token[2] not in self.tags_dict.keys() else self.tags_dict[token[2]]
        modified_data.append((word_map, tag_map))
      modified_data.extend([(self.words_dict["<e>"], self.tags_dict["<UNK>"])]*(self.s))
      extracted_data.append(modified_data)
    return extracted_data


  def data_process(self, data):
    X = []
    Y = []
    for tokens in data:
        for j in range(self.p, len(tokens) - self.s):
            word_idxs = torch.tensor([tokens[i][0] for i in range(j - self.p, j + self.s + 1)])
            X.append(word_idxs)
            Y.append(tokens[j][1])
    # apply one hot encoding to Y
    Y = torch.tensor(Y)
    X = torch.stack(X)
    return torch.utils.data.TensorDataset(X, Y)


  def forward(self, x):
    x = self.word_embeddings(x)
    x = x.view(x.size(0), -1) # Now shape is (p+s+1)*embedding_size
    return self.main(x)


  def get_accuracy(self, dataLoader):
    # self.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for inputs, labels in dataLoader:
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      accuracy = 100 * correct / total
    return accuracy


  def print_metrics(self, dataLoader):
    all_labels = []
    all_predicted = []
    for inputs, labels in dataLoader:
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels)
        all_predicted.extend(predicted.numpy())
    print(classification_report(all_labels, all_predicted, target_names=self.tags_dict.keys(), zero_division=0))
    print(confusion_matrix(all_labels, all_predicted))
    print("\n\n\n\n\n")


  def train(self, epochs=10):
    train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    for epoch in range(epochs):
      running_loss = 0.0
      for inputs, labels in train_loader:
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss_fnc(outputs, labels)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
      print(f"Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}")
    val_accuracy = self.get_accuracy(torch.utils.data.DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False))
    print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")
    print("\nValidation Metrics\n")
    self.print_metrics(torch.utils.data.DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False))
    return val_accuracy


  def test(self):
    test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    test_accuracy = self.get_accuracy(test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("\nTest Metrics\n")
    self.print_metrics(torch.utils.data.DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False))


  def predict(self, tokens):
    
    for i in range(len(tokens)):
        try:
            tokens[i] = self.words_dict[tokens[i]]
        except:
            tokens[i] = self.words_dict["<unk>"]
    for i in range(self.p):
        tokens.insert(0, self.words_dict["<s>"])
    for i in range(self.s):
        tokens.append(self.words_dict["<e>"])

    inputs = [torch.tensor(tokens[i:i+self.p+self.s+1]) for i in range(len(tokens)-self.p-self.s)]
    inputs = torch.stack(inputs)
    outputs = self.forward(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted_tags = predicted.numpy().tolist()
    return [self.unique_tags[predicted_tag] for predicted_tag in predicted_tags]    
