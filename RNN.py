import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders

from sklearn.metrics import classification_report, confusion_matrix

class POS_Tagging_RNN(nn.Module):
  def __init__(self, train_data, dev_data, test_data, embedding_size, hidden_size = 50, num_layers = 1, lr=0.005):
    '''
    train_data :- list of lists where each element has idx, word, pos_tag
    embedding_size :- vector size to represent the word
    '''
    super().__init__()
    self.embedding_size = embedding_size

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
    self.vocab.extend(["<unk>"])
    self.vocab_size = len(self.vocab)
    self.words_dict = {}
    for i in range(self.vocab_size):
      self.words_dict[self.vocab[i]] = i;

    self.word_embeddings = nn.Embedding(self.vocab_size, embedding_size)

    self.train_data = self.get_words_and_tags(train_data)
    self.dev_data = self.get_words_and_tags(dev_data)
    self.test_data = self.get_words_and_tags(test_data)

    self.loss_fnc = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
    self.hidden_to_tag = nn.Linear(hidden_size, self.num_of_tags)


  def get_words_and_tags(self, data):
    extracted_data = []
    for sent in data:
      modified_data = []
      for token in sent:
        word_map = self.words_dict["<unk>"] if token[1] not in self.words_dict.keys() else self.words_dict[token[1]]
        tag_map = self.tags_dict["<UNK>"] if token[2] not in self.tags_dict.keys() else self.tags_dict[token[2]]
        modified_data.append((word_map, tag_map))
      extracted_data.append(modified_data)
    return extracted_data


  def forward(self, x):
    embeddings = self.word_embeddings(x)
    lstm_out, _ = self.lstm(embeddings.view(len(x), 1, -1))
    tag_space = self.hidden_to_tag(lstm_out.view(len(x), -1))
    return tag_space


  def get_accuracy(self, dataLoader):
    with torch.no_grad():
      correct = 0
      total = 0
      for data in dataLoader:
        inputs = torch.tensor([x[0] for x in data])
        labels = torch.tensor([x[1] for x in data])
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      accuracy = 100 * correct / total
    return accuracy


  def print_metrics(self, dataLoader):
    all_labels = []
    all_predicted = []
    for data in dataLoader:
        inputs = torch.tensor([x[0] for x in data])
        labels = torch.tensor([x[1] for x in data])
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels)
        all_predicted.extend(predicted.numpy())
    print(classification_report(all_labels, all_predicted, target_names=self.tags_dict.keys(), zero_division=0))
    print(confusion_matrix(all_labels, all_predicted))
    print("\n\n\n\n\n")


  def train(self, epochs=10):
    train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=1, shuffle=True)
    for epoch in range(epochs):
      running_loss = 0.0
      for data in train_loader:
        inputs = torch.tensor([x[0] for x in data])
        labels = torch.tensor([x[1] for x in data])
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss_fnc(outputs, labels)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
      print(f"Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}")
    val_accuracy = self.get_accuracy(torch.utils.data.DataLoader(self.dev_data, batch_size=1, shuffle=False))
    print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")
    print("\nValidation Metrics\n")
    self.print_metrics(torch.utils.data.DataLoader(self.dev_data, batch_size=1, shuffle=False))
    return val_accuracy


  def test(self):
    test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1, shuffle=False)
    test_accuracy = self.get_accuracy(test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("\nTest Metrics\n")
    self.print_metrics(torch.utils.data.DataLoader(self.dev_data, batch_size=1, shuffle=False))


  def predict(self, tokens):
    
    for i in range(len(tokens)):
        try:
            tokens[i] = self.words_dict[tokens[i]]
        except:
            tokens[i] = self.words_dict["<unk>"]
    inputs = torch.tensor([ (tokens[i:i+1]) for i in range(len(tokens)) ])
    outputs = self.forward(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted_tags = predicted.numpy().tolist()
    return [self.unique_tags[predicted_tag] for predicted_tag in predicted_tags]    
