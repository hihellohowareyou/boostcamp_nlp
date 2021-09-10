import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, random_split
from konlpy.tag import Mecab

class corpus(Dataset):
    def __init__(self,test= False):
        train = 'nsmc/nsmc/ratings_train.csv'
        test = 'nsmc/nsmc/ratings_test.txt'
        if test:
            train = pd.read_table(test)
        else:
            train = pd.read_csv(train)
        self.train_data = train['document']
        self.label = train['label']
        self.word2idx = {}
        self.ids2word = []
        self.m = Mecab()
        for idx,sentence in enumerate(self.train_data):
            try:
                words = self.m.morphs(sentence)
                for word in words:
                    try:
                        self.word2idx[word]
                    except:
                        self.word2idx[word] = len(self.ids2word)+1
                        self.ids2word.append(word)
            except:
                self.train_data.drop(idx,inplace=True)
                print(f'이상한 문장 {idx}')
        print('전처리가 끝났어요')
        self.m = Mecab()
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return self.pregrograss(self.train_data[idx]),self.label[idx]

    def pregrograss(self,sentence):
        sentence = self.m.morphs(sentence)
        word_tensor = []
        for word in sentence:
            word_tensor.append(self.word2idx[word])
        return torch.Tensor(word_tensor)

    def split_dataset(self,val_ratio):
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

# class vocab:
#     def __init__(self):
#         train = 'nsmc/nsmc/ratings_train.csv'
#         train = pd.read_csv(train)
#         self.train_data = train['document']
#
#     def make_vocab(self):
#         for sentence in self.train_data:
#             words = self.m.morphs(sentence)
#             for word in words:
#                 try:
#                     self.word2idx[word]
#                 except:
#                     self.word2idx[word] = len(self.ids2word)+1
#                     self.ids2word.append(word)
#         return
#
# class rnn_model(nn.Module):
#     def __init__(self, vocab_size, hidden_dim):
#         super(rnn_model, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.encoder = nn.Embedding(vocab_size, hidden_dim)
#         self.rnn = rnn(hidden_dim)
#         self.decoder = nn.Linear(hidden_dim, 1)
#
#     def forward(self, input, batch_size):
#         encoded = self.encoder(input)
#         hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
#         output, hidden = self.rnn(encoded, hidden)
#         decode = self.decoder(hidden)
#         return decode
#
# class nsmcDataset(Dataset):
#     def __init__(self,path):
#         train = pd.read_csv(path)
#         self.train_data = train['document']
#         self.label = train.['label']
#         self.tokenizer = Mecab()
#
#     def tokenize(self,sentence):
#
"""
간단한 전처리
코퍼스 만들기  
init train, train.data train.label
tokenize(train_data)  tokenize -- for line in lines: <sos>+tokenize(lines)+<eos>
데이터로더 
train 
model = rnn_model(1,300)
loss_fn = nn.BCELoss()
for data,label in dataloader:
    data = data.to(device)
    label = label.to(divice)
    output = model(data)
    loss = loss_fn(output,label)
    loss.backward()
    optim.step()
"""