import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rnn import rnn,lstm,gru
from preprocessing import corpus
import multiprocessing
from tqdm import tqdm

class rnn_model(nn.Module):
    def __init__(self, vocab_size, hidden_dim,device):
        super(rnn_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = rnn(hidden_dim)
        self.fc = nn.Linear(hidden_dim,2)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        encoded = self.encoder(input)
        hidden = torch.zeros(input.size(1), self.hidden_dim).to(self.device)
        output, hidden = self.rnn(encoded, hidden)
        output = self.fc(hidden)
        return output
class lstm_model(nn.Module):
    def __init__(self, vocab_size, hidden_dim,device):
        super(lstm_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = lstm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 100)
        self.fc = nn.Linear(100,2)
        self.device = device
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, input):
        encoded = self.encoder(input)
        hidden = torch.zeros(input.size(1), self.hidden_dim).to(self.device)
        cell = torch.zeros(input.size(1), self.hidden_dim).to(self.device)
        _, hidden = self.lstm(encoded, hidden,cell)
        hidden = self.dropout(hidden)
        decode = self.relu(self.decoder(hidden))
        output = self.fc(decode)
        return output

class gru_model(nn.Module):
    def __init__(self, vocab_size, hidden_dim,device):
        super(gru_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.gru = gru(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 100)
        self.fc = nn.Linear(100,2)
        self.device = device
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, input):
        encoded = self.encoder(input)
        hidden = torch.zeros(input.size(1), self.hidden_dim).to(self.device)
        _, hidden = self.gru(encoded, hidden)
        hidden = self.dropout(hidden)
        decode = self.relu(self.decoder(hidden))
        output = self.fc(decode)
        return output

def collate_fn(batched_samples):
    batch_size = len(batched_samples)
    ### 아래에 코드 빈칸을 완성해주세요
    batched_samples = sorted(batched_samples, key=lambda x:len(x[0]), reverse=True) # 0번째 요소의 길이를 기준으로 내림차순 정렬
    src_sentences = []
    tgt_sentences = []
    for src_sentence, tgt_sentence in batched_samples:
        src_sentences.append(torch.tensor(src_sentence))
        tgt_sentences.append(torch.tensor(tgt_sentence))

    src_sentences = torch.nn.utils.rnn.pad_sequence(src_sentences, batch_first=True) # batch x longest seuqence 순으로 정렬 (링크 참고)

    tgt_sentences = torch.tensor(tgt_sentences)
    assert src_sentences.shape[0] == batch_size and tgt_sentences.shape[0] == batch_size
    return src_sentences.type(torch.LongTensor), tgt_sentences

def train():
    import wandb
    wandb.init(project="rnn",
               config={
                   "learning_rate": 0.001,
                   "dropout": 0.2,
                   "architecture": "gru",
               })
    epochs = 5
    batch_size = 16
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda')
    loss_fn = nn.CrossEntropyLoss()
    corpus_dataset = corpus()
    train,valid = corpus_dataset.split_dataset(0.2)
    model = gru_model(len(corpus_dataset),300,device)
    optim = torch.optim.Adam(params=model.parameters(),lr=0.001)
    train_loader = DataLoader(train,batch_size=batch_size,num_workers=multiprocessing.cpu_count()//2
                        ,pin_memory=use_cuda,collate_fn=collate_fn,shuffle=True)
    valid_loader = DataLoader(valid,batch_size=batch_size,num_workers=multiprocessing.cpu_count()//2
                        ,pin_memory=use_cuda,collate_fn=collate_fn)
    model.cuda()
    best_valid_acc = 0
    for epoch in range(epochs):
        loss_value = 0
        matches = 0
        model.train()
        all = len(train)
        for idx, (data,label) in enumerate(train_loader):
            data = data.to(device)
            data = torch.transpose(data,1,0)
            label = label.to(device)

            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output,label)
            loss.backward()
            optim.step()
            loss_value += loss.item()

            preds = torch.argmax(output, dim=-1)
            matches += (preds == label).sum().item()
            if idx % 100 == 0:
                print(epoch,(batch_size*(idx+1))/all,matches/(batch_size*(idx+1)), 'loss', loss.item())
        model.eval()
        all = len(valid)
        with torch.no_grad():
            matches = 0
            loss_value = 0
            for data,label in tqdm(valid_loader):
                data = data.to(device)
                data = torch.transpose(data,1,0)
                label = label.to(device)
                output = model(data)
                loss = loss_fn(output, label)
                preds = torch.argmax(output, dim=-1)
                matches += (preds == label).sum().item()
                loss_value += loss.item()
            if matches/all > best_valid_acc:
                best_valid_acc = matches/all
                torch.save(model.state_dict(), 'gru_model.pt')
        wandb.log({"acc":  matches/all, "loss": (loss_value*batch_size)/all})
        print(f'{epoch}epochs acc: {matches/all} loss:{(loss_value*batch_size)/all}')

    torch.save(model,'model.pt')
if __name__ == "__main__":
    train()