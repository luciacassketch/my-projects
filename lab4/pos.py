import torch
import pyconll
import random
from torch.utils.data import Dataset, DataLoader

trainbank = pyconll.load_from_file("it_isdt-ud-train.conllu")
testbank = pyconll.load_from_file("it_isdt-ud-test.conllu")



class SentencesDataset(Dataset):
    def __init__(self, data):
        wordMap = {"UNK": 0, "PAD": 1, "NOUN": 2}
        tagMap = {"MULTIWORD": 0}
        processed_data = []
        for sentence in data:
            sentence_words = []
            sentence_tags = []

            for token in sentence:
                word = token.form.lower() if token.form else None
                uposTag = token.upos

                # if word and uposTag:
                #     if random.random() < 0.1 and uposTag == "NOUN":
                #         sentence_words.append(wordMap["UNK"])
                #         sentence_tags.append(tagMap["NOUN"])
                #     else:
                #         if word not in wordMap:
                #             wordMap[word] = len(wordMap)
                #         if uposTag not in tagMap:
                #             tagMap[uposTag] = len(tagMap)

                #         sentence_words.append(wordMap[word])
                #         sentence_tags.append(tagMap[uposTag])

                ## this snippet would train the model on UNK words as well, supposing they are nouns, 
                ## but the accuracy is slightely lower
                
                if word and uposTag:
                    if word not in wordMap:
                        wordMap[word] = len(wordMap)
                    if uposTag not in tagMap:
                        tagMap[uposTag] = len(tagMap)

                    sentence_words.append(wordMap[word])
                    sentence_tags.append(tagMap[uposTag])
                
                if word and not uposTag:
                    if word not in wordMap:
                        wordMap[word] = len(wordMap)
                    
                    sentence_words.append(wordMap[word])
                    sentence_tags.append(tagMap["MULTIWORD"])

            if sentence_words and sentence_tags:
                processed_data.append((sentence_words, sentence_tags))
            
        self.wordMap = wordMap
        self.tagMap = tagMap
        self.processed_data = processed_data
    
    def __getitem__(self, index):
        words = self.processed_data[index][0]
        tags = self.processed_data[index][1]
        return torch.tensor(words, dtype=torch.long), torch.tensor(tags, dtype=torch.long)
    
    def __len__(self):
        return len(self.processed_data)


def padding(batch):
    words, tags = zip(*batch)
    
    words_padded = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=1) 
    tags_padded = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-10).long() # -10 to not overlap with MULTIWORD  
                                                                                                    # .log() for Crossentropyloss

    return words_padded, tags_padded


trainDataset = SentencesDataset(trainbank)
trainLoader = DataLoader(trainDataset, batch_size=50, shuffle=True, collate_fn=padding)

class LSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(LSTMTagger, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_scores = self.fc(lstm_out)
        return tag_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTagger(len(trainDataset.wordMap), len(trainDataset.tagMap))
model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


for epoch in range(10):
    total_loss = 0
    for words, tags in trainLoader:
        words, tags = words.to(device), tags.to(device)  # to GPU if available
        optimizer.zero_grad()
        preds = model(words).permute(0, 2, 1)  # for CrossEntropyLoss
        loss = criterion(preds, tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # same as loss.detach().cpu().numpy()[0]

    print(f"Epoch {epoch}, Loss: {total_loss}")


def predict_raw_sentence(model, sentence):
    model.eval()
    sentence = [token.form.lower() for token in sentence]

    with torch.no_grad():
        input = torch.tensor([trainDataset.wordMap.get(word.lower(), 0) for word in sentence], dtype=torch.long).unsqueeze(0).to(device)
        preds = model(input)
        max_preds = list(torch.argmax(preds, dim=2).squeeze())

        return [next((key for key, value in trainDataset.tagMap.items() if value == p)) for p in max_preds]


def accuracy(testbank):
    correct = 0
    total = 0
    for sentence in testbank:
        true_tags = [token.upos for token in sentence]
        pred_tags = predict_raw_sentence(model, sentence)
        for true_pred, pred in zip(true_tags, pred_tags):
            correct += (true_pred == pred)
        total += len(true_tags)
    
    print(f"Test Accuracy: {correct / total:.4f}")
    return correct / total

accuracy(testbank)



