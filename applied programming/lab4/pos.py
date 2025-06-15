import torch
import pyconll
import random
from torch.utils.data import Dataset, DataLoader


class SentencesDataset(Dataset):
    '''
    This is a class built on top of the Dataset class from pytorch to allow for better performance when fed into Dataloader.
    The __init__ method builds a word and a tag vocabulary, mapping each word and tag to an index.
    It also builds the dataset of processed sentences, where every unit is a pair of sentence - tag sequence.
    The __getitem__ method returns a single pair of the processed data as tensors, and it's used by Dataloader.
    The __len__ method is necessary to iterate over the dataset later in the training loop.
    '''
    def __init__(self, data):
        wordMap = {"UNK": 0, "PAD": 1, "NOUN": 2}
        tagMap = {"MULTIWORD": 0}
        processed_data = []
        for sentence in data:
            sentence_words = []
            sentence_tags = []

            for token in sentence:
                word = token.form.lower() if token.form else None  # avoid errors if word is None
                uposTag = token.upos

                ## this snippet would train the model on UNK words as well, supposing they are nouns, 
                ## but the accuracy is slightly lower:

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

                
                if word and uposTag:  # only execute if both word and tag are not None
                    if word not in wordMap:
                        wordMap[word] = len(wordMap)
                    if uposTag not in tagMap:
                        tagMap[uposTag] = len(tagMap)

                    sentence_words.append(wordMap[word])
                    sentence_tags.append(tagMap[uposTag])
                
                if word and not uposTag:  # this is when the italian combined word is split into its subwords ...
                    if word not in wordMap:
                        wordMap[word] = len(wordMap)
                    
                    sentence_words.append(wordMap[word])
                    sentence_tags.append(tagMap["MULTIWORD"]) # ... so the tag for it becomes MULTIWORD.

            if sentence_words and sentence_tags:  # append the sentence and its tags to the dataset
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
    '''
    This function pads all sentences in a batch to the same length.
    For words, the padding value is 1 (see wordMap).
    For tags, the padding value is -10 (to avoid overlapping with MULTIWORD).

    Args: 
        Batch of processed data from trainDataset.
    Returns:
        The same data but padded to the same length.
    '''
    words, tags = zip(*batch)
    
    words_padded = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=1) 
    tags_padded = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-10).long() # .long() for Crossentropyloss

    return words_padded, tags_padded


class LSTMTagger(torch.nn.Module):
    '''
    Bidirectional LSTM (Long Short-Term Memory) model.
    It consists of three layers: 
        Embedding Layer: converts word indices into dense vector representations.
        LSTM Layer: captures sequential patterns in the sentence.
        Fully Connected Layer: outputs POS-tag predictions.
    '''
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(LSTMTagger, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=1) # padding index ensures PAD tokens don't affect training
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_scores = self.fc(lstm_out)
        return tag_scores


def predict_raw_sentence(model, trainDataset, sentence, device):
    '''
    Using the evaluation mode of the pytorch model, it predicts the tags for one sentence from the conllu file.

    Args:
        model: the trained model.
        trainDataset: an instance of the SentenceDataset class needed to retrieve wordMap and tagMap.
        sentence: a single sentence instance from the conllu file.
        device: the device where the model has been trained.
    Returns:
        the sequence of predicted tags converted back from numbers to tags through tagMap.
    '''
    model.eval()
    sentence = [token.form.lower() for token in sentence]

    with torch.no_grad():
        input = torch.tensor([trainDataset.wordMap.get(word.lower(), 0) for word in sentence], dtype=torch.long).unsqueeze(0).to(device)
                            #  if the word does not exist in wordMap, it returns 0, the index for UNK 
        preds = model(input)
        max_preds = list(torch.argmax(preds, dim=2).squeeze())

        return [next((key for key, value in trainDataset.tagMap.items() if value == p)) for p in max_preds]


def accuracy(testbank, model, trainDataset, device):
    '''
    This function calculates the accuracy of the model on the test set.

    Args:
        testbank: the conllu file.
        model: the trained model.
        trainDataset: an instance of the SentenceDataset class to pass to predict_raw_sentence().
        device: the device where the model has been trained to pass to predict_raw_sentence().
    '''
    correct = 0
    total = 0
    for sentence in testbank:
        true_tags = [token.upos for token in sentence]
        pred_tags = predict_raw_sentence(model, trainDataset, sentence, device)
        for true_pred, pred in zip(true_tags, pred_tags):
            correct += (true_pred == pred)
        total += len(true_tags)
    
    print(f"Test Accuracy: {correct / total:.4f}")
    return correct / total



def main():

    ## load data from conllu files

    trainbank = pyconll.load_from_file("it_isdt-ud-train.conllu")
    testbank = pyconll.load_from_file("it_isdt-ud-test.conllu")

    ## process training data and build batches

    trainDataset = SentencesDataset(trainbank)
    trainLoader = DataLoader(trainDataset, batch_size=50, shuffle=True, collate_fn=padding)

    ## build model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTagger(len(trainDataset.wordMap), len(trainDataset.tagMap))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    ## train the model

    for epoch in range(8):
        total_loss = 0
        num_batches = 0
        for words, tags in trainLoader:
            words, tags = words.to(device), tags.to(device)  # to GPU if available
            optimizer.zero_grad()
            preds = model(words).permute(0, 2, 1)  # for CrossEntropyLoss
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() # same as loss.detach().cpu().numpy()[0]
            num_batches += 1

        print(f"Epoch {epoch}, Loss: {total_loss/num_batches}")

    ## test the accuracy

    accuracy(testbank, model, trainDataset, device)


if __name__ == "__main__":
    main()