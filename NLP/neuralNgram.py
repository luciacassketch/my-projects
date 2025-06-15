'''
Task:
Use pyTorch to implement the original neural language model of Bengio et al. (2003), a variant of which is also 
summarized in Sections 7.6-7.7 of Jurafsky and Martin (2025). You do not have to follow this paper in detail or 
implement all of their different experiments. It is sufficient to implement the following configuration:

    Use word embeddings (referred to in the paper as “word features”) of size 100;
    Use a hidden layer of size 80, with tanh activation;
    Use a context of the 4 previous words to predict the following one, equivalent to a 5-gram language model;
    Include only words that occur at least 5 times in english-train.txt.gz. Depending on tokenization, this should 
        give you a vocabulary of about 31000 words. All other tokens should be replaced by a special <UNK> token 
        for unknown words;
    As with the n-gram models, use a special <PAD> token to fill out before the start of the sentence in order to 
        always get a 4-word context. Also add a special end-of-sentence <EOS> token at the end of each sentence;
    Use gradient clipping with a threshold of 1.0.

Train for two epochs on english-train.txt.gz, using the Adam optimizer (not available when Bengio et al. (2003) 
created their model - but it makes the training faster) with a learning rate of 0.001 and a batch size of 1024.
Exclude examples where <UNK> is the target word, but leave those where it occurs in the 4-word context. Write the 
embeddings to a file, so that you can easily perform a sanity check.  
'''



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import gzip
from myNLP import tokenize



class Generator():
    '''
    A basic file-reader class to make going through files several times smoother.
    '''
    def __init__(self, path):
        self.path = path
    
    def __iter__(self):
        return self.generator()
    
    def generator(self):
        with gzip.open(self.path, 'rt') as f:
            for line in f:
                line = line.strip()
                yield tokenize(line)
    
    def __len__(self):
        return sum(1 for _ in self.generator())


class NGramDataset(Dataset):
    '''
    A pytorch.Dataset-based dataset. It process the data creating a vocabulary, encoding sentences,
    padding sequences, selecting the n-grams.
    '''
    def __init__(self, file, min_freq, context_size):
        self.file = file
        self.data = []

        counter = Counter()
        for line in file:
            counter.update(line)
        vocab = ["UNK", "PAD", "EOS"] + [word for word,freq in counter.items() if freq > min_freq]
        self.word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}    # sorting to make sure it's reproducible

        for line in self.file:
            sentence = ["PAD"] * context_size + line + ["EOS"]
            encoded = [self.word2idx.get(word, self.word2idx["UNK"]) for word in sentence]

            for i in range(context_size, len(encoded)):
                context = encoded[i - context_size: i]
                target = encoded[i]
                if target != self.word2idx["UNK"]:
                    self.data.append((torch.tensor(context), torch.tensor(target)))        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NeuralLM(nn.Module):
    '''
    A neural n-gram language model created using pytorch's nn.Module. It has an embedding layer,
    a hidden layer with tanh activation, and a linear layer that outputs probability scores for each word.
    '''
    def __init__(self, vocab_size, context_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 100)
        self.fc1 = nn.Linear(context_size * 100, 80)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(80, vocab_size)

    def forward(self, x):
        embeds = self.embed(x)  
        embeds = embeds.view(embeds.size(0), -1)  # reshape to (batch_size, 400) 
        h = self.tanh(self.fc1(embeds))
        out = self.fc2(h)      # applies CrossEntropyLoss
        return out


def train(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for epoch in range(2):
        total_loss = 0
        total_tokens = 0

        for contexts, targets in dataloader:
            contexts, targets = contexts.to(device), targets.to(device)
            outputs = model(contexts)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_tokens += targets.size(0)

        avg_loss = total_loss / total_tokens
        print(f"Epoch {epoch+1}/2, avg loss: {avg_loss:.4f}")

    return model


def extract_embeddings(dataset, model):
    '''
    This function extracts the embeddings created by the model to perform a sanity check and prints
    them to a file of the format:
        31301 100
        word1 0.123 0.432 0.731 ...
        word2 0.791 0.183 -0.221 ...
        ... [more words] ...
    '''
    idx2word = {idx: word for word,idx in dataset.word2idx.items()}

    embeddings = model.embed.weight.data.cpu().numpy()      # get embeddings matrix
    vocab_size, embedding_dim = embeddings.shape            # get its dimensions

    with open("embeddings.txt", "w") as f:
        f.write(f"{vocab_size} {embedding_dim}\n")

        for i in range(vocab_size):
            word = idx2word[i]
            vector = " ".join(f"{x:.6f}" for x in embeddings[i])
            f.write(f"{word} {vector}\n")



def main():

    path = "/path"
    file = Generator(path)

    dataset = NGramDataset(file, 5, 4)
    print("dataset created")
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    print("dataloader created")

    model = NeuralLM(len(dataset.word2idx), 4)
    print("model created")
    model = train(model, dataloader)
    
    extract_embeddings(dataset, model)



if __name__ == "__main__":
    main()



'''
References:
    Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A neuralprobabilistic language model. Journal of 
    Machine Learning Research, 3:1137-1155.
    
    Jurafsky, Daniel and James H. Martin (2025). Speech and Language Processing: An Introduction to Natural 
    Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3rd. Online 
    manuscript released January 12, 2025. URL: https://web.stanford.edu/~jurafsky/slp3/.
'''
