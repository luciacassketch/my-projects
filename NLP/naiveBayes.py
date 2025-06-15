import myNLP
import math
import os

class NaiveBayesDataset():
    '''
    A bag-of-words Naive Bayes text classifier based on equations (4.10) and (4.14) in Jurafsky and Martin (2025, Chapter 4).
    '''
    def __init__(self, ccTrain, ccTest):
        self.train = ccTrain    # training data
        self.test = ccTest      # testing data
        self.V = set()          # vocabulary
        self.Vpos = {}          # positive words and their count
        self.Vneg = {}          # negative words and their count
        self.N = 0              # tot number of words
        self.Npos = 0           # tot number of positive words
        self.Nneg = 0           # tot number of negative words

        for text, cat in ccTrain:
            text = set(myNLP.tokenize(text.lower()))
            self.N += 1
            for word in text:
                self.V.add(word)
                if cat == "0":
                    self.Nneg += 1
                    if word in self.Vneg:
                        self.Vneg[word] += 1
                    else:
                        self.Vneg[word] = 1
                elif cat == "1":
                    self.Npos += 1
                    if word in self.Vpos:
                        self.Vpos[word] += 1
                    else:
                        self.Vpos[word] = 1
    
    def predict(self, text):
        text = set(myNLP.tokenize(text.lower()))
        negSomma = 0
        posSomma = 0
        negPc = math.log(self.Nneg / self.N)
        posPc = math.log(self.Npos / self.N)

        for word in text:
            negSomma += math.log((self.Vneg.get(word,0)+1) / (sum(self.Vneg.values()) + len(self.V)))
            posSomma += math.log((self.Vpos.get(word,0)+1) / (sum(self.Vpos.values()) + len(self.V)))
        
        negativeText = negPc + negSomma
        positiveText = posPc + posSomma

        if negativeText > positiveText:
            pred = "0"
        else:
            pred = "1"
        
        return pred

    def accuracy(self):
        correct = 0
        tot = 0
        for text,cat in self.test:
            correct += (self.predict(text) == cat)
            tot += 1
        
        return correct/tot
            


def main():
    path = "/..."
    names = ["amazon", "imdb", "yelp"]

    for name in names:
        for file in os.listdir(path):
            if file.startswith(name):
                if file.split(".")[0].endswith("train"):
                    ccTrain = myNLP.CategorizedCorpus(os.path.join(path,file))
                if file.split(".")[0].endswith("test"):
                    ccTest = myNLP.CategorizedCorpus(os.path.join(path,file))
        nc = NaiveBayesDataset(ccTrain,ccTest)
        print(f"{name}: {nc.accuracy()}")


if __name__ == "__main__":
    main()




'''
References:
Jurafsky, Daniel and James H. Martin (2025). Speech and Language Processing: An Introduction to Natural 
    Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3rd. Online 
    manuscript released January 12, 2025. URL: https://web.stanford.edu/~jurafsky/slp3/.
'''