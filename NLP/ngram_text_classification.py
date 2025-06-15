import myNLP
from collections import defaultdict

class NgramTextClassfier():
    '''A class that performs text classification using one n-gram language model for each category.
    The class is intended to be general to handle two apparently different problems (and preferably others, 
    of similar structure): language classification using a character-based n-gram model, and sentiment analysis 
    using a word-based n-gram model. The data is supposed to be of the format "text    cat".
    '''
    def __init__(self, path, char=False):
        self.data = myNLP.CategorizedCorpus(path)   # training data
        self.char = char    # for choice bw word-based and char-based n-grams
        self.categories = set()     # we will build a model for each category
        self.categorized_texts = defaultdict(list)  # dict to map each category to its texts

        ## preprocess training data
        for text,cat in self.data:
            self.categories.add(cat)
        for text,cat in self.data:
            if self.char:
                text = tuple(myNLP.char_tokenizer(text)) # char-based tokenizer if flagged
            else:
                text = tuple(myNLP.tokenize(text))
            self.categorized_texts[cat].append(text)
    
    def train(self,n,k):
        self.models = {}
        ## train an n-gram model for every category
        for cat in self.categories:
            self.models[cat] = myNLP.NGramModel(n, k, self.categorized_texts[cat])
    
    def test(self, test_path):
        tot = 0
        correct = 0
        self.test = myNLP.CategorizedCorpus(test_path)  # test file

        ## prepare test data
        for text,cat in self.test:
            tot += 1
            if self.char:
                text = tuple(myNLP.char_tokenizer(text))
            else:
                text = tuple(myNLP.tokenize(text))

            ## test every model in the object and extract the one that had best results for every text (which equals to
            ## predicting the category of the text)
            scores = []
            for model in self.models:
                scores.append((model, self.models[model].score(text)))
            
            best = max(scores, key=lambda x: x[1])
            correct += (best[0] == cat)
            
        accuracy = correct/tot
        return accuracy



c = NgramTextClassfier("/path", char=True)
c.train(3,0.1)
print(c.test("/path"))

c = NgramTextClassfier("/path")
c.train(2,0.1)
print(c.test("/path"))