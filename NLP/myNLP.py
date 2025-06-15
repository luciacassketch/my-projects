import re
import os
import math


def char_tokenizer(text: str) -> list[str]:
    '''
    A function that tokenizes words into characters.
    '''
    text = text.lower()
    output = []
    for i in range(len(text)):
        output.append(text[i])
    
    return output



def tokenize(text: str) -> list[str]:
    '''
    A function that tokenizes text using some generic Latin orthography.
    '''
    text = text.lower()
    pattern = r'''(?x) # set flag to allow verbose regexps
    (?:[A-Z]\.)+ # abbreviations, e.g. U.S.A.
    | \w+(?:-\w+)* # words with optional internal hyphens
    | [€\$?]\d+(?:\.\d+)?%? # currency, percentages, e.g. $12.40, 82%
    | \.\.\. # ellipsis
    | [][.,;"'?():_`-]''' # these are separate tokens; includes ], [

    return re.findall(pattern,text)



class CategorizedCorpus:
    '''
    A class which can read a collection of texts and their categories. It supports the format used
    for files provided by the university covering sentiment analysis (directory with .txt files containing
    the text and the category) and imdb data (a directory containing two subdirectories - positive and
    negative - each with their texts).
    '''
    def __init__(self, path):
        self.path = path
        self.ss = False     # flag for sentiment sentences directory
        self.imdb = False   # flag for imdb directory
        if not os.path.exists(path):
            return None
        
        if os.path.isdir(path):     # if it contains subdirectories then it's imdb data
            self.imdb = True
        else:
            self.ss = True
        
    def __iter__(self):
        if self.ss:
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip().split("\t")
                    yield line

        elif self.imdb:
            for folder in os.listdir(self.path):
                if folder == "neg":
                    for file in os.listdir(os.path.join(self.path,"neg")):
                        with open(os.path.join(self.path,"neg",file),"r") as f:
                            yield f.readlines() + ["0"]
                elif folder == "pos":
                    for file in os.listdir(os.path.join(self.path,"pos")):
                        with open(os.path.join(self.path,"pos",file),"r") as f:
                            yield f.readlines() + ["1"]



class NGramModel:
    '''
    A class that extracts n-grams from a text and can apply operations to them.
    The __init__() function creates suitable data structures (as attributes in the NGramModel instance) to 
    efficiently work with n-gram models for the given n-parameter, using the smoothing parameter k. 
    The p() function returns the probability of word following after context (which has length n-1).
    The score() function returns the log-probability of the whole sentence. That is, the sum of the (natural)
    logarithm of the probability of each word given its n-1 word context.
    '''

    def __init__(self, n: int, k: float, sentences: list[tuple[str]]):

        self.n = n
        self.k = k

        dic: dict = {}
        uniques = set()

        if self.n == 1:  # unigrams

            for sentence in sentences:

                for word in sentence:
                    uniques.add(word)  # build the set of words (for V)
                    if word in dic:  # build dictionary
                        dic[word] += 1.0
                    else:
                        dic[word] = 1.0

                if "END" in dic:  # add the "END" pad at the end of every sentence
                    dic["END"] += 1
                else:
                    dic["END"] = 1

            uniques.add("END")  # add the "END" to the set as well
            self.dic = dic
            self.N = sum(self.dic.values())
            self.V = len(uniques)

        elif self.n == 2:  # bigrams

            for sentence in sentences:
                sentence = tuple(["PAD"] + list(sentence) + ["END"])  # padding sentence for analysis
                for word in sentence:  # build the set of words (for V)
                    uniques.add(word)

                i = 0
                while i+2 <= len(sentence):  # build the dictionary
                    ngram = sentence[i:i+2]
                    context = sentence[i]

                    if ngram in dic:
                        dic[ngram] += 1.0
                    else:
                        dic[ngram] = 1.0
                    if context in dic:
                        dic[context] += 1.0
                    else:
                        dic[context] = 1.0

                    i += 1

            self.V = len(uniques)
            self.dic = dic

        else:  # trigrams or more

            for sentence in sentences:
                sentence = tuple(["PAD"]*(self.n-1) + list(sentence) + ["END"])  # padding sentence for analysis
                for word in sentence:  # build the set of words (for V)
                    uniques.add(word)

                i = 0  # index start
                j = self.n  # index end
                while j <= len(sentence):
                    ngram = sentence[i:j]
                    context = sentence[i:j-1]

                    if ngram in dic:
                        dic[ngram] += 1.0
                    else:
                        dic[ngram] = 1.0
                    if context in dic:
                        dic[context] += 1.0
                    else:
                        dic[context] = 1.0

                    i += 1
                    j += 1

            self.V = len(uniques)
            self.dic = dic

    def p(self, word: str, context: tuple[str]) -> float:  # calculating probability

        if word not in self.dic:
            self.dic[word] = 0.0

        if self.n == 1:
            return (self.k + self.dic[word])/(self.k*self.V+self.N)

        elif self.n == 2:
            tot_exp = tuple(list(context)+[word])
            context = context[0]  # avoiding type-realted problems (tuple instead of string)
            if context not in self.dic:
                self.dic[context] = 0.0
            if tot_exp not in self.dic:
                self.dic[tot_exp] = 0.0
            return (self.k + self.dic[tot_exp])/(self.k*self.V+self.dic[context])

        else:
            tot_exp = tuple(list(context)+[word])
            if context not in self.dic:
                self.dic[context] = 0.0
            if tot_exp not in self.dic:
                self.dic[tot_exp] = 0.0
            return (self.k + self.dic[tot_exp])/(self.k*self.V+self.dic[context])

    def score(self, sentence: tuple[str]) -> float:  # computing score of a sentence

        log_prob = 0.0

        if self.n == 1:
            sentence = tuple(list(sentence) + ["END"])
            for word in sentence:
                log_prob += math.log(self.p(word, ()))

        else:
            sentence = tuple(["PAD"]*(self.n-1) + list(sentence) + ["END"])
            i = 0
            while i+self.n <= len(sentence):
                ngram = sentence[i:i+self.n]
                log_prob += math.log(self.p(ngram[-1], ngram[:-1]))
                i += 1

        return log_prob



class fileReader():
    '''
    A class that can read Unimorph (Batsuren et al. 2022) files, of the format: 
    "carpmes carpmesarnas    N;GEN;PL;DEF".
    '''
    def __init__(self,path):
        self.path = path
    
    def __iter__(self):
        return self.generator()
    
    def generator(self):    # needed so the generator is rebuilt every iteration
        with open(self.path) as f:
            for line in f:
                line = line.strip().split("\t")
                if len(line) > 2:
                    yield line
    
    def get_line(self, n):   
        count = -1      # so the first line (0) can get chosen
        for line in self.generator():
            count += 1
            if count == n:
                return line

    def __len__(self):
        return sum(1 for _ in self.generator())




'''
References:
Batsuren, Khuyagbaatar et al. (2022). UniMorph 4.0: Universal Morphology. 
    arXiv: 2205.03608 [cs.CL]. URL: https://arxiv.org/abs/2205.03608.
'''