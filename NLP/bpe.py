'''
Task: 
Write a Byte Pair Encoding (BPE) tokenizer, as described in Jurafsky and Martin (2025, Section 2.5.2).  
The tokenizer could be implemented as a class, with two main methods:
1.  A method that takes a path to a text corpus and an integer indicating the desired vocabulary size, 
    and uses the BPE algorithm to compute such a vocabulary.
2.  A method that reads a text file and produces a tokenized text, using
    the BPE vocabulary produced by the previous method. You should also add functionality
    to save/load the BPE vocabulary to/from a file, in order to avoid having to run the (slow) 
    BPE algorithm more than once.
'''

import os


class BPE():
    def __init__(self,corpus,text,k):
        self.corpus = corpus  # the training text to build the vocab
        self.text = text  # the text to tokenize
        self.V = set()  # inizializing the vocab
        self.k = k  # desired length of the vocab

        with open(corpus, "r") as f:  # building base vocab with all letters from the corpus
            for line in f:
                line = line.strip().lower().split()
                for word in line:
                    for letter in word:
                        self.V.add(letter)
    

    def comboMaker(self, combinations, words):
        '''
        This function finds all combinations of characters in a line in the corpus.

        Args: 
            the combinations dict to update with new combos and counts for this line;
            the line, already split into characters from V by bpeTokenizer().
        
        Returns: the updated combinations dict.
        '''
        for word in words:
            for i in range(len(word)-1):
                substr = word[i]+word[i+1]
                if substr in combinations:
                    combinations[substr] += 1
                else:
                    combinations[substr] = 1

        return combinations
    

    def bpeTokenizer(self,words,V):
        '''
        This functions takes a line in the corpus and splits it according to the characters in V.

        Args: 
            words, the line split into words;
            V, the vocab to use to split the words. Necessary because it will be self.V in buildV(), 
                but a new V in tokenize_text().

        Returns: the updated list of words split according to the V. 
        '''
        for i in range(len(words)): 
            new_split = [] # initializing empty list to store the new split
            remaining_string = words[i] # initializing the string with the word

            while (len(remaining_string) != 0): # until the string is empty, so all letters have been recognised and subtracted
                found = False
                for comb in V:
                    if remaining_string.startswith(comb): # finding the comb in V that matches the beginning of the string
                        found = True
                        new_split.append(comb) # appending it to the new split
                        remaining_string = remaining_string.replace(comb,"",1) # subtracting it from the string

                        break # this ensures the loop starts from the beginning of the list at the next iteration
                              # which is needed for the longest combinations to be recognised first
                    
                if not found and remaining_string: # only for cases where the character isn't in the vocab
                    new_split.append(remaining_string[0])
                    remaining_string = remaining_string[1:]
            
            if new_split: 
                words[i] = new_split

        return words
    

    def buildV(self):
        '''
        This function uses bpeTokenizer() and comboMaker() to build the vocabulary. 
        It takes every line in the corpus, tokenizes it according to self.V, finds all possible combinations
        with this new split and updates the combinations dict. Then it retrieves the most common combination,
        it updates self.V accordingly, and casts it back to a list sorting it to make sure that, when 
        tokenizing the words, the longest characters are checked first. This avoids splitting the words into 
        single characters when they should be split into combinations of characters instead.
        It repeats this until the vocabulary reaches the length indicated by the user.

        Args: None, everything is saved as an argument of the BPE object.

        Returns: the updated self.V.
        '''
        while len(self.V) <= self.k-1: # until the vocab is as long as indicated by the user
            combinations = {} # initializing the combinations dict
            with open(self.corpus, "r") as f:
                for line in f:
                    words = line.strip().lower().split()
                    words = self.bpeTokenizer(words,self.V) # tokenizing the line with self.V
                    combinations = self.comboMaker(combinations,words) # updating combinations
            
            # if combinations is empty, it means no more morges were possible, in any of the lines,
            # which means that all words are whole and not split into subwords.
            if not combinations: 
                print(f"Stopping process: no more merges possible.")
                return self.V
            
            most_freq = max(combinations, key=combinations.get) # retrieveing most frequent combo
            self.V = set(self.V) # casting self.V back to set to add the combination
            self.V.add(most_freq)
            self.V = sorted(list(self.V), key=len, reverse=True) # sorting it to make sure the longest are first
        return self.V
    

    def printV(self):
        '''
        This function executes buildV() and prints the vocab into a file.

        Args: None, everything is saved as an argument of the BPE object.

        Returns: None.
        '''
        self.buildV()
        with open("vocab.txt","w") as f:
            for comb in self.V:
                print(comb, file=f)
    

    def tokenize_text(self):
        '''
        This functions uses the previously built vocab to tokenize the text for the user.

        Args: None, everything is saved as an argument of the BPE object.

        Returns: None.
        '''
        if not os.path.exists("vocab.txt"): # check if the vocab has been built.
            print("You need to run bpe.printV() first to output a file with the vocabulary.") 
            return 0
    
        with open("vocab.txt","r") as vocab, open(self.text,"r") as f, open("tokenized_text.txt","w") as output:
            V = vocab.readlines()
            V = [line.strip() for line in V] # rebuilding V as a list of strings 
            for line in f:
                line = line.strip().lower().split()
                line = self.bpeTokenizer(line,V) # tokenizing the words using V

                new_line = "" # printing the tokenized words in one string separated by white spaces
                for word in line:
                    for i in range(len(word)):
                        new_line += (word[i]+" ")
                print(new_line, file=output) # printing the tokenized words into a file
        print("Tokenization completed.")
            


def main():
    bpe = BPE("/path","/path",23)
    bpe.printV()
    bpe.tokenize_text()

if __name__ == "__main__":
    main()






'''
References:
Jurafsky, Daniel and James H. Martin (2025). Speech and Language Processing: An Introduction to Natural 
    Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3rd. Online 
    manuscript released January 12, 2025. URL: https://web.stanford.edu/~jurafsky/slp3/.
'''