'''
Task:
Extend the feature-based perceptron (see file perceptron.py), so that it can handle sequences. This is called the 
structured perceptron and is a straightforward extension of the standard perceptron algorithm. Instead of computing 
(as in perceptron.py) one predicted tag at a time and updating the weights based on a comparison between that and 
the actual tag, you should compute the best tagging of a whole sequence of tags using some kind of search. Then 
decrease the weights for the features of the predicted tag sequence, and increase the weights for the features of the 
gold standard tag sequence. Note that features are now able to use sequences of tags. 
'''

from collections import defaultdict

class StructuredPerceptron:
    ''' An extension of the simple perceptron. Instead of predicting one word at a time, it takes a sequence (sentence) as input and 
    predicts tags for every word using some kind of context-based feature-function. It then outputs a sequence of tags.
    '''

    def __init__(self, train_path):
        ''' This function preprocesses the data to get the tags the model will have to predict, gathered in NERtags.
            Args:
                Path to training file.
            Returns:
                None.
        '''
        self.train_path = train_path
        NERtags = {}    # initializing the tag dictionary
        
        sentences = self.retrieve_sentences(self.train_path)
        for sentence in sentences:
            for word in sentence:   # the lines are of the format: European NNP I-NP I-ORG
                word = word.split()
                ## fill the dictionary
                if word[-1] not in NERtags:
                    NERtags[word[-1]] = len(NERtags)

        self.NERtags = NERtags

    def retrieve_sentences(self,path):
        ''' Since every line in the file is a word or an empty line, this function retrieves whole sentences.
            Args:
                Path to training file.
            Returns:
                None.
        '''
        sentence = []   # initializing list to collect sentence (every line in the file is either a word or an empty line)
        with open(path) as f:
            for line in f:
                line = line.strip()     
                if line == "" or line == '-DOCSTART- -X- -X- O':
                    ## if the line is empty or it's DOCSTART but the sentence list is also empty, just keep going ...
                    if sentence == []:
                        pass
                    ## ... instead, if it is full it means it has found a whole sentence, so it's time to process it.
                    else:
                        yield sentence
                    sentence = []
                else:
                    ## if the line isn't empty, add the line to the sentence
                    sentence.append(line)
    
    def processing(self, sentence, test=False):
        ''' This function processes the sentence to extract all feature functions and get the best one for every word.
            Args:
                sentence: list of words with their tags;
                test: flag to control the process of updating the weights' dict depending on whether it's training 
                    or testing.
            Returns:
                final_outputs: the model's guesses on every word's tags;
                gold_sentence: the gold standard for the whole sentence;
                total_outputs: all (not just the best) feature-function outputs for every word.
        '''
        gold_sentence = [word.split()[-1] for word in sentence]     # the right NER-tags
        tags_sentence = ["<START>"] + [word.split()[1] for word in sentence] + ["<END>"]    # pos-tags for every word + pads
        words_sentence = ["<START>"] + [word.split()[0] for word in sentence] + ["<END>"]   # words + pads
        final_outputs = []      # list of outputs of NER-tags for the whole sentence
        total_outputs = []      # list of lists of all feature-functions outputs for every word (needed to update weights)

        i = 1
        while i <= len(tags_sentence)-2:    # in the range of the length of the sentence without pads
            outputs_dict = defaultdict(list)    # dict of all feature-functions outputs for every word organized by tag
            list_of_outputs = []    # list of all feature-functions outputs for every word 

            ## check tag of previous word, tag of target word, tag of following word
            seq = (tags_sentence[i-1], tags_sentence[i], tags_sentence[i+1])
            for tag in self.NERtags:
                f_n = (seq,tag)     # the feature_function (key to weight dict)
                if test:
                    outputs_dict[tag].append((f_n,self.weights.get((f_n),0)))
                else:
                    outputs_dict[tag].append((f_n,self.weights[f_n]))   # initialize weights only in training
                    list_of_outputs.append((f_n,self.weights[f_n]))

            ## check previous word, tag of previous word, and if target word's first letter is capitalized
            seq = (words_sentence[i-1], tags_sentence[i-1], words_sentence[i][0].isupper()) 
            for tag in self.NERtags:
                f_n = (seq,tag)
                if test:
                    outputs_dict[tag].append((f_n,self.weights.get((f_n),0)))
                else:
                    outputs_dict[tag].append((f_n,self.weights[f_n]))
                    list_of_outputs.append((f_n,self.weights[f_n]))
            
            ## check target word and if target word's first letter is capitalized
            seq = (words_sentence[i], words_sentence[i][0].isupper())
            for tag in self.NERtags:
                f_n = (seq,tag)
                if test:
                    outputs_dict[tag].append((f_n,self.weights.get((f_n),0)))
                else:
                    outputs_dict[tag].append((f_n,self.weights[f_n]))
                    list_of_outputs.append((f_n,self.weights[f_n]))
            
            ## check next word
            seq = (words_sentence[i+1])
            for tag in self.NERtags:
                f_n = (seq,tag)
                if test:
                    outputs_dict[tag].append((f_n,self.weights.get((f_n),0)))
                else:
                    outputs_dict[tag].append((f_n,self.weights[f_n]))
                    list_of_outputs.append((f_n,self.weights[f_n]))
            
            ## only if it's the second or more word of the sentence (to make sure we already have at least one output) ...
            if i >= 2:
                ## ... check previous output, tag of previous word, if target word's first letter is capitalized
                seq = (final_outputs[i-2], words_sentence[i][0].isupper()) 
                for tag in self.NERtags:
                    f_n = (seq,tag)
                    if test:
                        outputs_dict[tag].append((f_n,self.weights.get((f_n),0)))
                    else:
                        outputs_dict[tag].append((f_n,self.weights[f_n]))
                        list_of_outputs.append((f_n,self.weights[f_n]))
            
            total_outputs.append(list_of_outputs)

            # get best tag based on the sum of its weights ...
            final_tag, final_sum = max(((tag, sum(weight for _, weight in outputs)) for tag, outputs in outputs_dict.items()), key=lambda x: x[1])           
            final_outputs.append(final_tag)     # ...and append its output to the final outputs.
            i += 1
        
        return final_outputs, gold_sentence, total_outputs

    def train(self, n=3):
        ''' This function builds the weights by learning which feature-function works best to predict the correct tag.

            Args:
                n: number of epochs, default 3.
            Returns:
                None.
        '''
        self.weights = defaultdict(int)     # initializing weight dictionary, with structure <feature_function: weight>  

        for epoch in range(n):
            sentences = self.retrieve_sentences(self.train_path)

            for sentence in sentences:
                final_outputs, gold_sentence, total_outputs = self.processing(sentence)

                for j,output_list in enumerate(total_outputs):      # for every list j of feature-functions outputs ...
                    if final_outputs[j] != gold_sentence[j]: 
                        for output in output_list:                      # ...for every output in that list ...
                            if output[0][1] == gold_sentence[j]:        # ...if it is equal to the right prediction for the word j, increment the weight...
                                self.weights[output[0]] += 1
                            else:
                                self.weights[output[0]] -= 1            # ...if not, decrease it.


    def test(self, test_path):
        ''' This function tests the model on the test data.

            Args:
                test_path: path to test file.
            Returns:
                Accuracy score.
        '''
        sentences = self.retrieve_sentences(test_path)
        tot = 0     # counter for all words
        correct = 0 # counter for correct predictions
        for sentence in sentences:
            tot += len(sentence)
            final_outputs, gold_sentence, _ = self.processing(sentence, test=True)
            
            for j in range(len(gold_sentence)):
                if gold_sentence[j] == final_outputs[j]:
                    correct += 1    # update counter if prediction is correct

        return(correct/tot)        # return accuracy score


p = StructuredPerceptron("/path")
p.train()
print(p.test("/path"))