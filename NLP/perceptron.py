'''
Task:
Use the perceptron algorithm in a feature-based model to learn morphological analysis with the UniMorph (Batsuren et al. 2022)
database. Given a word-form and a part of speech tag, the task is to predict the combination of morphological features. For 
instance, given the English verb 'rectangularizes', the correct feature combination (using UniMorph's schema) is 
'PRS;3;SG', for present tense, third person, singular.
'''


from math import inf
import numpy
from myNLP import fileReader
from collections import defaultdict


class Perceptron():
    def __init__(self, path):
        ''' This function prepares the data for training. It creates the dictionaries for prefixes and suffixes plus 
        one that maps tags (N, V, ADJ) to all the combinations they appear with in the data. Every prefix and
        suffix is mapped to a number. Another dictionary for feature combinations is then created where every
        combination is also mapped to a number.

        Args: 
            Path to training file.
        Returns:
            None.
        '''
        self.trainpath = path

        self.prefixes = {}
        self.suffixes = {}
        self.tag_to_features = {}

        file = fileReader(self.trainpath)
        for line in file:
            # each line is of the format:   "carpmes carpmesarnas    N;GEN;PL;DEF"
            tag = line[2].split(";")[0]

            if tag not in self.tag_to_features:
                self.tag_to_features[tag] = set()
            self.tag_to_features[tag].add(line[2])

            if len(line[1]) >= 5:   
                for i in range(1,6):    # only take the first 1-5 letters as self.prefixes ...
                    prefix = line[1][:i]
                    if prefix not in self.prefixes:
                        self.prefixes[prefix] = len(self.prefixes)+1
                    suffix = line[1][len(line[1])-i:]   # ...and the last 1-5 as self.suffixes.
                    if suffix not in self.suffixes:
                        self.suffixes[suffix] = len(self.suffixes)+1
            else:   # if the word is shorter than 5
                for i in range(1,len(line[1])+1):
                    prefix = line[1][:i]
                    if prefix not in self.prefixes:
                        self.prefixes[prefix] = len(self.prefixes)+1
                    suffix = line[1][len(line[1])-i:]
                    if suffix not in self.suffixes:
                        self.suffixes[suffix] = len(self.suffixes)+1

        # build the dictionary with feature combinations mapped to a number
        self.feature_combinations = {}
        for tag in self.tag_to_features:
            for combo in self.tag_to_features[tag]:
                self.feature_combinations[combo] = len(self.feature_combinations)+1
        

    def train(self):
        '''This function performs training of the model. It initializes a dictionary of weights which will be zero 
        until updated during training. For every prefix, suffix, and every tag-features combination 
        the model builds the function (key to the dict) with (combination_index, n, affix_index) in which n is 0
        if the affix is a prefix and 1 if it's a suffix. This way, each and every combination of affix and tag-features
        has a unique weight in the dictionary. All combinations and functions are gathered in the outputs list and in 
        the outputs_dict, from which the model extracts the combo with the highest weights sum. Then it updates the 
        weights if the prediction was wrong.

        Args:
            Self.
        Returns:
            None.
        '''
        file = fileReader(self.trainpath)
        self.weights = defaultdict(int)
        for epoch in range(1):
            for line in file:
                tag = line[2].split(";")[0]     # get tag for input
                outputs = []
                outputs_dict = defaultdict(list)    # dict to compute weight sum for every combo 

                # for every tag get the weight associate with the combination of that tag and each affix ...
                for combo in self.tag_to_features[tag]:  
                    if len(line[1]) >= 5:
                        for i in range(1,6):
                            prefix = line[1][:i]
                            f_n = (self.feature_combinations[combo], 0, self.prefixes[prefix])
                            outputs.append((combo,f_n))  # ... and append to output list ...
                            outputs_dict[combo].append(f_n) # ... and to outputs dict.

                            suffix = line[1][len(line[1])-i:]
                            f_n = (self.feature_combinations[combo], 1, self.suffixes[suffix])
                            outputs.append((combo,f_n))
                            outputs_dict[combo].append(f_n)
                    else:
                        for i in range(1,len(line[1])+1):
                            prefix = line[1][:i]
                            f_n = (self.feature_combinations[combo], 0, self.prefixes[prefix])
                            outputs.append((combo,f_n))
                            outputs_dict[combo].append(f_n)

                            suffix = line[1][len(line[1])-i:]
                            f_n = (self.feature_combinations[combo], 1, self.suffixes[suffix])
                            outputs.append((combo,f_n))
                            outputs_dict[combo].append(f_n)

                ## get the combo with the highest weight sum
                max_weight = -inf
                for combo in outputs_dict:
                    summed_weights = sum(self.weights[function] for function in outputs_dict[combo])
                    if summed_weights > max_weight:
                        max_weight = summed_weights
                        predicted = combo

                gold = line[2]
                if predicted != gold:   # update the self.weights if the prediction was wrong
                    for output in outputs:
                        combo, f_n = output
                        if combo == gold:
                            self.weights[f_n] += 1
                        elif combo == predicted:
                            self.weights[f_n] -= 1



    def test(self,testpath):
        '''This function tests the accuracy of the model on a test set. It builds functions and output list the same
        way as train() does, and then retrieves the output with the highest weight.
        
        Args:
            Path to test file, self.
        Return:
            Accuracy statistics.
        '''
        file = fileReader(testpath)
        tot = 0     # counter for total test lines
        correct = 0     # counter for correct predictions
        preds = []      # needed for bootstrap test (minipaper) - ignore it
    
        for line in file:
            tot += 1    # update counter
            tag = line[2].split(";")[0]     # get tag for input
            outputs_dict = defaultdict(list)

            for combo in self.tag_to_features[tag]:
                if len(line[1]) >= 5:
                    for i in range(1,6):
                        prefix = line[1][:i]
                        f_n = (self.feature_combinations[combo], 0, self.prefixes.get(prefix,-1))
                        outputs_dict[combo].append(f_n)

                        suffix = line[1][len(line[1])-i:]
                        f_n = (self.feature_combinations[combo], 1, self.suffixes.get(suffix,-1))
                        outputs_dict[combo].append(f_n)
                else:
                    for i in range(1,len(line[1])+1):
                        prefix = line[1][:i]
                        f_n = (self.feature_combinations[combo], 0, self.prefixes.get(prefix,-1))
                        outputs_dict[combo].append(f_n)

                        suffix = line[1][len(line[1])-i:]
                        f_n = (self.feature_combinations[combo], 1, self.suffixes.get(suffix,-1))
                        outputs_dict[combo].append(f_n)

            max_weight = -inf
            for combo in outputs_dict:
                summed_weights = sum(self.weights[function] for function in outputs_dict[combo])
                if summed_weights > max_weight:
                    max_weight = summed_weights
                    predicted = combo

            if predicted == line[2]:
                correct += 1    # if correct prediction, update the counter
                preds.append(1)
            else:
                preds.append(0)

        accuracy = correct/tot
        preds = numpy.array(preds)
        return accuracy, preds



def main():

    p = Perceptron("/path")
    p.train()
    accuracy, _ = p.test("/path")

    print(accuracy)


if __name__ == "__main__":
    main()




'''
References:
Batsuren, Khuyagbaatar et al. (2022). UniMorph 4.0: Universal Morphology. 
    arXiv: 2205.03608 [cs.CL]. URL: https://arxiv.org/abs/2205.03608.
'''