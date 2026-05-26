import json


def preprocess(files: list):
    '''
    This function reads the JSON files where the sentences have been 
    saved for processing.

    Args: 
        files: list of JSON files with the structure sentence_code:sentence
    
    Output:
        a list of the sentence codes, a list of the sentences,
        a list with groud truth values.
    '''

    keys_to_sents = {}
    ground_truth_seed = 0
    ground_truth = []

    for file in files:
        dictionary = {}

        with open(file, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        
        if dictionary:  # sanity check
            
            # the whole file gets saved back into a dict 
            keys_to_sents.update(dictionary)

            # the gt is built by multiplying the seed (different for every file, 
            # symbolises a sense) for the lenght of the dictionary
            ground_truth.extend([ground_truth_seed] * len(dictionary))
            ground_truth_seed += 1      # to give every file(=sense) a different code

    return list(keys_to_sents.keys()), list(keys_to_sents.values()), ground_truth
