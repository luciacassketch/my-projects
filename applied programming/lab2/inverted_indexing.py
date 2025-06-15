import os
import re
import sys
import gzip
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def readfile(dir:str, filename:str):
    """ Function to read the zipped file into a dictionary,
    with sentence-code as key and sentence-string as value.
    """
    dic = {}
    with gzip.open(os.path.join(dir,filename), "rt") as f:
        for line in f:
            line = line.strip().split("\t")
            dic[line[0]] = line[1]
    return dic

def find_freq(dic:dict):
    """ Function to select only the words that occur 10 or more times in the data.
    Goes through every sentence in the input dictionary. 
    """
    freq = {}
    for code in dic:
        for word in dic[code]:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    return [word for word in freq if freq[word]>=10]


def main():

    args = []
    path = sys.argv[1]

    for file in os.listdir(path):
        if file.startswith("."):
            continue
        args.append(file)

## reading the text file into a dictionary

    DICT = {}

    for file in args:
        DICT[os.path.basename(file).split(".")[0]] = readfile(path,file)

## keep common sentences, tokenize words (+ lower them and eliminate duplicates)

    l1 = os.path.basename(args[0]).split(".")[0]
    common = set(DICT[l1].keys())

    for i in range(1,len(DICT)): 
        common_l = set(DICT[os.path.basename(args[i]).split(".")[0]].keys())       
        diff1 = common - common_l
        diff2 = common_l - common
        diff = diff1.union(diff2)
        common = common - diff     # final list of common sentences

    for l in DICT:
        # keep common sentences
        DICT[l] = {key: value for key, value in DICT[l].items() if key in common}
        # tokenize words, lower them and eliminate duplicates
        for code in DICT[l]:
            DICT[l][code] = re.findall(r'\w+(?:-\w+)?', DICT[l][code])
            prefix = l + "/"
            DICT[l][code] = list(set([prefix+word.lower() for word in DICT[l][code]]))
                   

## filter for words that occur 10 times or more 

    words_to_keep = []
    for l in DICT:
        words_to_keep_in_l = find_freq(DICT[l])
        words_to_keep += words_to_keep_in_l


## build matrix

    words_map = {word:i for i,word in enumerate(words_to_keep)}
    common = sorted(common)
    common_map = {sentence:i for i,sentence in enumerate(common)}
    i = []
    j = []

    for l,sentences in DICT.items():
        for code,sentence in sentences.items():
            for word in sentence:
                if word in words_map:
                    i.append(words_map[word])
                    j.append(common_map[code])


    i = np.array(i)
    j = np.array(j)

    data = np.ones(len(i), dtype=np.float32)
    sparse_arrays = csr_matrix((data, (i,j)), shape=(len(words_to_keep), len(common)))


## reduce matrix to 100 columns

    svd = TruncatedSVD(n_components=100)
    svd.fit(sparse_arrays)
    word_vectors = svd.transform(sparse_arrays)


## count words for language (for the first line of the file)

    words_for_lang = {}
    for l in DICT:
        count = 0
        for word in words_to_keep:
            if word.startswith(l + "/"):
                count += 1
        words_for_lang[l] = count


## format embeddings and print out the file

    for l in DICT:
        file = l + ".vec.gz"
        with gzip.open(file, "wt", encoding="utf-8") as f:
            print(f'{words_for_lang[l]} 100', file=f)
            for word in words_to_keep:
                if word.startswith(l + "/"):
                    vector = word_vectors[words_map[word]]
                    str_vector = " ".join(map(str, vector))
                    print(f'{word} {str_vector}', file=f)
    



if __name__ == "__main__":
    main()