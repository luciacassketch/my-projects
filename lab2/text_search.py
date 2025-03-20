import os
import numpy as np
import gzip


class WordVectors:

    def __init__(self, filename):
        self.lang_code = os.path.basename(filename).split(".")[0]
        dic = []

        with gzip.open(filename, "rt") as file:
            line1 = file.readline().split()
            # read number of words and length of embedding to initialize matrix
            self.rows = int(line1[0]) 
            self.cols = int(line1[1])
            embeddings = np.zeros((self.rows, self.cols), dtype=np.float32)

            # now fill the matrix with the rest of the lines
            for i in range(self.rows):
                line = file.readline().split()
                try:
                    word = line[0].lower()
                    dic.append(word[len(self.lang_code)+1:]) # subtract the language code and \
                    embeddings[i] = np.array([float(n) for n in line[1:]]) 
                except IndexError: # to avoid empty lines
                    pass

        self.dic_map = {word:i for i,word in enumerate(dic)} # to efficiently link words to their vector through index
        self.embeddings = embeddings


    def make_sentence_vector(self, words: list[str]) -> np.ndarray:
        words = [word.lower() for word in words if word.lower() in self.dic_map]
        words = set(words)

        if not words:
            raise ValueError
        
        sentence = np.array([self.embeddings[self.dic_map[word]] for word in words], dtype=np.float32)
        
        return np.mean(sentence, axis=0)
        

class TextSearch:

    def __init__(self, filenames):
        # build a dictionary with language_code,WordVectors as key,value pairs for every file in input
        DICT = {}
        for filename in filenames:
            DICT[os.path.basename(filename).split(".")[0]] = WordVectors(filename)
        self.DICT = DICT

    
    def index_text(self, filename: str, language_code: str, min_words: int= 1, max_words: int = None) -> None:
        file_vectors = []
        file_sentences = []
        with gzip.open(filename, mode="rt") as f:
            for line in f:
                line = line.strip()
                listed_line = line.split()
                length = len(listed_line)
                if (min_words is None or length >= min_words) and (max_words is None or length <= max_words):
                    try:
                        sentence_vector = self.DICT[language_code].make_sentence_vector(listed_line)
                        file_vectors.append(sentence_vector)
                        file_sentences.append(line)
                    except (ValueError):
                        pass

        self.filename = filename
        self.file_vectors = file_vectors
        self.file_sentences = file_sentences


    
    def search(self, query: list[str], language_code: str, n_matches: int=1)-> list[tuple[float, str, str]]:
        
        vector_query = self.DICT[language_code].make_sentence_vector(query)
        norm_query = np.linalg.norm(vector_query)

        cos_sim = []
        for i, vector in enumerate(self.file_vectors):
            cos_sim.append((np.dot(vector_query,vector)/(norm_query*np.linalg.norm(vector)), i))
        set_cos_sim = set(cos_sim)
        sorted_cos_sim = sorted(set_cos_sim, reverse=True)[:n_matches]

        result = []
        for n in sorted_cos_sim:
            result.append((n[0], self.filename, self.file_sentences[n[1]]))
        
        return result