import math


class NGramModel:

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
