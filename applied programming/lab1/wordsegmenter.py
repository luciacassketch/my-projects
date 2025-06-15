import json
import sys
import math


class WordSegmenter:

    def __init__(self, filename):
        with open(filename, "r") as file:
            self.dic = json.load(file)
        self.logN = math.log(sum(self.dic.values()))
        self.maxlen = max([len(w) for w in self.dic])

    def segment(self, text:  str) -> list[str]:
        # j index to go back
        # i index to iterate the string

        best_probs: list[float] = [0.0]  # best probs for every i
        seq: list[list[str]] = [[""],]  # seq linked to best probs

        i: int = 1
        next_seq: list[str] = [""]  # optimal sol for each i (goes into best_probs)

        while (i <= len(text)):

            next_seq = [""]
            max_prob: tuple[float, list[str]] = (-math.inf, [])  # initialized max prob

            limit: int = 0  # limit to only go back to the length of the longest word in the dic
            if (i > self.maxlen):
                limit = i-self.maxlen

            for j in range(limit, i):

                poss_seq: list[str] = []  # list to keep the sequence linked to the best prob
                word: str = text[j:i]

                if (len(word) == 1 and word not in self.dic):  # check if it is a single char
                    poss_seq = seq[j] + [word]  # what comes before the word + the word
                    new_prob: tuple[float, list[str]] = (best_probs[j] + math.log(1) - self.logN, poss_seq)  # (prob, seq)
                    if (new_prob > max_prob):  # update max_prob if the prob is higher than last one
                        max_prob = new_prob

                if (word in self.dic):  # check if word is in dic before calculating prob
                    poss_seq = seq[j] + [word]
                    new_prob = (best_probs[j] + math.log(self.dic[word]) - self.logN, poss_seq)
                    if (new_prob > max_prob):
                        max_prob = new_prob

            best_probs.append(max_prob[0])  # update the list with the computed optimal prob for this i

            next_seq = max_prob[1]  # the corresponding word sequence
            seq.append(next_seq)  # update the list with the optimal split for this i

            i = i+1

        return next_seq


def main():

    filename = sys.argv[1]
    chinese_lex = WordSegmenter(filename)

    with open(sys.argv[2], "r") as inputfile:
        text = inputfile.readlines()

    for line in text:
        line = line.strip()
        newline = chinese_lex.segment(line)
        print(" ".join(newline).strip())


if __name__ == "__main__":
    main()
