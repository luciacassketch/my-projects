'''
Task:
Perform a comparative evaluation of the perceptron algorithm and multinomial logistic regression on the morphological feature 
classification (see file perceptron.py).  Multinomial logistic regression is described in more detail in Jurafsky and Martin 
(2025, Section 5.3 and 5.8). Justify the evaluation metrics and methodology, and use statistical significance testing when appropriate.
For this, see Jurafsky and Martin (2025, Section 4.9) on the paired boot-strap test, and implement it yourself using pure Python 
with the random module from the standard library.
'''


import random
from perceptron import Perceptron
from multinomial_log_reg import MLR
from myNLP import fileReader

p = Perceptron("itatrain.txt")
p.train()
print("perceptron trained")
p_accuracy_x, p_preds = p.test("itatest.txt")

model = MLR("itatrain.txt")
model.train(lr=0.2)
print("model trained")
m_accuracy_x, m_preds = model.test("itatest.txt")

delta_x = m_accuracy_x - p_accuracy_x

file = fileReader("itatest.txt")
len_file = len(file)


## bootstrap function
s = 0
b = 10000
for i in range(b):      # for every sample ...
    print(i)
    p_xi = []
    m_xi = []

    for j in range(len_file):     # ... get random test results
        n = random.randrange(len_file)
        p_xi.append(p_preds[n])
        m_xi.append(m_preds[n])

    p_accuracy_xi = sum(p_xi)/len(p_xi)
    m_accuracy_xi = sum(m_xi)/len(m_xi)
    delta_xi = m_accuracy_xi - p_accuracy_xi

    s += (delta_xi >= 2*delta_x)
    print(delta_x, "\t", delta_xi)


print(f"{(s/b):.4f}")       # p-value



'''
References:
Jurafsky, Daniel and James H. Martin (2025). Speech and Language Processing: An Introduction to Natural 
    Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3rd. Online 
    manuscript released January 12, 2025. URL: https://web.stanford.edu/~jurafsky/slp3/.
'''