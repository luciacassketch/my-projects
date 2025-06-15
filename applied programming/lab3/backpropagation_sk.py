import numpy as np
import re
import csv
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def read_essay_sets(filename: str) -> dict[int, list[dict]]:
    """Read all essay sets from an ASAP data file.

    Args:
        filename -- tab-separated ASAP data file, cleaned to be unicode
                    compatible

    Returns:
        dict mapping essay set number (int) to a list of dicts, representing
        all the essays in the given essay set.
    """
    essay_set = defaultdict(list)
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            essay_set[int(row["essay_set"])].append(
                    {"text": row["essay"],
                     "score": int(row["domain1_score"])})
    return essay_set


def main():
    
    essay_set = read_essay_sets("/path/to/file.tsv")
    train = []
    test = []
    y_train = []
    y_test = []
    for essay in essay_set[1]:
        train.append(re.findall(r"[\w']+", re.sub(r'@[A-Z]+\d+', '', essay["text"]).lower()))
        y_train.append(int(essay["score"]))
    for essay in essay_set[2]:
        test.append(re.findall(r"[\w']+", re.sub(r'@[A-Z]+\d+', '', essay["text"]).lower()))
        y_test.append(int(essay["score"]))

    Xtrain = np.zeros((len(train), 2), dtype=np.float32)
    for i, essay in enumerate(train):
        n = len(essay)
        length = pow(n,1/4)
        k = len(set(essay))
        OVIX = np.log(n)/(2 - (np.log(k)/np.log(n)))
        Xtrain[i][0] = length
        Xtrain[i][1] = OVIX

    Xtest = np.zeros((len(test), 2), dtype=np.float32)
    for i, essay in enumerate(test):
        n = len(essay)
        length = pow(n,1/4)
        k = len(set(essay))
        OVIX = np.log(n)/(2 - (np.log(k)/np.log(n)))
        Xtest[i][0] = length
        Xtest[i][1] = OVIX

    y_train_z = (y_train - np.mean(y_train)) / np.std(y_train)
    y_test_z = (y_test - np.mean(y_test)) / np.std(y_test)


    model = LinearRegression().fit(Xtrain, y_train_z)
    preds_train = model.predict(Xtrain)
    preds_test = model.predict(Xtest)

    mse_train = mean_squared_error(y_train_z,preds_train)
    mse_test = mean_squared_error(y_test_z,preds_test)

    print(f"{mse_train=}")
    print(f"{mse_test=}")
    print(f"{model.coef_=}")
    print(f"{model.intercept_=}")


if __name__ == "__main__":
    main()