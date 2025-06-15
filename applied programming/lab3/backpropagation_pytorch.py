import re
import csv
import torch
import numpy as np
from collections import defaultdict


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

            
def no_batches_model(data, score, model, optimizer, test_data, test_score):
    """Train the model without mini-batches.

    Args:
        training features, training label, linear model, pytorch-optimizer,
        test features, test label. 

    Returns:
        the trained model.
    """
    for epoch in range(30):
        for i,example in enumerate(data):
            x = example
            y = score[i]
            model.zero_grad()
            y_pred = model(x)
            loss = ((y_pred - y)**2).mean()
            loss.backward()
            optimizer.step()

        w = model.weight.detach().cpu().numpy()[0]
        bias = model.bias.detach().cpu().numpy()
        train_preds = model(data).flatten()
        train_MSE = ((train_preds - score)**2).mean()
        preds = model(test_data).flatten()
        MSE = ((preds - test_score)**2).mean()
        print(f"{epoch=}, w = [{w[0]:8.4f}, {w[1]:8.4f}], bias = {bias}, loss = {train_MSE}, test loss = {MSE}")

    return model


def mini_batches_model(data, score, model, optimizer, test_data, test_score, batch_size):
    """Train the model with mini-batches.

    Args:
        training features, training label, linear model, pytorch-optimizer,
        test features, test label, batch size. 

    Returns:
        the trained model.
    """
    for epoch in range(50):
        i = 0
        lastBatch = 0   # flag that signals when all the training data has been processed
        while (lastBatch == 0):
            if (i + batch_size <= len(data)):  # checking if, by slicing the data, we would exceed the length of it
                x = data[i:i+batch_size]
                y = score[i:i+batch_size]
                model.zero_grad()
                y_pred = model(x).flatten()
                loss = ((y_pred - y)**2).mean()
                loss.backward()
                optimizer.step()
                i += batch_size
                if i == len(data):  # needed for when the data is divisible by the batch size so "else" won't be accessed
                    lastBatch = 1
            else:  # if we cannot slice without exceeding the length of the data, it means the last batch is smaller than the batch size ...
                x = data[i:len(data)]  # ... so we just take all data that is left ...
                y = score[i:len(data)]
                model.zero_grad()
                y_pred = model(x).flatten()
                loss = ((y_pred - y)**2).mean()
                loss.backward()
                optimizer.step()
                lastBatch = 1  # ... and it also means this is the last batch.

        w = model.weight.detach().cpu().numpy()[0]
        bias = model.bias.detach().cpu().numpy()
        train_preds = model(data).flatten()
        train_MSE = ((train_preds - score)**2).mean()
        preds = model(test_data).flatten()
        MSE = ((preds - test_score)**2).mean()
        print(f"{epoch=}, w = [{w[0]:8.4f}, {w[1]:8.4f}], bias = {bias}, loss = {train_MSE}, test loss = {MSE}")

    return model


def main(method = 0):
    """Prepare data and model, choose and execute a training and testing method.

    Args:
        Method: 0 (default) for the no-batch model, 1 for the mini-batches model.

    Returns:
        None.
    """
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

    device = 'cpu'

    data = torch.tensor(Xtrain.astype(np.float32), device=device)
    score = torch.tensor(y_train_z.astype(np.float32), device=device)
    test_data = torch.tensor(Xtest.astype(np.float32), device=device)
    test_score = torch.tensor(y_test_z.astype(np.float32), device=device)
    model = torch.nn.Linear(2, 1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.008)    

    if method == 0:
        no_batches_model(data, score, model, optimizer, test_data, test_score)
    elif method == 1:
        mini_batches_model(data, score, model, optimizer, test_data, test_score, 16)


if __name__ == "__main__":
    main(1)
