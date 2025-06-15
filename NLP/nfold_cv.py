import naiveBayes
import myNLP


def evaluation_function(training_set: list, test_set: list) -> float:
    nc = naiveBayes.NaiveBayesDataset(training_set,test_set)
    return nc.accuracy()


def n_fold(data_points: list, evaluation_function, n: int) -> list[float]:
    '''
    A generic n-fold cross validation function that can be applied to, for instance, the Naive Bayes classifier 
    (see file naiveBayes.py).

    Args:
        data_points: a list or tuple of data points;
        evaluation_function: function that returns some evaluation metric, such as accuracy, after training
            a classifier on the training set and evaluating on the test set;
        n: the number of folds.
    
    Returns:
        Accuracy score on every fold.
    '''
    accuracy = []

    # create folds and pass the data to the evaluation_function to compute accuracy.
    for i in range(n):
        test_set = []
        train_set = []
        for j in range(len(data_points)):
            rem = j%n
            if rem == i:
                test_set.append(data_points[j])
            else:
                train_set.append(data_points[j])

        accuracy.append(evaluation_function(train_set,test_set))

    return accuracy

            
                    
cc = myNLP.CategorizedCorpus("/path")
data = []

print(n_fold(data,evaluation_function,10))




