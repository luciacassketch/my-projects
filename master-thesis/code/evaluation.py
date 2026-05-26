from itertools import combinations
from sklearn.metrics import v_measure_score as vm



def paired_f_score(y_true, y_pred):
    """
    Calculates the Paired F-Score as described in SemEval-2010 Task 14 (Manandhar et al., 2010, Section 3.1.2).

    This metric evaluates clustering correctness by comparing pairs of instances that are 
    grouped together in the predictions versus the gold standard. Instead of comparing 
    cluster labels directly, it analyses whether pairs of items assigned to the same cluster 
    in the prediction also belong to the same cluster in the ground truth.

    Args:
        y_true (list or np.ndarray): The ground truth labels (gold standard).
        y_pred (list or np.ndarray): The predicted cluster labels.

    Returns:
        float: The Paired F-Score. 
    """
    
    # 1. generate all index pairs for the Gold Standard (S in the paper)
    # group indices by each gold label
    gold_groups = {}
    for idx, label in enumerate(y_true):
        if label not in gold_groups:
            gold_groups[label] = []
        gold_groups[label].append(idx)
    
    # create a set of pairs (ordered tuples) for the Gold Standard (F(S) in the paper)
    gold_pairs = set()
    for indices in gold_groups.values():
        if len(indices) > 1:
            gold_pairs.update(combinations(sorted(indices), 2))  # generates all possible pairs (i, j)
    

    # 2. generate all index pairs for the Predictions (K in the paper)
    # group indices by each predicted cluster
    pred_groups = {}
    for idx, label in enumerate(y_pred):
        if label not in pred_groups:
            pred_groups[label] = []
        pred_groups[label].append(idx)
        
    # create a set of pairs for the Predictions (F(K) in the paper)
    pred_pairs = set()
    for indices in pred_groups.values():
        if len(indices) > 1:
            pred_pairs.update(combinations(sorted(indices), 2))
    

    # 3. calculate the intersection (correctly grouped pairs)    
    # |F(K) ∩ F(S)|
    common_pairs = gold_pairs.intersection(pred_pairs)
    n_common = len(common_pairs)
    
    # 4. calculate Precision and Recall according to formulas 7 and 8 in the paper:
    # Precision = |Intersection| / |Predicted|
    # Recall = |Intersection| / |Gold|
    
    n_pred = len(pred_pairs)
    n_gold = len(gold_pairs)
    
    if n_pred == 0 and n_gold == 0:
        return 1.0  # edge case: no data or all singletons, we consider it perfect
            
    if n_pred == 0:
        precision = 0.0
    else:
        precision = n_common / n_pred
        
    if n_gold == 0:
        recall = 0.0
    else:
        recall = n_common / n_gold
        

    # 5. calculate the harmonic mean (F-Score)
    if precision + recall == 0:
        return 0.0
        
    f_score = 2 * (precision * recall) / (precision + recall)
    
    return f_score


def evaluate(predictions, ground_truth):
    """
    Computes two clustering evaluation metrics: V-Measure and Paired F-Score.

    Args:
        predictions (list or np.ndarray): The predicted cluster labels.
        ground_truth (list or np.ndarray): The ground truth labels.

    Returns:
        tuple:
            v_measure (float): The V-Measure score.
            f_score (float): The Paired F-Score.
    """

    v_measure = vm(ground_truth, predictions)
    f_score = paired_f_score(ground_truth, predictions)

    return v_measure, f_score



'''
References:

Suresh Manandhar, Ioannis Klapaftis, Dmitriy Dligach, and Sameer Pradhan. 2010. 
    SemEval-2010 task 14: Word sense induction & disambiguation. In Proceedings of 
    the 5th International Workshop on Semantic Evaluation, pages 63–68, Uppsala, Sweden. 
    Association for Computational Linguistics.
'''
