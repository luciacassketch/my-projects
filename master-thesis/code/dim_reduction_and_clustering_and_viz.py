import umap
import hdbscan
from collections import Counter


def clustering(transformed_sents, n_sents, neighbours, hdb_min_cluster_size, hdb_min_samples=2):
    """
    Performs dimensionality reduction and density-based clustering on sentence embeddings.

    Args:
        transformed_sents (np.ndarray): The input sentence embeddings (high-dimensional).
        n_sents (int): The total number of sentences in the dataset.
        neighbours (int): The number of neighbors to use for UMAP manifold approximation.
        hdb_min_cluster_size (int): The minimum number of samples in a cluster for HDBSCAN.
        hdb_min_samples (int, optional): The number of samples in a neighborhood for a point 
                                         to be considered a core point. Defaults to 2.

    Returns:
        tuple:
            clusters (np.ndarray): The cluster labels assigned by HDBSCAN.
            mask (list): A list of indices corresponding to non-noisy points, needed to later filter the corpus of sentences.
    """

    ## dimensionality reduction with UMAP

    # fallback if N of sentences is greater than or equal to the passed neighbours value
    if n_sents <= neighbours:
        neighbours = n_sents // 5
        print(f"\t\tInitial n_neighbors overridden to {neighbours}")

    reducer = umap.UMAP(
        n_neighbors=neighbours, 
        n_components=50, 
        metric="cosine", 
        random_state=42     # for replicability
        ) 
    
    reduced = reducer.fit_transform(transformed_sents)

    ## clustering with HDBSCAN

    # fallback if the N of sentences is greater than or equal to the passed neighbours value
    if n_sents <= hdb_min_cluster_size:
        hdb_min_cluster_size = 2
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size, 
                                min_samples=hdb_min_samples, 
                                allow_single_cluster=True, # in case the algorithm cannot distinguish more than one cluster, 
                                                           # this allows labeling with a n >= 0 instead of the noise label -1
                                )
    
    fit_clusterer = clusterer.fit(reduced)
    outliers = fit_clusterer.outlier_scores_    # extract outlier scores
    
    clusters = fit_clusterer.labels_    # extract label for each point

    # filter out the outliers and the noisy points
    mask = [i for i,(outlier,label) in enumerate(zip(outliers,clusters)) if outlier == 0 and label != -1]

    return clusters, mask


def visualize(clusters, list_of_keys):
    """
    Prints a detailed mapping of cluster IDs to their assigned sentence keys.

    This function creates a dictionary grouping all sentence keys by their assigned cluster ID 
    and prints the full list of keys for every cluster. This provides a raw view of which 
    specific sentences were grouped together.

    Args:
        clusters (np.ndarray): The array of cluster labels corresponding to the sentences.
        list_of_keys (list): A list of sentence identifiers (keys) to be grouped.
    """

    cluster_map = {}
    for i, key in enumerate(list_of_keys):
        cluster_id = clusters[i]
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(key)

    for key in cluster_map:
        print(key,"\n",cluster_map[key])


def visualize2(clusters, list_of_keys):
    """
    Prints a summarized distribution of semantic senses within each identified cluster.

    This function maps cluster IDs to their corresponding sentence keys. It assumes the 
    last character of each key represents a 'sense' identifier (refer to the generation of the 
    sentence codes in extract_wiki_sentences()). For each cluster, it counts the frequency of 
    each sense and prints a summary report. 

    Args:
        clusters (np.ndarray): The array of cluster labels corresponding to the sentences.
        list_of_keys (list): A list of sentence identifiers (keys), where the last character 
                             denotes the sense ID.
    """

    cluster_map = {}
    for i, key in enumerate(list_of_keys):
        cluster_id = clusters[i]
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(key)

    for label in cluster_map:
        print(f"\n\t\tIn cluster {label} ... ")
        retained_senses = []
        for key in cluster_map[label]:
            retained_senses.append(key[-1])
        retained_senses = Counter(retained_senses)
        for sense in retained_senses:
            print(f'\t\tsense {sense} appears {retained_senses[sense]} times.')
