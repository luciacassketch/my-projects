import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cossim


def static_avg_pooling(sentences: list, target_word, target_v, nlp):
    """
    Generates sentence embeddings by averaging the vectors of all nouns in the sentence,
    excluding the target word itself.

    Args:
        sentences (list): A list of sentence strings.
        target_word (str): The lemma of the target word.
        target_v (np.ndarray): The vector representation of the target word (fallback value).
        nlp: The loaded spaCy NLP model.

    Returns:
        np.ndarray: An array of embedding vectors, one per sentence.
    """

    transformed_sents = []
    mask = []

    for y,s in enumerate(sentences):

        # read the sentence with the spacy pipeline
        doc = nlp(s)

        embeddings = []

        # find all nouns (excluding the target) as save their vector
        for t in doc:
            if t.pos_ == "NOUN" and t.lemma_.lower() != target_word:
                embeddings.append(t.vector)

        # fallback for when no nouns are found
        if not embeddings:
            transformed_sents.append(target_v)
            mask.append(y)
            continue

        embeddings = np.array(embeddings)

        # form the sentence embedding by averaging all noun vectors
        new_sent = np.mean(embeddings, axis = 0)
        transformed_sents.append(new_sent)

    if mask:
        print(f'\tNo context found in sentences: {mask}')

    return np.array(transformed_sents)



def static_sentence_vector(sentences: list, target_v, nlp):
    """
    Generates sentence embeddings by averaging the vectors of ALL tokens in the sentence.

    Args:
        sentences (list): A list of sentence strings.
        target_v (np.ndarray): The vector representation of the target word (fallback value).
        nlp: The loaded spaCy NLP model.

    Returns:
        np.ndarray: An array of embedding vectors, one per sentence.
    """

    transformed_sents = []
    mask = []

    for y,s in enumerate(sentences):

        # read the sentence with the spacy pipeline
        doc = nlp(s)
        embeddings = []

        # retrieve all sentence vectors
        for t in doc:
            embeddings.append(t.vector)

        # fallback for when no words are found 
        if not embeddings: 
            transformed_sents.append(target_v)
            mask.append(y)
            continue

        embeddings = np.array(embeddings)

        # shape sentence representation by averaging all vectors
        new_sent = np.mean(embeddings, axis = 0)
        transformed_sents.append(new_sent)

    if mask:
        print(f'\tNo context found in sentences: {mask}')

    return np.array(transformed_sents)



def exp_decay_pooling(sentences: list, target_word, target_v, nlp, alpha = 0.6):
    """
    Generates sentence embeddings using a weighted average based on semantic 
    similarity to the target.

    Args:
        sentences (list): A list of sentence strings.
        target_word (str): The lemma of the target word.
        target_v (np.ndarray): The vector representation of the target word.
        nlp: The loaded spaCy NLP model.
        alpha (float, optional): The scaling factor for the exponential weighting. 
                                Defaults to 0.6.

    Returns:
        np.ndarray: An array of weighted embedding vectors, one per sentence.
    """

    transformed_sents = []
    mask = []

    for y,s in enumerate(sentences):

        # read the sentence with the spacy pipeline
        doc = nlp(s)

        embeddings = []
        
        # find all nouns and retrieve their vectors
        for t in doc:
            if t.pos_ == "NOUN" and t.lemma_.lower() != target_word:
                embeddings.append(t.vector)

        # fallback for when no words are found 
        if not embeddings:
            transformed_sents.append(target_v)
            mask.append(y)
            continue

        embeddings = np.array(embeddings)

        # calculate similarity of every noun embedding with that of the target
        similarity_matrix = cossim(embeddings, target_v.reshape(1, -1))
        similarities = similarity_matrix.flatten()

        # calculate weights with similarity-based exponential function
        weights = np.exp(alpha * np.array(similarities)) 

        # normalizing weights to sum up to 1 and so the output is a weighted average not sum
        total_weight = np.sum(weights)
        weights = weights / total_weight

        # compute weighted average of embeddings
        weighted_feature = np.zeros(len(target_v))
        for i in range(len(weights)):
            weighted_feature += weights[i] * embeddings[i]

        transformed_sents.append(weighted_feature)

    if mask:
        print(f'\tNo context found in sentences: {mask}')
    
    return np.array(transformed_sents)
