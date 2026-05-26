import torch
import numpy as np
from collections import defaultdict


def cont_avg_pooling(list_of_keys, list_of_sentences, ground_truth, model, tokenizer, target_word, nlp):
    """
    Generates context-aware sentence embeddings by averaging the vectors of context nouns. 
    Uses a transformer-based sentence-level contextual embedding model and spaCy alignment.

    Args:
        list_of_keys (list): The sentence codes returned by preprocess().
        list_of_sentences (list): The input sentences.
        ground_truth (list): Corresponding ground truth labels for filtering.
        model: The pre-trained contextual embedding model.
        tokenizer: The model's tokenizer.
        target_word (str): The lemma of the target word to isolate.
        nlp: The loaded spaCy NLP model for POS tagging and lemmatization.

    Returns:
            list_of_keys (list): Filtered list of keys (removed masked entries).
            X (np.ndarray): Array of computed context embeddings.
            ground_truth (list): Filtered list of ground truth labels.
    """

    embeddings = []
    mask = []   # to later filter out sentences without target noun that couldn't be processed

    for i, sentence in enumerate(list_of_sentences):

        # tokenize sentence with the contextual model's tokenizer
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

        # access the last hidden state to get token-level vectors
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0] 

        # align tokens to words in the sentence
        word_ids = inputs.word_ids()  # maps each token to its word index (None for special tokens)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # gather subtokens and their vectors
        word_to_token_indices = defaultdict(list)
        word_to_embedding_indices = defaultdict(list)
        for t,id,vector in zip(tokens,word_ids,token_embeddings):
            if id != None:
                word_to_token_indices[id].append(t)
                word_to_embedding_indices[id].append(vector)

        # join sub-word tokens and vectors 
        for id in word_to_token_indices:
            # form full words
            list_of_t = word_to_token_indices[id]
            full_word = "".join([t.replace("##", "") for t in list_of_t])
            word_to_token_indices[id] = full_word

            # average over subwords embeddings
            stacked_vectors = torch.stack(word_to_embedding_indices[id])
            mean_vector = torch.mean(stacked_vectors, dim=0)
            word_to_embedding_indices[id] = mean_vector

        # initialize target and context embedding
        target_embedding = None
        other_nouns = []

        # ALIGNMENT BW SPACY AND EMBEDDING MODEL
        # sort by key to ensure the order matches the sentence flow
        sorted_word_ids = sorted(word_to_token_indices.keys())
        trf_word_list = [word_to_token_indices[w_id] for w_id in sorted_word_ids]
        pseudo_sentence = " ".join(trf_word_list)

        # pass the copy of the sentence to the spacy pipeline
        doc = nlp(pseudo_sentence)

        use_fallback = 0    # flag for failed alignment

        # first we check if there are as many tokens in spacy as there are in the transformer model
        alignment_valid = (len(doc) == len(sorted_word_ids))
        if not alignment_valid:
            use_fallback = 1

        # this is to check if there are still mismatches between spacy and transformer tokens
        for token in doc:
            if token.text != word_to_token_indices[token.i]:
                use_fallback = 1
                break

        # reset the variable to avoid scope leakage
        token = None

        if not use_fallback:
            for token in doc:
                # select embeddings
                if token.pos_ == "NOUN":
                    # retrieve the target's embedding to verify presence and for later fallback if no nouns are found
                    if token.lemma_ == target_word:
                        if target_embedding is None:
                            target_embedding = word_to_embedding_indices[token.i]
                    
                    # get all context nouns' embeddings
                    else:
                        other_nouns.append(word_to_embedding_indices[token.i])

        
        # if something failed, force spacy to read one of the transformer's tokens at a time
        if use_fallback:

            for idx in sorted_word_ids:
                t = word_to_token_indices[idx]
                tmp = nlp(t)
                pos = tmp[0].pos_
                lemma = tmp[0].lemma_
                
            # then proceed as before
                if pos == "NOUN":
                    if lemma == target_word: 
                        if target_embedding is None:
                            target_embedding = word_to_embedding_indices[idx]
                    
                    else:
                        other_nouns.append(word_to_embedding_indices[idx])
                    

        if target_embedding is None:      # this should not happen, but sometimes it does ...
            mask.append(i)      #... so we keep track of it
            continue

        # average all other context noun embeddings (if any)
        if other_nouns:
            other_embedding = torch.stack(other_nouns).mean(dim=0)
        else:
            other_embedding = None

        # form the final representation by averaging noun embeddings
        if other_embedding is not None:
            context_embedding = other_embedding
        
        # if no context nouns have been found, use the target's embedding
        elif target_embedding is not None:
            context_embedding = target_embedding
            print(f"\tNo context in sentence {list_of_keys[i]}")
        
        # store embedding
        embeddings.append(context_embedding.numpy())

    # fix lists according to the mask to avoid mismatches
    list_of_keys = [value for i, value in enumerate(list_of_keys) if i not in mask]
    ground_truth = [value for i, value in enumerate(ground_truth) if i not in mask]

    X = np.array(embeddings)

    if mask:
        print(f'\n\tNo target word found in sentences: {mask}')

    return list_of_keys, X, ground_truth


def cont_sentence_vector(list_of_sentences, model):
    """
    Retrieves the sentence embedding outputted by the model (transformer-based sentence-level contextual embedding model).

    Args:
        list_of_sentences (list): The input sentences to encode.
        model: A model object possessing an '.encode()' method.

    Returns:
        np.ndarray: An array of sentence embeddings, one per input sentence.
    """
    
    embeddings_list = [model.encode(sentence) for sentence in list_of_sentences]
    
    return np.array(embeddings_list)


def only_target_vector(list_of_keys, list_of_sentences, ground_truth, model, tokenizer, target_word, nlp):

    """
    Extracts and returns only the embedding of the first occurrence of the target noun, 
    ignoring all other context words.

    Args:
        list_of_keys (list): The sentence codes returned by preprocess().
        list_of_sentences (list): The input sentences to process.
        ground_truth (list): Corresponding ground truth labels for filtering.
        model: The pre-trained contextual embedding model.
        tokenizer: The model's tokenizer.
        target_word (str): The lemma of the target word to extract.
        nlp: The loaded spaCy NLP model.

    Returns:
        tuple:
            list_of_keys (list): Filtered list of keys.
            X (np.ndarray): Array of target word embeddings only.
            ground_truth (list): Filtered list of ground truth labels.
    """

    embeddings = []
    mask = []   # to later filter out sentences without target noun that couldn't be processed

    for i, sentence in enumerate(list_of_sentences):

        # tokenize sentence with the contextual model's tokenizer
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

        # access the last hidden state to get token-level embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]

        # align tokens to words in the sentence
        word_ids = inputs.word_ids()  # Maps each token to its word index (None for special tokens)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # gather subtokens and their vectors
        word_to_token_indices = defaultdict(list)
        word_to_embedding_indices = defaultdict(list)
        for t,id,vector in zip(tokens,word_ids,token_embeddings):
            if id != None:
                word_to_token_indices[id].append(t)
                word_to_embedding_indices[id].append(vector)
        
        # join sub-word tokens and vectors 
        for id in word_to_token_indices:
            # form full words
            list_of_t = word_to_token_indices[id]
            full_word = "".join([t.replace("##", "") for t in list_of_t])
            word_to_token_indices[id] = full_word

            # average over subwords embeddings
            stacked_vectors = torch.stack(word_to_embedding_indices[id])
            mean_vector = torch.mean(stacked_vectors, dim=0)
            word_to_embedding_indices[id] = mean_vector


        # initialize target embedding
        target_embedding = None

        # sort by key to ensure the order matches the sentence flow
        sorted_word_ids = sorted(word_to_token_indices.keys())
        trf_word_list = [word_to_token_indices[w_id] for w_id in sorted_word_ids]
        pseudo_sentence = " ".join(trf_word_list)

        # pass the copy of the sentence to the spacy pipeline
        doc = nlp(pseudo_sentence)

        missmatch = 1   # signals failure

        # check if there are as many spacy token as there are transfomer tokens
        if len(doc) == len(sorted_word_ids):
            for token in doc:
                # isolate the target word and check if the vector with the same index returned by the transfomer actually IS linked to the right word
                if token.lemma_ == target_word and token.pos_ == "NOUN" and token.text == word_to_token_indices[token.i]:
                    target_embedding = word_to_embedding_indices[token.i]
                    missmatch = 0
                    break   # stop at the first occurrence to represent the sense in this sentence
        
        # if something failed, force spacy to read one of the transformer tokens at a time
        if missmatch == 1:
            for idx in sorted_word_ids:
                token = word_to_token_indices[idx]
                tmp = nlp(token)
                pos = tmp[0].pos_
                lemma = tmp[0].lemma_
                
                if lemma == target_word and pos == "NOUN":
                    target_embedding = word_to_embedding_indices[idx]
                    break


        if target_embedding is None: # this should not happen, but we keep track
            mask.append(i)
            continue

        # store embedding
        embeddings.append(target_embedding.numpy())

    # fix lists according to mask to avoid mismatches
    list_of_keys = [value for i, value in enumerate(list_of_keys) if i not in mask]
    ground_truth = [value for i, value in enumerate(ground_truth) if i not in mask]

    X = np.array(embeddings)

    if mask:
        print(f'\tNo target word found in sentences: {mask}')

    return list_of_keys, X, ground_truth

