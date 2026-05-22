import sys
import json
import spacy
import umap
import hdbscan
import numpy as np
from spacy.language import Language
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from preprocessing import preprocess
from static_models import static_avg_pooling, static_sentence_vector, exp_decay_pooling
from contextual_models import cont_avg_pooling, only_target_vector, cont_sentence_vector
from dim_reduction_and_clustering_and_viz import clustering, visualize, visualize2
from evaluation import evaluate


# ==============================================================================
# 1. SPACY CONFIGURATION & CUSTOM LEMMATIZATION
# ==============================================================================
# load the large English model which includes word vectors
nlp = spacy.load('en_core_web_lg')
_lemmatizer = nlp.get_pipe("lemmatizer")

# tweak spaCy's pipeline (refer to Section 3.4.2 of the thesis for details)
@Language.component("propn_lemma_normalizer")
def propn_lemma_normalizer(doc):
    """Re-lemmatize PROPN tokens as NOUNs via spaCy's own lemmatizer."""
    for token in doc:
        if token.pos_ == "PROPN":
            # Create a one-word doc with lowercased text, tagged as NOUN
            tmp = nlp.make_doc(token.lower_)
            tmp[0].pos_ = "NOUN"
            tmp[0].tag_ = "NNS" if token.tag_ == "NNPS" else "NN"
            _lemmatizer(tmp)
            token.lemma_ = tmp[0].lemma_
    return doc

# add the custom component to the end of the pipeline
nlp.add_pipe("propn_lemma_normalizer", last=True)


# ==============================================================================
# 2. DATA CONFIGURATION
# ==============================================================================

# dictionary mapping target words to their corresponding sense-specific JSON files
files_dict = {
    ## example usage:

    # "terminal" : ["terminal_computer.json", "terminal_airport.json", "terminal_electronic.json",
    #         "terminal_batter.json"],
    # ...
}


# iterate over each target word
for term in files_dict:

    target_word = term
    files = files_dict[term]

    # print header and sense overview for the current target word
    print("\n____________________________________________________________________________________")
    print(f'*{term.upper()}*')
    print("Senses:")
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
            # extract sense number from the last character of the first key (e.g., "u1s0" -> "0")
            for key in dictionary.keys():
                sense_number = key[-1]
                break
            print(f'\t{sense_number}: {file[:-4]} (containing {len(dictionary)} sentences.)')
    print("\n")


    # ==============================================================================
    # 3. EXPERIMENT LOOP: STATIC (N=0) vs CONTEXTUAL (N=1) MODELS
    # ==============================================================================
    
    for N in [0, 1]:
        
        # -------------------------------------------------------------------------
        # STATIC EMBEDDINGS PIPELINE
        # -------------------------------------------------------------------------
        
        if N == 0:

            # handle beta parameter for Exponential Decay via command line argments
            if len(sys.argv) == 2:
                beta = float(sys.argv[1])
                print(f"beta: {beta}")
            else:
                beta = 2.0

            # get the static vector for the target word from spaCy
            target_doc = nlp(target_word)
            target_v = target_doc.vector

            # verify the word exists in spaCy's vocabulary with vectors
            if not target_doc[0].has_vector:
                print("Error: no such word in spaCy's vocab.")
                continue

            else:
                # preprocess files: load sentences, keys, and ground truth labels
                list_of_keys, list_of_sents, ground_truth = preprocess(files)
                print("\tN of sentences: ",len(list_of_keys))

                # iterate over the three static models:
                # (I didn't use range(3) because sometimes you might want to exclude 0 and 2 to test betas on 1 and it's easy like this)
                for f in [0, 1, 2]:     
                    
                    if f == 0:
                        print("\n\tSTATIC AVERAGE TERM POOLING:")
                        # average vectors of nouns only
                        processed_sents = static_avg_pooling(list_of_sents, target_word, target_v, nlp)
                        
                    if f == 1:
                        print("\n\tEXPONENTIAL DECAY TERM POOLING:")
                        # weighted average of nouns based on semantic similarity to target
                        processed_sents = exp_decay_pooling(list_of_sents, target_word, target_v, nlp, beta=beta)

                    if f == 2:
                        print("\n\tSTATIC SENTENCE VECTOR:")   
                        # BASLINE, average of ALL tokens in the sentence
                        processed_sents = static_sentence_vector(list_of_sents, target_v, nlp)


                    # clustering (UMAP + HDBSCAN). Parameters: 10 neighbors, min cluster size 20
                    predictions, mask = clustering(processed_sents, len(list_of_keys), 10, 20)

                    # apply mask to filter out noise/outliers identified by HDBSCAN
                    masked_predictions = np.array(predictions)[mask]
                    masked_list_of_keys = np.array(list_of_keys)[mask]
                    masked_truth = np.array(ground_truth)[mask]

                    print("\t\tN of sentences with clear sense detected: ",len(masked_predictions))

                    # visualise cluster composition (sense distribution)
                    visualize2(masked_predictions, masked_list_of_keys)

                    # evaluate clustering performance
                    V, F = evaluate(masked_predictions, masked_truth)

                    # results table
                    print(f"\n\t\t{'V-measure':<20} | {'Paired F-score':<20} | Noise ")
                    print("\t\t","-" * 50)
                    print(f"\t\t{V:<20.4f} | {F:<20.4f} | {len(processed_sents) - len(mask)}")
        

        # -------------------------------------------------------------------------
        # CONTEXTUAL EMBEDDINGS PIPELINE
        # -------------------------------------------------------------------------

        if N == 1:

            # load transformer tokenizer and model (all-MiniLM-L6-v2) ...
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            #... to access the last hidden state ...
            model = AutoModel.from_pretrained(model_name)
            #... and to get the full-sentence output.
            full_sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # preprocess files: load sentences, keys, and ground truth labels
            list_of_keys, list_of_sents, ground_truth = preprocess(files)
            
            # iterate over the three contextual models:
            for f in range(3):

                if f == 0:
                    print("\n\tCONTEXTUAL AVERAGE TERM POOLING:")
                    # average vectors of nouns only
                    # (returns filtered lists due to alignment checks)
                    safe_keys, processed_sents, safe_truth = cont_avg_pooling(list_of_keys, list_of_sents, ground_truth, model, tokenizer, target_word, nlp)

                if f == 1:
                    print("\n\tONLY TARGET VECTOR:")
                    # only the target word's contextual embedding
                    # (returns filtered lists due to alignment checks)
                    safe_keys, processed_sents, safe_truth = only_target_vector(list_of_keys, list_of_sents, ground_truth, model, tokenizer, target_word, nlp)

                if f == 2:
                    print("\n\tSENTENCE VECTOR:")
                    # BASELINE, whole sentence encoding (average of all tokens)
                    processed_sents = cont_sentence_vector(list_of_sents, full_sentence_model)

                # clustering (UMAP + HDBSCAN). Parameters: 10 neighbors, min cluster size 20
                # note: len(processed_sents) is used here instead of len(list_of_keys) as contextual models may discard sentences internally                
                predictions, mask = clustering(processed_sents, len(processed_sents))
                masked_predictions = np.array(predictions)[mask]

                print("\t\tN of sentences with clear sense detected: ",len(masked_predictions))
                
                # apply mask to filter out noise/outliers identified by HDBSCAN but you need to select the correct lists for evaluation 
                # based on whether internal filtering occurred (it does not occurr in Cont. Sentence Vector)
                if f in [0,1]:
                    masked_list_of_keys = np.array(safe_keys)[mask]
                    masked_truth = np.array(safe_truth)[mask]

                    # visualise cluster composition (sense distribution)
                    visualize2(masked_predictions, masked_list_of_keys)
                    # evaluate clustering
                    V, F = evaluate(masked_predictions, masked_truth)

                else:
                    masked_list_of_keys = np.array(list_of_keys)[mask]
                    masked_truth = np.array(ground_truth)[mask]

                    # visualise cluster composition (sense distribution)
                    visualize2(masked_predictions, masked_list_of_keys)
                    # evaluate clustering
                    V, F = evaluate(masked_predictions, masked_truth)
                
                # result table
                print(f"\n\t\t{'V-measure':<20} | {'Paired F-score':<20} | Noise ")
                print("\t\t","-" * 50)
                print(f"\t\t{V:<20.4f} | {F:<20.4f} | {len(processed_sents) - len(mask)}")
