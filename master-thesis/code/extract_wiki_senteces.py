import wikipedia as wiki
import spacy
import json
import re

nlp = spacy.load("en_core_web_lg")



def print_files(sense_dict, target_word):
    """
    Extracts sentences containing a specific target noun from a dictionary of Wikipedia pages
    and saves them into separate JSON files.

    Args:
        sense_dict (dict): A dictionary where keys are sense names (e.g., 'terminal_computer') 
                           and values are Wikipedia page objects with a .content attribute.
        target_word (str): The word to search for (must be a noun) within the texts.
    
    Output:
        Creates one JSON file per key in 'sense_dict' in the current directory.
    """

    # initialize sense label
    sense = -1

    for name,file in sense_dict.items():

        # create string to name the files
        filename = name + ".json"

        # read the content of each page
        doc = nlp(file.content)

        # find the sentences where the target is present and it is a noun
        doc_dict = {}
        counter = 0
        sense += 1
        for sent in doc.sents:
            for token in sent:
                if token.lemma_.lower() == target_word and token.pos_ == "NOUN":
                    counter += 1

                    # create the unique key with index and sense label
                    key = "u"+str(counter)+"s"+str(sense)

                    # clean sentences from unwanted char and save it 
                    cleaned_sent = re.sub(r'\s+', ' ', str(sent).replace("\u21b5", "")).strip()
                    doc_dict[key] = cleaned_sent
                    break
        
        # print the dict to a file
        with open(filename, "w", encoding="utf-8")as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)



# An example usage with "terminal":

computer_terminal = wiki.page(title="Computer terminal")
airport_terminal = wiki.page(title="Airport terminal")
electronic_terminal = wiki.page(title="Terminal (electronics)")
battery_terminal = wiki.page(title="Battery terminal")

sense_dict = {
            "terminal_computer" : computer_terminal,
            "terminal_airport" : airport_terminal,
            "terminal_electronic": electronic_terminal,
            "terminal_batter" : battery_terminal,
              }

print_files(sense_dict, target_word="terminal")
