Welcome to my Master Thesis: _Make it make sense_ - An exploratory study on Word Sense Induction for the terminology industry.

_This thesis explored various approaches to Word Sense Induction within industrial technical and noisy data. 
Its two-level architecture first hypothesises that noun-based sentence embeddings - static and contextual - are 
more efficient than full-sentence versions, then evaluates four methods of leveraging these embeddings against 
two baselines._

For more information, read the Abstract or the full study.

Here, you'll also find the full analysis of the results on both datasets, and a visualization of the variance of two hyperparameters fine-tuned during the project.

This directory gathers all information needed to replicate the results presented in the thesis on the Wikipedia dataset, which you can find here together with a list of links to the webpages from where the data was extracted (wikipedia_sources). The custom datatset used in the study is private and cannot be shared.

In the "code" sub-directory you'll find all the code needed to run the pipeline I wrote. This code is made public under the CC BY-NC license (visit this website for an explanation of the license: https://creativecommons.org/licenses/by-nc/4.0/).
The coding environment I used is the following:
- Python 3.11.9
- SpaCy 3.8.11
- Scikit-learn 1.8.0
- Numpy 1.26.4
- Pytorch 2.10.0
- Transformers 4.49.0
- Sentence-transformers 5.2.2
- Umap-learn 0.5.11
- Hdbscan 0.8.41
- Wikipedia 1.4.0
