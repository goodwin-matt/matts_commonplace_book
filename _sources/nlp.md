# NLP

## General NLP Techniques

A lot of this comes from a Real Python podcast called "Natural Language Processing and 
How ML Models Understand Text". I felt like it did a great job going through at a high level common 
NLP techniques and how they build on each other, leading up to transformers.

Really the basic first problem to solve is how translate words to numbers or vectors so we can 
do math. Here are some of the most basic and common approaches:

1. `Binary Vectorization` - the most basic approach here would be to create a table where each column represents a word in your corpus of 
documents. If each row represents a document then we would put a "1" in the column of a word that 
appears in the document. Here are some problem and fixes to be aware of:
   1. If we encode each word as a column this can lead to a lot of columns and can lead to a lot of 
   noise in a model so maybe we subset the number of words to just the important words by some definition.
   2. Another problem is that we might have words that mean the same thing "cats" and "cat". 
   Stemming and lemmatization are two techniques that help combine these words. Both have the same goal 
   to reduce the words to their base word. Stemming is a more blunt where it removes endings like "ing" 
   or "ed" whereas lemmatization is more expensive where it tries to apply language rules to get at base words.
   3. The next thought is how do we encode context? A simple technique is to use `n-grams`, or essentially 
   creating more columns that combine two, three,..n words together and encoding the presence of that 
   phrase in the document or not.
2. `Count Vectorization` - instead of using binary vectors, we can use count vectors to count the number of times a word appears 
in the document in order to encode the document better around what topics exist in the document.
   1. The most common words "and", "the" will appear most so we remove these stop words.
3. `TF-IDF` - Instead of a simple count vector we may want to weight words according to how 
frequently they appear in our documents across the corpus. In particular, we would want to upweight words
that are more unique to our document and downweight words that are more common across all documents.
   1. Documents are going to be different lengths so you need to apply `normalization` where you 
   weight each word in a document in a way that allows you to compare across all documents fairly.
4. `word2vec` - this technique uses neural nets to encode the context and meaning of a word. The idea
is that we one-hot encode each word and use one as input and one as the response variable. The hidden 
layer of that neural net is then used as the embedding for these different words. In table form we 
would have each row being a word, and each column being some encoding of the word in our high dimension
space. 

What Python packages are available to implement these techniques?
- sklearn
- nltk
  - wordnet (find synonyms)
  - stemming and lemmatization
- spaCy
- Gensim
- fastText




# Transformers

http://jalammar.github.io/illustrated-transformer/




