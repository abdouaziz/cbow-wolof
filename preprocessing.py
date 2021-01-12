import re
import nltk
import numpy as np 
from nltk import word_tokenize
import emoji


with open (file="sentences.txt" , encoding='utf8') as f:
    data = f.read()

def tokenize(data=data):
    data=re.sub(r'[;,?!-]','.' , data) 
    data = word_tokenize(data)
    data = [ch.lower() for ch in data if ch.isalpha() or ch == '.' or emoji.get_emoji_regexp().search(ch) ]
    return data 

def get_windows(words , C):
    i=C
    while i <len(words) -C:
        center_words = words[i]
        context_words = words[(i-C) : i] + words[(i+1):(i+C+1)]
        yield  center_words , context_words 
        i +=1 

def get_dict(data):
    words = sorted(list(set(data)))
    idx = 0
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word 

def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector 

def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors 

def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

 
