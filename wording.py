#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 07:40:23 2019

@author: ptdrow
"""
import unidecode
import re

def tokenize_sentences(sentences):
    words = []
    i = 0
    for sentence in sentences:
        i +=1
        w = extract_words(sentence)
        words.extend(w)
        if i % 1000 == 0:
            print(i)
        
    words = sorted(list(set(words)))
    print('Tokenizing: Done')
    return words

def extract_words(sentence, ignore_words):
    sentence = unidecode.unidecode(sentence)
    sentence = re.sub('_', ' ', sentence)
    all_words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
    all_words = [characters for characters in all_words if is_word(characters)]
    all_words = [w.lower() for w in all_words]
    cleaned_words = [w for w in all_words if w not in ignore_words]
    while '' in cleaned_words:
        cleaned_words.remove('')
    
    return cleaned_words

def to_singular (words):
    singular_words = [re.sub('(s)$', '', w) if len(w)>4 else w for w in words ]
    return(singular_words)
    
def remove_digits(characters):
    pattern = '[0-9]'
    characters = [re.sub(pattern, '', i) for i in characters] 
    return characters

def is_word(characters):
    digit_count = sum([i.isdigit() for i in characters ])
    return (digit_count != len(characters))

def import_stopwords(lang):
    with open('data/stopwords_'+ lang +'.txt','r') as f:
        word_list = f.readlines()
        word_list.sort()
        return [re.sub('\\n$','',line) for line in word_list] 


"""def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag, dtype=int)"""