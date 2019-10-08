#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:02:13 2019

Customized by Pedro Villarroel from Peter Norvig's spell.py

Words list retrieved from FrequencyWords by Hermit Dave (@hermitdave on github)
"""
"""
#es_WORDS = pd.read_csv('Documents/Machine_Learning/MeLi_Classifier/data/es_50k.txt',
es_WORDS = pd.read_csv('data/es_50k.txt',
                            sep=' ', header = None, index_col = 0, names = ['freq'],
                            dtype={'freq' : np.uint32})

#pt_WORDS = pd.read_csv('Documents/Machine_Learning/MeLi_Classifier/data/pt_br_50k.txt',
pt_WORDS = pd.read_csv('data/pt_br_50k.txt',
                            sep=' ', header = None, index_col = 0, names = ['freq'],
                            dtype={'freq' : np.uint32})
"""

class corrector:
    def __init__(self, contador):
        self.N = sum(list(contador.values()))
        self.freq = contador
                
    def P(self, word): 
            #Probabilidad de la palabra
        return self.freq[word] / self.N

    def correction(self, word): 
        #Correction más probable
        return max(self.candidates(word), key=self.P)
    
    def candidates(self, word): 
        #"Generate possible spelling corrections for word."
        return (self.known([word]) or 
                self.known(self.edits1(word)) or 
                self.known(self.edits2(word)) or 
                [word])

    def known(self, words): 
        #"The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.freq)

    def edits1(self, word):
        #"All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        #"All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))