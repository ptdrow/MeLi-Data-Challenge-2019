#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:25:50 2019

@author: ptdrow
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import wording

class ptKMeans:
    def __init__(self, PATH, version, true_k, max_iter):
        self.PATH = PATH
        self.version = version
        #Iniciar variables para el par de idiomas
        self.vectorizer = [None] * 2
        self.X = [None] * 2
        self.vocabulary = [None] * 2
        self.model = [None] * 2
        self.selector = [None] * 2
        self.true_k = true_k
        self.max_iter = max_iter
        
    def load_data(self,lang = ''):
        print('Loading data '+ lang +'...')
        if lang == '':
            df = pd.read_csv( self.PATH + 'data/train.csv')
        else:
            df = pd.read_csv( self.PATH + 'data/train_' + lang + self.version + '.csv')
        
        print('Data loaded')
        return df
    
    def load_categories(self):
        return pd.read_csv (self.PATH + 'data/categories.csv', index_col = 0)
    
    def create_vocabulary(self, lang, p = 0.2):
        lang_train = self.load_data(lang)
        if lang == 'pt':
            i = 1
        else:
            i = 0
        print('Counting words for lang={0}:'.format(lang))
        stopwords = wording.import_stopwords(lang)
        all_words = list()
        
        j=0
        # Conteo de palabras por categoría
        for category in lang_train.category.unique():
            sentences = lang_train[lang_train.category == category]['title']
            words = list()
            
            for sentence in sentences:
                words.extend(wording.extract_words(sentence,stopwords))
            
            #Eliminar las palabras de baja ocurrencia
            cantidad_total = len(words)
            bigger = True
            words2 = list(words)
            while bigger:
                for word in np.unique(words2):
                    words2.remove(word)
                if len(words2) >= p * cantidad_total:
                    words = list(words2)
                else: bigger=False
                
            j+=1
            print('\r{0:.2f}%'.format(100*j/1588), end='')
            
            all_words.extend(words)
        print('\r100%  \n')
        words_list = list(np.unique(all_words))
        words_list.sort()
        self.vocabulary[i] = words_list
    
    def create_vectorizer(self,lang):
        train = self.load_data(lang)
        if lang == 'pt':
            i = 1
        else:
            i = 0
        self.vectorizer[i] = TfidfVectorizer(vocabulary = self.vocabulary[i])
        self.X[i] = self.vectorizer[i].fit_transform(list(train['title']))
        
    def create_model(self,lang):
        if lang == 'pt':
            i = 1
        else:
            i = 0
        self.model[i] = KMeans(n_clusters=self.true_k, init='k-means++', max_iter=self.max_iter, n_init=1)
        self.model[i].fit(self.X[i])
        
    def get_macro_categories(self,lang):
        if lang == 'pt':
            i = 1
        else:
            i = 0
        #load data
        train = self.load_data(lang)
        #predict macro category
        n = train.shape[0]
        
        macro_categories = list()
        j = 0
        for row in train.itertuples():
            macro_categories.append(self.model[i].predict(self.vectorizer[i].transform([row.title])))
            j +=1
            if j % 100 == 0:
                print('\r{0: .2f}%'.format(j*100/n), end ='')
        print('\n')
        return [number[0] for number in macro_categories]
        
    def create_selector(self, lang):
        if lang == 'pt':
            i = 1
        else:
            i = 0
        #load data
        train = self.load_data(lang)
        #predict macro category
        joined = pd.DataFrame()
        joined['category'] = train['category']
        joined['macro'] = self.get_macro_categories(lang)
        joined = joined.drop_duplicates()
        selector = list()
        categories = self.load_categories()
        for row in categories.itertuples():
            selections = list(joined[joined.category == row.Index].macro)
            selections.sort()
            selector.append(selections)
        self.selector[i] = selector
    
    def fit(self):
        langs = ['es','pt']
        for lang in langs:
            self.create_vocabulary(lang)
            self.create_vectorizer(lang)
            self.create_model(lang)
            self.create_selector(lang)


#subset by macro category
#get sub-categories
#train Bayes