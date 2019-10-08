#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:41:10 2019

@author: Pedro Villarroel @ptdrow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
import re
import os
from random import choices
import wording


version = '052'
PATH = ''
    
def load_data(lang = ''):
    print('Loading data '+ lang +'...')
    if lang == '':
        df = pd.read_csv( PATH + 'data/train.csv')
    else:
        df = pd.read_csv( PATH + 'data/train_' + lang + version + '.csv')
    
    print('Data loaded')
    return df

def load_categories():
    return pd.read_csv (PATH + 'data/categories.csv', index_col = 0)

def load_validation(fold):
    return pd.read_csv (PATH + 'data/validation'+ str(fold) + version + '.csv')

def load_test():
    return pd.read_csv (PATH + 'data/test.csv')

def calc_p(data):
    frecuencias = data['freq']
    total = sum(frecuencias)
    data['p'] = frecuencias / total

def assign_num_id(data):
    data['id'] = list(range(data.shape[0]))
    
def calc_categories_info(df):
    df_categories = pd.DataFrame(df.category.value_counts())
    df_categories.rename(columns={'category':'freq'}, 
                      inplace=True)
    df_categories = df_categories.sort_index()
    calc_p(df_categories)
    assign_num_id(df_categories)
    return df_categories

def subset_data(df, train_r, test_r, lang = 'all'):
    if lang == 'all':
        print('Subsetting by language...')
        categories = calc_categories_info(df)
        categories.to_csv(PATH + 'data/categories.csv')
        
        df_es = df[df.language =='spanish']
        df_pt = df[df.language == 'portuguese']
        
        test_es = subset_data(df_es, train_r, test_r, 'es')
        test_pt = subset_data(df_pt, train_r, test_r, 'pt')
        
        test_set = pd.concat([test_es,test_pt])
        test_set.to_csv(PATH + 'data/test'+version+'.csv',index=False)

    else:
        print('Splitting languange ' + lang)
        lang_train, lang_test = train_test_split(df, 
                                                 train_size=int(train_r*len(df)),
                                                 test_size=int(test_r*len(df)),
                                                 random_state=42,
                                                 stratify=df.category)
        
        lang_train = lang_train.sort_values('category')
        lang_train.to_csv(PATH + 'data/train_' + lang + version + '.csv',
                          index=False)
        
        return lang_test

def kfold(train_df,exclude,k=3):  
    j = range(k)
    n = train_df.shape[0]
    sequence = [j[i % k] for i in range(n)]
    
    validation_subset = train_df[[sequence[i] == exclude for i in range(n)]]
    train_subset = train_df[[sequence[i] != exclude for i in range(n)]]
    
    return train_subset, validation_subset

def load_vocabulary(fold,lang):
    return pd.read_csv (PATH + 'vocabulary/vocabulary'+ str(fold) + '_' + lang + version + '.csv',index_col = 0)
  
def load_vocabulary_p(fold,lang):
    return np.load(PATH + 'vocabulary/vocabulary_p'+ str(fold) + '_' + lang + version +'.npy')

def train_fold(train_list, validation, fold, p):
    categories = load_categories()
    langs = ['es','pt']
    i = 0
    for lang in langs:
        count_words(train_list[i],lang, p)
        vocabulary = create_vocabulary(lang)
        vocabulary.to_csv(PATH + 'vocabulary/vocabulary'+ str(fold) + '_' + lang + version + '.csv')
        vocabulary_freq = word_cat_freq(lang, vocabulary, categories)
        vocabulary_p = word_cat_p(vocabulary.shape[0], vocabulary_freq, categories, 0.0625)
        np.save(PATH + 'vocabulary/vocabulary_p'+ str(fold) + '_' + lang + version +'.npy', vocabulary_p)
        i+=1
    del vocabulary, vocabulary_freq, vocabulary_p
    print('Creating frequency matrices: Done')
    #agregar chequeos para buscar archivos
    result = compute_result(validation,
                            [load_vocabulary(fold,'es'),
                             load_vocabulary(fold,'pt')],
                            [load_vocabulary_p(fold,'es'),
                             load_vocabulary_p(fold,'pt')])
    fold_cat_accuracy = calc_accuracy(result)
    fold_accuracy = sum(result.Prediction == result.true_category)/result.shape[0]
    fold_balanced_accuracy = metrics.balanced_accuracy_score(result.true_category,result.Prediction)
    return [fold_cat_accuracy, fold_accuracy, fold_balanced_accuracy]

def count_words(lang_train, lang, p, save_categories = True):
    print('Counting words for lang={0}:'.format(lang))
    stopwords = wording.import_stopwords(lang)
    all_words = list()
    category_list=list()
    
    j=0
    for category in lang_train.category.unique():
        sentences = lang_train[lang_train.category == category]['title']
        words = list()
        # Conteo de palabras por categoría
        i = 0
        for sentence in sentences:
            i += 1
            words.extend(wording.extract_words(sentence,stopwords))
        
        #Eliminar las palabras de baja ocurrencia
        cantidad_total = len(words)
    
        k=0
        bigger = True
            
        words2 = list(words)
        while bigger:
            for word in np.unique(words2):
                words2.remove(word)
            if len(words2) >= p * cantidad_total:
                k += 1
                words = list(words2)
            else: bigger=False
            
        this_vocabulary = Counter(words).most_common()
        if save_categories:
            with open(PATH + 'vocabulary/'+ lang +'/'+ category + '.txt', 'w') as f:
                for word, n in this_vocabulary:
                    f.write("%s,%s\n" % (word,n))
        j+=1
        print('\r{0:.2f}%'.format(100*j/1588), end='')
        category_list.append(category)
        
        all_words.extend(words)
    print('\n')
    category_list.sort()
    
    return(Counter(all_words))

def create_vocabulary(lang):
    vocabulary = pd.DataFrame(columns = ['freq'])
    i=0
    print('Creating vocabulary for lang={0}:'.format(lang))

    for file in os.listdir(PATH + 'vocabulary/'+ lang):
        vocabulary = vocabulary.add(pd.read_csv (PATH + 'vocabulary/'+lang+'/'+file,
                                  index_col = 0, names = ['freq']), fill_value=0)
        i+=1
        if i %10 == 0:
            print('\r{0:.2f}%'.format(100*i/1588), end='')
    vocabulary = vocabulary.dropna()
    if type(vocabulary.index[0]) == float:
        vocabulary = vocabulary.drop(vocabulary.index[0])
    calc_p(vocabulary)
    assign_num_id(vocabulary)
    print('\n')
    return(vocabulary)

def word_cat_freq(lang, vocabulary, categories):
    # Matrix de Categorias X Palabras
    # Indica la cantidad de veces que aparece cada palabra en cada categoría
    vocabulary_freq = np.zeros(shape=(categories.shape[0],vocabulary.shape[0]), dtype='uint32')
    
    for file in os.listdir(PATH + 'vocabulary/'+lang):
        category = re.sub('.txt', '', file)
        category_vocabulary = pd.read_csv (PATH + 'vocabulary/'+lang+'/'+file,
                                  names = ['Index','freq'])
        category_vocabulary = category_vocabulary.dropna()
        category_vocabulary = category_vocabulary.set_index('Index')
        category_id = categories.loc[category,'id']
        for row in category_vocabulary.itertuples():
            word = row.Index
            freq = row.freq
            #Find id
            word_id = vocabulary.loc[word,'id']
            vocabulary_freq[category_id,word_id] += freq
    return(vocabulary_freq)

def word_cat_p(n_vocabulary, vocabulary_freq, categories, alfa = 1):
    vocabulary_p = np.zeros(shape=vocabulary_freq.shape)
    
    for category_id in categories['id']:
        vocabulary_p[category_id,] = (vocabulary_freq[category_id,]+alfa)/(np.sum(vocabulary_freq[category_id,]+alfa * n_vocabulary))

    return(vocabulary_p)
    
def reduce_variance(data, r):
    mean = np.average(data)
    return (mean + (data - mean) * r)

def decider0(words_ids, vocabulary_p, category_dict, categories_p):
    if len(words_ids) > 0:
        decider = np.prod(vocabulary_p[:, words_ids],axis=1) * categories_p
        return(category_dict[np.argmax(decider)])
    else: return('UNDECIDED')

def classify_title(row, vocabulary, vocabulary_p, 
                   decider,stopwords, 
                   category_dict, categories_p):
    
    words = np.unique(wording.extract_words(row.title,stopwords))
    words_ids = list()
    for word in words:
        if word in vocabulary.index:
            words_ids.append(vocabulary.loc[word,'id'])
    return decider(words_ids, vocabulary_p, category_dict, categories_p)

def compute_result(test_set,vocabularies,vocabularies_p, categories_p = 1, final = False):
    stopwords_es = wording.import_stopwords('es')
    stopwords_pt = wording.import_stopwords('pt')
    result = pd.DataFrame(index=test_set.index)
    n = test_set.shape[0]
    result['Prediction'] = 'UNDECIDED'
    if not final:
        result['true_category'] = 'UNKNOWN'
    i=0
    categories = load_categories()
    category_dict = dict(zip(categories['id'], categories.index))
    for row in test_set.itertuples():
        i+=1
        if row.language == 'spanish':
            result.at[row.Index,'Prediction']=classify_title(row,
                                                             vocabularies[0],
                                                             vocabularies_p[0],
                                                             decider0,
                                                             stopwords_es,
                                                             category_dict,
                                                             categories_p)
        else:
            result.at[row.Index,'Prediction']=classify_title(row,
                                                             vocabularies[1],
                                                             vocabularies_p[1],
                                                             decider0,
                                                             stopwords_pt,
                                                             category_dict,
                                                             categories_p)
        if not final:
            result.at[row.Index,'true_category']=row.category
        if i % 1500 == 0:
            print('\rComputing results {0:.2f}%'.format(100*i/n), end='')
    print('\n')
    return result

def calc_accuracy(result):
    categories = load_categories()
    category_accuracy = pd.DataFrame(index=list(categories.index))
    
    category_accuracy['true_+'] = 0
    category_accuracy['false_+'] = 0
    category_accuracy['true_-'] = 0
    category_accuracy['false_-'] = 0
    category_accuracy['accuracy'] = 0.
    category_accuracy['precision'] = 0.
    category_accuracy['recall'] = 0.
    category_accuracy['total'] = 0.
    
    i=0
    for category in np.unique(result.true_category):
        category_result = result[result.true_category == category]
        category_prediction = result[result['Prediction'] == category]
        i+=1
        tp = sum(category_result['Prediction'] == category_result.true_category)
        fn = sum(category_result['Prediction'] != category_result.true_category)
        fp = sum(category_prediction['Prediction'] != category_prediction.true_category)
        tn = result.shape[0]-tp-fn-fp
        category_accuracy.loc[category,'true_+'] = tp
        category_accuracy.loc[category,'false_-'] = fn
        category_accuracy.loc[category,'false_+'] = fp
        category_accuracy.loc[category,'true_-'] = tn
        category_accuracy.loc[category,'accuracy'] = (tp + tn)/(tp + tn + fp + fn)
        if tp != 0:
            category_accuracy.loc[category,'precision'] = tp/(tp + fp)
        category_accuracy.loc[category,'recall'] = tp/(tp + fn)
        category_accuracy.loc[category,'total'] = tp + fn
        if i % 100 == 0:
            print('\rComputing accuracy {0:.2f}%'.format(100*i/1588), end='')
    print('\n')
    return category_accuracy

def train_all(folds,k, p):
    train_es = load_data(lang = 'es')
    train_pt = load_data(lang = 'pt')
    
    all_fold_accuracies = list()
    for fold in range(folds):
        print('Training fold: {0}'.format(fold))
        fold_train_es, validation_es = kfold(train_es, fold, k)
        fold_train_pt, validation_pt = kfold(train_pt, fold, k)
        
        validation = pd.concat([validation_es,validation_pt])
        validation.to_csv(PATH + 'data/validation'+ str(fold) + version + '.csv',
                          index=False)
        
        fold_accuracies = train_fold([fold_train_es,fold_train_pt],
                                     validation,
                                     fold,
                                     p)

        all_fold_accuracies.append(fold_accuracies)
    return all_fold_accuracies

def checking_results(folds, previous_fold_accuracies,r):
    post_fold_accuracies = list()
    for fold in range(folds):
        fn = np.array(all_fold_accuracies[fold][0]['false_-'])
        fp = np.array(all_fold_accuracies[fold][0]['false_+'])
        categories_p =  reduce_variance(fn,r)/reduce_variance(fp,r)
        result = compute_result(load_validation(fold),
                                      [load_vocabulary(fold,'es'),
                                      load_vocabulary(fold,'pt')],
                                      [load_vocabulary_p(fold,'es'),
                                      load_vocabulary_p(fold,'pt')],
                                       categories_p)
        fold_cat_accuracy = calc_accuracy(result)
        fold_accuracy = sum(result.Prediction == result.true_category)/result.shape[0]
        fold_balanced_accuracy = metrics.balanced_accuracy_score(result.true_category,result.Prediction)
        post_fold_accuracies.append([fold_cat_accuracy, fold_accuracy, fold_balanced_accuracy])
        
    return post_fold_accuracies


def final_results(folds, fold_accuracies,r):
    categories = load_categories()
    category_list = list(categories.index)
    results = list()
    for fold in range(folds):
        print('Testing with fold: {0}'.format(fold))
        fn = np.array(fold_accuracies[fold][0]['false_-'])
        fp = np.array(fold_accuracies[fold][0]['false_+'])
        categories_p =  reduce_variance(fn,r)/reduce_variance(fp,r)
        results.append(compute_result(load_test(),
                                [load_vocabulary(fold,'es'),load_vocabulary(fold,'pt')],
                                [load_vocabulary_p(fold,'es'),load_vocabulary_p(fold,'pt')],
                                categories_p,
                                final = True))
    
    n = results[0].shape[0]
    final_results = list()
    print('Voting')
    for i in range(n):
        poll = list()
        poll.append(results[0].iloc[i, 0])
        poll.append(results[1].iloc[i, 0])
        poll.append(results[2].iloc[i, 0])
        poll.sort()
        if poll[1] != 'UNDECIDED':
            final_results.append(poll[1])
        elif poll[0] != 'UNDECIDED':
            final_results.append(poll[0])
        elif poll[2] != 'UNDECIDED':
            final_results.append(poll[2])
        else:
            final_results.append(choices(category_list, np.array(categories['p']))[0])
        if i % 1500 == 0:
            print('\r{0:.2f}%'.format(100*i/n), end='')
    print('\n')
    print('Testing: Done')
    return final_results
 
subset_data(load_data(), train_r = 0.04, test_r = 0.0075)
all_fold_accuracies = train_all(3, 3, 0.6)
MeLi_result = final_results(3, all_fold_accuracies, 0.8)

MeLi_result2 = pd.DataFrame()
MeLi_result2['id'] = range(len(MeLi_result))
MeLi_result2['category'] = MeLi_result

MeLi_result2.to_csv('model_' + version + '_result.csv',index=False)