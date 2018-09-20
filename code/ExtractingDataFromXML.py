#!/usr/bin/env python
# coding: utf-8

# In[47]:


from bs4 import BeautifulSoup
import re
from num2words import num2words
import os
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from gensim.models.phrases import Phrases, Phraser
from gensim.models import word2vec
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# In[48]:


def filtering_data(text):
    
    eligibility_list, features, labels, inclusion_features,exclusion_features,inclusion_labels,exclusion_labels    = [], [], [], [], [], [], []
    text = text.lower()
    len_inclusion_str = len('inclusion criteria')
    len_exclusion_str = len('exclusion criteria')
    inclusion_index = text.find('inclusion criteria')
    exclusion_index = text.find('exclusion criteria')
    
    
    if inclusion_index != -1:
        inclusion_string = text[inclusion_index+len_inclusion_str:]
        inclusion_features = preprocessing(inclusion_string)
        for idx, each_feature in enumerate(inclusion_features):
            if each_feature.startswith('no'):
                inclusion_labels.append(1)
                inclusion_features[idx] = inclusion_features[idx].replace('no', '')
            else:
                inclusion_labels.append(0)

    if exclusion_index!=-1:
        
        exclusion_string = text[exclusion_index+len_exclusion_str :]
        exclusion_features = preprocessing(exclusion_string)
        exclusion_labels = [1] * len(exclusion_features)
    
    features = inclusion_features + exclusion_features
    labels = inclusion_labels + exclusion_labels
            
    return features, labels


# In[90]:


def preprocessing(text):
    #bullet_pattern = r'(\W+\d+\.[^\n\w]|\n\n|\W-\W)'
    bullet_pattern = r'\n+\s+\d+\.+'
    decimal_pattern = r'([\d]+\.[\d]+)'
    greater_than_pattern = r'(>| >=)'
    less_than_pattern = r'(<| <=)'
    equal_to_pattern = r'(=)'
    multiply_pattern = r'\s[x]\s'
    positive_pattern = r'\++'
    
    contraction_pattern = r"'|-|\"|/|\(|\)|,|:|/"
#     text = re.sub(contraction_pattern, '', text)
#     substitution_str = ' .'+str_type+' is '
#     text = re.sub(bullet_pattern, substitution_str, text)
    
    for match in re.findall(decimal_pattern, text):
        text = text.replace(match, match.split('.')[0] + ' dot ' + match.split('.')[1])
    y_temp = []
    text = re.sub(greater_than_pattern, ' greater_than ', text)
    text = re.sub(less_than_pattern, ' less_than ', text)
    text = re.sub(equal_to_pattern, ' equal_to ', text)
    text = re.sub(multiply_pattern, ' multiply ', text)
    text = re.sub(positive_pattern, ' positive ', text)
    text = convert_num2words(text)
    temp = re.split('\n\n+', re.sub(contraction_pattern, ' ', text))
    temp = [re.sub('\s+', ' ', element) for element in temp if len(element) > 3] 
    [y_temp.extend(e.split('.')) for e in temp]
    temp = [element for element in y_temp if len(element) > 3]  
    return temp


# In[91]:


def convert_num2words(line):
        digits_pattern = r'\d+'
        for d in re.findall(digits_pattern, line):
            line = line.replace(d, num2words(int(d)))
        return line


# In[92]:


def read_files(file_path):
    all_tags = ['eligibility', 'condition', 'intervention']
    soup = BeautifulSoup(open(file_path), 'html.parser')
    for specific_tag in all_tags:
        if specific_tag == 'eligibility':
            for tag in soup.findAll('eligibility'):
                eligibility = tag.find('textblock').text
        if specific_tag == 'condition':
            conditions = []
            for tag in soup.findAll('condition'):
                conditions.append(tag.text)
                
        if specific_tag == 'intervention':
            intervention = []
            for tag in soup.findAll('intervention'):
                for type_in, text_in in zip(tag.find('intervention_type'), tag.find('intervention_name')):
                    intervention.append((type_in, text_in))    
    return eligibility, conditions, intervention

    


# In[93]:


def isCancer(s):
    return any(ext in s.lower() for ext in ["cancer" , "neoplasm" , "oma", "tumor"])


# In[134]:


def main_method(folder_path):
    stop_words = set(stopwords.words('english') + ['and', 'the', 'for', 'with', 'are', 'who', 'from'])
    all_features = []
    all_labels = []
    count = 1
    for f in os.listdir(folder_path)[:100]:
        print(f, count)
        count += 1
        eligibility, conditions, intervention = read_files(os.path.join(folder_path, f))
        features, labels = filtering_data(eligibility)
        for idx,e in enumerate(features):
            for c in conditions:
                if isCancer(c):
                    for i in intervention:
                        append_str = 'Diagnosis is '+ c + ' and ' + 'intervention is ' + i[0] + ' ' + i[1] + ' and ' + e
                        
                        append_str = ' '.join([word.lower() for word in append_str.split() if word not in stop_words])    
                        all_features.append(append_str)
                        all_labels.append(labels[idx])
    final_dict = {'features': all_features, 'labels': all_labels}
    final_df =  pd.DataFrame.from_dict(final_dict)
    return final_df
                


# In[135]:


def generating_bigrams(final_df):
    eligibility_criteria = final_df['features']
    bigrams_input = [each_row.split() for each_row in eligibility_criteria]
    bigram_transformer = Phrases(bigrams_input, min_count=20, threshold=500)
    bigram_transformer.save("bigrams", pickle_protocol=4)

    fd = open("bigrams.txt", 'a')
    for phrase, score in bigram_transformer.export_phrases(bigrams_input):
        fd.write(u'{0}   {1}'.format(phrase, score))
    fd.close()

    return bigram_transformer
        


# In[136]:


def bigrams_text2input(bigram_transformer, features):
    word2vec_input = []
    for f in features:
        line = [word for word in bigram_transformer[f.split()] if len(word)>2]
        word2vec_input.append(line)
    return word2vec_input


# In[137]:


def tsne_plot_all(model):
    "Creates and TSNE model and plot it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('tsne_all_words.png')


# In[138]:


def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.savefig('tsne_most_similar_to_'+word+'.png')
#     plt.show()


# In[149]:


def plot_most_common(model, num=20):
    "Creates and TSNE model and plot it"
    labels = []
    tokens = []
    
    for word, obj in nltk.FreqDist(model.wv.vocab).most_common(num):
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('tsne_most_common_words_'+str(num)+'.png')
    


# In[139]:


final_df = main_method('../../data/search_result')


# In[141]:


final_df.to_csv('clinical_trial_training_data_all.csv')


# In[142]:


bigram_transformer = generating_bigrams(final_df)


# In[143]:


word2vec_input = bigrams_text2input(bigram_transformer, final_df['features'])


# In[151]:


# model = word2vec.Word2Vec(word2vec_input, size=100, window=4, min_count=3, workers=4)
# nltk.FreqDist(model.wv.vocab).most_common(10)


# In[144]:


model = word2vec.Word2Vec(word2vec_input, size=100, window=20, min_count=500, workers=4)
tsne_plot(model)
display_closestwords_tsnescatterplot(model, 'paclitaxel')
plot_most_common(model, num=100)


# In[152]:


# model = word2vec.Word2Vec(word2vec_input, size=100, window=4, min_count=3, workers=4)
# tsne_plot(model)


# In[ ]:




