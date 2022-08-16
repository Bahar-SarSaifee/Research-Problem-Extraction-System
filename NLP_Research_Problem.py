#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import numpy as np
import pandas as pd
import bs4
import requests
import pytextrank
import yake

import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
import en_core_web_sm

import spacy
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
# pd.set_option('display.max_colwidth', 200)
# %matplotlib inline


# In[ ]:


# Knowledge Graph
def get_entities(sent):
  ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""
    
    #############################################################

    for tok in nlp(sent):
    ## chunk 2
        #remove stopword
        if (tok.text.lower() not in stop_words):
            #remove digits
            New_text = re.sub("[0-9]","",tok.text)
            #remove tags
            New_text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",New_text)
            #remove words less than two letters
            if(len(New_text)>2):
        # if token is a punctuation mark then move on to the next token
                if tok.dep_ != "punct":
              # check: token is a compound word or not
                    if tok.dep_ == "compound":
                        prefix = New_text
                # if the previous word was also a 'compound' then add the current word to it
                        if prv_tok_dep == "compound":
                            prefix = prv_tok_text + " "+ New_text

              # check: token is a modifier or not
                    if tok.dep_.endswith("mod") == True:
                        modifier = New_text
                    # if the previous word was also a 'compound' then add the current word to it
                        if prv_tok_dep == "compound":
                            modifier = prv_tok_text + " "+ New_text

              ## chunk 3
                    if tok.dep_.find("subj") == True:
                        ent1 = modifier +" "+ prefix + " "+ New_text
                        prefix = ""
                        modifier = ""
                        prv_tok_dep = ""
                        prv_tok_text = ""      

                    ## chunk 4
                    if tok.dep_.find("obj") == True:
                        ent2 = modifier +" "+ prefix +" "+ New_text

          ## chunk 5  
          # update variables
                    prv_tok_dep = tok.dep_
                    prv_tok_text = New_text

  #############################################################
    return [ent2.strip(), ent1.strip()]


# In[ ]:


def Rake_NLTK_Keywords(doc):
    
    r = Rake()
    keywordList_Rake = []

    for i in range(len(doc)):
        r.extract_keywords_from_text(doc[i])
        rankedList = r.get_ranked_phrases_with_scores()
        keywordList_Rake.append([i])
        for keyword in rankedList:
            keyword_updated = keyword[1].split()
            keyword_updated_string = " ".join(keyword_updated[:3])
            keywordList_Rake[i].append(keyword_updated_string)

    keywords_Rake_new = []
    for val in keywordList_Rake:
        if len(val) != 1 :
            keywords_Rake_new.append(val)
            
    return (keywords_Rake_new)


# In[ ]:


def TextRank_Keywords(doc):
    
    keywords_Textrank = []

    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")
    # add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")
    for i in range(len(doc)):
        document = nlp(doc[i])
        keywords_Textrank.append([i])
    # examine the top-ranked phrases in the document
        for phrase in document._.phrases:
            keywords_Textrank[i].append(phrase.text)

    keywords_Textrank_new = []
    for val in keywords_Textrank:
        if len(val) != 1 :
            keywords_Textrank_new.append(val)
            
    return (keywords_Textrank_new)


# In[ ]:


def Yake_Keywords(doc):
    
    keywords_Yake = []

    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 2

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,
                                         dedupLim=deduplication_threshold,
                                         top=numOfKeywords, features=None)
    for i in range(len(doc)):
        keywords = kw_extractor.extract_keywords(doc[i])
        keywords_Yake.append([i])
        for kw in keywords:
            keywords_Yake[i].append(kw[0])


    # remove none value in list of keywords
    keywords_Yake_new = []
    for val in keywords_Yake:
        if len(val) != 1 :
            keywords_Yake_new.append(val)
            
    return (keywords_Yake_new)


# In[ ]:


def get_similarity(entity_doc):

    similarity_list = []

    ## Use Title for Similarity with keywords
    doc1 = nlp(df['title'][0])
    counter = 0
    
    for i in range(len(entity_doc)):        
        for j in range(len(entity_doc[i])):
            if j!=0 :
                similarity_list.append([i])
                doc2 = nlp(entity_doc[i][j])
                sim = doc1.similarity(doc2)    
                if sim != 0:
                    similarity_list[counter].append(entity_doc[i][j])
                    similarity_list[counter].append(sim)
                counter = counter +1
            
    #remove index without keywords        
    similarity_list_new = []
    for val in similarity_list:
        if len(val) != 1 :
            similarity_list_new.append(val)
            
    #Sort with similarity value
    similarity_list_new.sort(key=lambda row: (row[2]), reverse=True)
    
    # cosine similarity
    return (similarity_list_new)


# In[ ]:


Keywords_collectin = []

for k in range(155):
    print(k)
    entity_pairs = []
    Rake_ky = []
    Textrank_ky = []
    Yake_ky = []
    keywords_all = []
    similarity_KG = []
    
    d = pd.read_csv("test_dataset/Stanza Text files/%d.txt" % k, delimiter='\n', on_bad_lines='skip')
    df = pd.DataFrame(data=d)

    # get entity pairs
    counter = 0
    for i in tqdm(df['title']):
        entity_pairs.append([counter])
        entity_pairs[counter].extend(get_entities(i))

        counter = counter + 1

    # keywords extraction with Rake
    Rake_ky = Rake_NLTK_Keywords(df['title'])

    # keywords extraction with Textrank
    Textrank_ky = TextRank_Keywords(df['title'])

    # keywords extraction with Yake
    Yake_ky = Yake_Keywords(df['title'])

    # Merge all of keywords
    keywords_all.extend(entity_pairs[:20])
    keywords_all.extend(Rake_ky[:20])
    keywords_all.extend(Textrank_ky[:20])
    keywords_all.extend(Yake_ky[:20])
    
    # get similarity with title   
    similarity_KG = get_similarity(keywords_all)
    
    Keywords_collectin.append([k])
    Keywords_collectin[k].append(similarity_KG[0][1])


# In[ ]:


d = pd.read_csv("test_dataset/Test_Keywords/test_collection.csv", delimiter='\n', header = None)
test_set_1 = pd.DataFrame(data=d)

test_set_2 = test_set_1.values.tolist()
test_set_list = list()

for item in test_set_2:
    item = item[0]
    item = item.replace("['", "")
    item = item.replace("']", "")
    item = item.replace("', '", ",")
    test_set_list.append(item)
    
y_true =  [1] * len(test_set_1)


# In[ ]:


#Compare test_set and our keywords collection ------> Determine y_pred
Keywords_collection_list = [rec[1] for rec in Keywords_collectin]
y_pred = list()

for i, item in enumerate(Keywords_collection_list):
    if "," not in test_set_list[i]:
        if Keywords_collection_list[i].lower() == test_set_list[i].lower():
            y_pred.append(1)
        else:
            y_pred.append(0)
    elif "," in test_set_list[i] and len(test_set_list[i].split(",")) > 1:
        if Keywords_collection_list[i].lower() in test_set_list[i].lower().split(","):
            y_pred.append(1)
        else:
            y_pred.append(0)
    else:
        print("Error")


# # Evaluation

# In[ ]:


from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

print(precision_score(y_true, y_pred))


# In[ ]:


print(recall_score(y_true, y_pred))


# In[ ]:


print(f1_score(y_true, y_pred))


# In[ ]:


accuracy_score(y_true, y_pred)


# In[ ]:




