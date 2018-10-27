# -*- coding: utf-8 -*-
"""Created on Thu Nov  2 10:56:09 2017
@author: ynuwm
"""
import re
import html
import json
import pickle
'''
import time
import sys
import gc
import numpy
import gensim
import scipy
'''
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.ensemble import AdaBoostRegressor
# from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import tensorflow as tf

from pandas import DataFrame
from scipy.stats import spearmanr, pearsonr
from sklearn import ensemble

from keras.models import Sequential, Model
from keras.layers import  Convolution1D,Merge,Dense, Dropout, Activation, Flatten, Embedding, Input
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adagrad
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer

#import os
#os.getcwd()
#os.chdir("F:\\2018_SemEval\\SemEval2018\\src\\V_REG")   
#


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf-8')
    model = {}
    num = 1
    for line in f:
        try:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = np.array(embedding)
            num += 1
        except Exception as e:
            print("Failed at line " + str(num))
    print("Done.",len(model)," words loaded!")
    return model

wv_model_4 = loadGloveModel("F:\WASSA\glove.42B.300d.txt")

glove_dimensions = len(wv_model_4['word'])
print(glove_dimensions)


import numpy
vec1=wv_model_4['happy']
vec2=wv_model_4['angry']

num=float(numpy.sum(vec1*vec2))
denom=numpy.linalg.norm(vec1)*numpy.linalg.norm(vec2)

cos=num/denom
sim=0.5+0.5*cos






def get_word2vec_embedding(word, model, dimensions):

    vec_rep = np.zeros(dimensions)
    if word in model:
        vec_rep = model[word]
    
    return vec_rep

########################  Data Cleaning  #####################
tknzr = TweetTokenizer()

def remove_stopwords(string):
    split_string = \
        [word for word in string.split()
         if word not in stopwords.words('english')]
    
    return " ".join(split_string)
    
def clean_str(string):  
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    string = string.replace("_NEG", "")
    string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_(),!?\'\`]+", " ", string) # removing any twitter handle mentions
    string = re.sub(r"\d+", " ", string) # removing any words with numbers
    string = re.sub(r"_", " ", string)
    string = re.sub(r":", " ", string)
    string = re.sub(r"/", " ", string)
    string = re.sub(r"#", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"n\’t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\’re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\’d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\’ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    #string = re.sub(r"'", " ", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"-", " ", string)
    string = re.sub(r"<", " ", string)
    string = re.sub(r">", " ", string)
    string = re.sub(r";", " ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return remove_stopwords(string.strip().lower())

#####################  Metadata and Class Definitions  ##################
class Tweet(object):

    def __init__(self, id, text, emotion, intensity):
        self.id = id
        self.text = text
        self.emotion = emotion
        self.intensity = intensity

    def __repr__(self):
        return \
            "id: " + self.id + \
            ", text: " + self.text + \
            ", emotion: " + self.emotion + \
            ", intensity: " + self.intensity
            
def read_training_data(training_data_file_path):

    train_list = list()
    with open(training_data_file_path,encoding='utf-8') as input_file:
        for i,line in enumerate(input_file):
            if i == 0:
                continue
            else:
                line = line.strip()
                array = line.split('\t')
                train_list.append(Tweet(array[0], list(tknzr.tokenize(clean_str(array[1]))), 
                                        array[2], float(array[3])))
    return train_list
            
def read_training_data_verbatim(training_data_file_path):

    train_list = list()
    with open(training_data_file_path,encoding='utf-8') as input_file:
        for i,line in enumerate(input_file):
            if i == 0:
                continue
            else:
                line = line.strip()
                array = line.split('\t')
                train_list.append(Tweet(array[0], array[1], array[2], float(array[3])))
    return train_list
    
def read_test_data(training_data_file_path):

    test_list = list()
    with open(training_data_file_path) as input_file:
        for i,line in enumerate(input_file):
            if i == 0:
                continue
            else:
                line = line.strip()
                array = line.split('\t')
                test_list.append(Tweet(array[0], clean_str(array[1]), array[2], None))
    return test_list

non_linear_factor = PolynomialFeatures(3)          

training_data_file_path = "../dataset/3_VAD_REG/2018-Valence-reg-En-train/2018-Valence-reg-En-train.txt"
predictions_file_path = "../dataset/3_VAD_REG/predictions/2018-Valence-reg-En-dev-pred.txt"
dev_set_path = "../dataset/3_VAD_REG/2018-Valence-reg-En-dev/2018-Valence-reg-En-dev.txt"
#test_data_file_path = "../data/test/" + emotion + ".test.gold.txt"
#debug_file_path = "../debug/" + emotion + ".tsv"
word_embeddings_path = "./word_embedding/" + "valence-word-embeddings.pkl"

###################### Feature Extraction Snippets  ########################
import os
os.getcwd()
os.chdir("F:\\2018_SemEval\\SemEval2018\\src\\V_REG")   



#Emoji Intensity
with open('../lexicons/emoji_map.json',encoding='utf-8') as emoji_file:
    emoji_list = json.load(emoji_file)
    
emoji_dict = dict()

for emoji in emoji_list:
    emoji_dict[emoji["emoji"]] = (emoji["name"], emoji["polarity"])

def get_emoji_intensity(word):   
    score = 0.0
    if word in emoji_dict.keys():
        score = float(emoji_dict[word][1])
    
    vec_rep = np.array(score)
    
    return non_linear_factor.fit_transform(vec_rep)[0]


#SentiWordNet
def get_sentiwordnetscore(word):
    
    vec_rep = np.zeros(2)
    
    synsetlist = list(swn.senti_synsets(word))

    if synsetlist:
        vec_rep[0] = synsetlist[0].pos_score()
        vec_rep[1] = synsetlist[0].neg_score()

    return non_linear_factor.fit_transform([vec_rep])[0]

#Emoticon Sentiment Lexicon
emoticon_lexicon_unigrams_file_path = "../lexicons/Emoticon-unigrams.txt"
emoticon_lexicon_bigrams_file_path ="../lexicons/Emoticon-bigrams.txt"
    
emoticon_lexicon_unigrams = dict()
emoticon_lexicon_bigrams = dict()

def get_emoticon_lexicon_unigram_dict():
    with open(emoticon_lexicon_unigrams_file_path,encoding='utf-8') as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_lexicon_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return emoticon_lexicon_unigrams

def get_emoticon_lexicon_bigram_dict():
    with open(emoticon_lexicon_bigrams_file_path,encoding='utf-8') as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_lexicon_bigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return emoticon_lexicon_bigrams
emoticon_lexicon_unigram_dict = get_emoticon_lexicon_unigram_dict()
emoticon_lexicon_bigram_dict = get_emoticon_lexicon_bigram_dict()

def get_unigram_sentiment_emoticon_lexicon_vector(word):
    
    vec_rep = np.zeros(3)
    if word in emoticon_lexicon_unigram_dict.keys():
        vec_rep = emoticon_lexicon_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

def get_bigram_sentiment_emoticon_lexicon_vector(word):
    
    vec_rep = np.zeros(3)
    if word in emoticon_lexicon_bigram_dict.keys():
        vec_rep = emoticon_lexicon_bigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

get_unigram_sentiment_emoticon_lexicon_vector("fury")
get_bigram_sentiment_emoticon_lexicon_vector("add everyone")

#Emoticon Sentiment Aff-Neg Lexicon
emoticon_afflex_unigrams_file_path ="../lexicons/Emoticon-AFFLEX-NEGLEX-unigrams.txt"
emoticon_afflex_bigrams_file_path = "../lexicons/Emoticon-AFFLEX-NEGLEX-bigrams.txt"
    
emoticon_afflex_unigrams = dict()
emoticon_afflex_bigrams = dict()

def get_emoticon_afflex_unigram_dict():
    with open(emoticon_afflex_unigrams_file_path,encoding='utf-8') as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_afflex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return emoticon_afflex_unigrams

def get_emoticon_afflex_bigram_dict():
    with open(emoticon_afflex_bigrams_file_path,encoding='utf-8') as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_afflex_bigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return emoticon_afflex_bigrams

emoticon_afflex_unigram_dict = get_emoticon_afflex_unigram_dict()
emoticon_afflex_bigram_dict = get_emoticon_afflex_bigram_dict()

def get_unigram_sentiment_emoticon_afflex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in emoticon_afflex_unigram_dict.keys():
        vec_rep = emoticon_afflex_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

def get_bigram_sentiment_emoticon_afflex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in emoticon_afflex_bigram_dict.keys():
        vec_rep = emoticon_afflex_bigram_dict[word]
    
    return non_linear_factor.fit_transform([vec_rep])[0]

get_unigram_sentiment_emoticon_afflex_vector("fury")
# get_bigram_sentiment_emoticon_afflex_vector("pay vip")


#Hashtag Sentiment Aff-Neg Lexicon
hashtag_affneglex_unigrams_file_path = "../lexicons/HS-AFFLEX-NEGLEX-unigrams.txt"
hashtag_affneglex_bigrams_file_path = "../lexicons/HS-AFFLEX-NEGLEX-bigrams.txt"
    
hashtag_affneglex_unigrams = dict()
hashtag_affneglex_bigrams = dict()

def get_hashtag_affneglex_unigram_dict():
    with open(hashtag_affneglex_unigrams_file_path,encoding='utf-8') as hashtag_sent_lex_file:
        for line in hashtag_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            hashtag_affneglex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return hashtag_affneglex_unigrams

def get_hashtag_affneglex_bigram_dict():
    with open(hashtag_affneglex_bigrams_file_path,encoding='utf-8') as hashtag_sent_lex_file:
        for line in hashtag_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            hashtag_affneglex_bigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return hashtag_affneglex_bigrams

hashtag_affneglex_unigram_dict = get_hashtag_affneglex_unigram_dict()
hashtag_affneglex_bigram_dict = get_hashtag_affneglex_bigram_dict()

def get_unigram_sentiment_hashtag_affneglex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in hashtag_affneglex_unigram_dict.keys():
        vec_rep = hashtag_affneglex_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

def get_bigram_sentiment_hashtag_affneglex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in hashtag_affneglex_bigram_dict.keys():
        vec_rep = hashtag_affneglex_bigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

get_unigram_sentiment_hashtag_affneglex_vector("#great")
get_bigram_sentiment_hashtag_affneglex_vector("#good luck")

#Hashtag Sentiment Lexicon
hash_sent_lex_unigrams_file_path = "../lexicons/HS-unigrams.txt"
hash_sent_lex_bigrams_file_path = "../lexicons/HS-bigrams.txt"

def get_hash_sent_lex_unigram_dict():
    
    hash_sent_lex_unigrams = dict()
    with open(hash_sent_lex_unigrams_file_path,encoding='utf-8') as hash_sent_lex_file:
        for line in hash_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            if clean_str(word_array[0]):
                hash_sent_lex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return hash_sent_lex_unigrams

def get_hash_sent_lex_bigram_dict():

    hash_sent_lex_bigrams = dict()
    with open(hash_sent_lex_bigrams_file_path,encoding='utf-8') as hash_sent_lex_file:
        for line in hash_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            if clean_str(word_array[0]):
                hash_sent_lex_bigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return hash_sent_lex_bigrams

hash_sent_lex_unigram_dict = get_hash_sent_lex_unigram_dict()
hash_sent_lex_bigram_dict = get_hash_sent_lex_bigram_dict()

def get_unigram_sentiment_hash_sent_lex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in hash_sent_lex_unigram_dict.keys():
        vec_rep = hash_sent_lex_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]


def get_bigram_sentiment_hash_sent_lex_vector(word):

    vec_rep = np.zeros(3)
    if word in hash_sent_lex_bigram_dict.keys():
        vec_rep = hash_sent_lex_bigram_dict[word]
            
    return non_linear_factor.fit_transform([vec_rep])[0]

# get_unigram_sentiment_hash_sent_lex_vector("#fabulous")
# get_bigram_sentiment_hash_sent_lex_vector(". #perfection")


#Depeche Mood
depeche_mood_file_path = "../lexicons/DepecheMood_normfreq.txt"
def get_depeche_vector_dict():
    depeche_vector_dict = dict()
    with open(depeche_mood_file_path) as depeche_mood_file:
        for line in depeche_mood_file:
            word_array = line.replace("\n", "").split("\t")
            depeche_vector_dict[word_array[0].split("#")[0]] = np.array([float(val) for val in word_array[1:]])
    
    return depeche_vector_dict
        
depeche_vector_dict = get_depeche_vector_dict()  
  
def get_depeche_mood_vector(word):
    
   vec_rep = np.zeros(8)
   if word in depeche_vector_dict.keys():
       vec_rep = np.array(depeche_vector_dict[word])

   return non_linear_factor.fit_transform([vec_rep])[0]  

#get_depeche_mood_vector("thanks")


#==============================================================================
# valence    
#==============================================================================
valence = open('../lexicons/Valence.txt',encoding='utf-8').readlines()
valence_dict = {}
valence_list = list()

for i,line in enumerate(valence):
    tmp = line.strip('\n').split(' ')
    valence_dict[tmp[0]]=tmp[1]    
    valence_list.append(tmp[0])  

def tweetToValenceVector(word):
    vec = np.zeros(1)
    if word in valence_list:
        vec = np.array(float(valence_dict[word])/10.0)
    else:
        vec = np.array(0.0)
    return non_linear_factor.fit_transform(vec)[0]

#################   Reading & Vectorizing Data   #######################
#     print(embedding_features.shape)
    
#     lexicon_features = get_unigram_embedding(word, embedding_info[0], unigram_feature_string)
#     poly_lexicon_features = non_linear_factor.fit_transform([lexicon_features])[0]
#     print(poly_lexicon_features.shape)

#     final_features = np.concatenate((embedding_features, poly_lexicon_features))
#     print(final_features.shape)


def is_active_vector_method(string):
    return int(string)

def learn_unigram_word_embedding(word):
    
    word_feature_embedding_dict = dict()    

    index = 0
    word_feature_embedding_dict[index] = get_word2vec_embedding(word, wv_model_4, glove_dimensions)

    '''WordNet'''
    index = 1
    word_feature_embedding_dict[index] = get_sentiwordnetscore(word)


    index = 2
    word_feature_embedding_dict[index] = get_unigram_sentiment_emoticon_lexicon_vector(word)

    index = 3
    word_feature_embedding_dict[index] = get_unigram_sentiment_emoticon_afflex_vector(word)

    index = 4
    word_feature_embedding_dict[index] = get_unigram_sentiment_hash_sent_lex_vector(word)

    index = 5
    word_feature_embedding_dict[index] = get_unigram_sentiment_hashtag_affneglex_vector(word)

    '''Emoji Polarities'''
    index = 6
    word_feature_embedding_dict[index] = get_emoji_intensity(word)
    
    '''Depeche Mood'''
    index = 7
    word_feature_embedding_dict[index] = get_depeche_mood_vector(word)
    
    '''valence'''
    index = 8
    word_feature_embedding_dict[index] = get_depeche_mood_vector(word)

    
    return word_feature_embedding_dict
    
    
def get_unigram_embedding(word, word_embedding_dict, bin_string):      
    word_feature_embedding_dict = word_embedding_dict[word]
    final_embedding = np.array([])
    
    for i in range(9):
        if is_active_vector_method(bin_string[i]):
            final_embedding = np.append(final_embedding, word_feature_embedding_dict[i])
    
    return final_embedding

    
unigram_feature_string = "111111110"

training_tweets = read_training_data(training_data_file_path)
# dev_tweets = read_training_data(dev_set_path)

score_train = list()
tweet_train = list()
for tweet in training_tweets:
    tweet_train.append(tweet.text)
    score_train.append(float(tweet.intensity))
'''
for tweet in dev_tweets:
    tweet_train.append(tweet.text)
    score_train.append(float(tweet.intensity))
'''    
print(len(score_train))
score_train = np.asarray(score_train)


raw_test_tweets = read_training_data_verbatim(dev_set_path)
test_tweets = read_training_data(dev_set_path)

tweet_test_raw = list()
tweet_test_id = list()

tweet_test = list()
y_gold = list()

for tweet in raw_test_tweets:
    tweet_test_raw.append(tweet.text)
    tweet_test_id.append(tweet.id)
    
for tweet in test_tweets:
    tweet_test.append(tweet.text)
    y_gold.append(float(tweet.intensity))
    
print(len(y_gold))

def build_word_embeddings(tweets): 
    max_tweet_length = -1
    word_embedding_dict = dict()

    for tweet in tweets:
        if len(tweet) > max_tweet_length:
            max_tweet_length = len(tweet)

        for token in tweet:
            if token not in word_embedding_dict.keys():
                word_embedding_dict[token] = learn_unigram_word_embedding(token)
                
    return word_embedding_dict, max_tweet_length
    
all_tweets = tweet_train + tweet_test
embedding_info = build_word_embeddings(all_tweets)







# Save vectors
with open(word_embeddings_path, 'wb') as word_embeddings_file:
    pickle.dump(embedding_info, word_embeddings_file)
                  
# Restore vectors
with open(word_embeddings_path, 'rb') as word_embeddings_file:
    embedding_info = pickle.load(word_embeddings_file)               
                
embeddings_index = embedding_info[0]
MAX_SEQUENCE_LENGTH = embedding_info[1]
MAX_NB_WORDS = 20000
EMBEDDING_DIM = len(get_unigram_embedding("!", embedding_info[0], unigram_feature_string))
print(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)                

word_indices = dict()
current_index = 1


def sequence_tweets(tweets):
    global current_index
    vectors = list()
    for tweet in tweets:        
        vector = list()
        for word in tweet:
            word_index = None
            
            if word in word_indices:
                word_index = word_indices[word]
            else:
                word_index = current_index
                current_index += 1
                word_indices[word] = word_index
            
            vector.append(word_index)
        
        vectors.append(vector)

    return vectors
    
x_train = sequence_tweets(tweet_train)
x_test = sequence_tweets(tweet_test)

len(word_indices)

# display(tweet_train)

word_embedding_matrix = list()
word_embedding_matrix.append(np.zeros(EMBEDDING_DIM))

for word in sorted(word_indices, key=word_indices.get):
    embedding_features = get_unigram_embedding(word, embedding_info[0], unigram_feature_string)    
    word_embedding_matrix.append(embedding_features)

word_embedding_matrix = np.asarray(word_embedding_matrix, dtype='f')
print(word_embedding_matrix.shape)

word_embedding_matrix = scale(word_embedding_matrix)


 
    
    
tmp1 =  word_indices['happy']
tmp2 =  word_indices['angry']   
    

import numpy
vec1=word_embedding_matrix[tmp1+1]
vec2=word_embedding_matrix[tmp2+1]

num=float(numpy.sum(vec1*vec2))
denom=numpy.linalg.norm(vec1)*numpy.linalg.norm(vec2)

cos=num/denom
sim=0.5+0.5*cos

















#==============================================================================
#  Recurrent Neural Network Implementation in Keras   
#==============================================================================
pre_padding = 6

x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
'''
x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH + pre_padding, padding="pre")
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH + pre_padding, padding="pre")
'''
len(x_train), len(x_test), len(x_train[0])


shuffle_indices = np.random.permutation(np.arange(len(x_train)))
x_shuffled = x_train[shuffle_indices]
y_shuffled = score_train[shuffle_indices]


embed_1 = Embedding(input_dim=len(word_indices) + 1, output_dim=EMBEDDING_DIM, weights=[word_embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH, trainable=True)

conv_1 = Conv1D(128, 3, activation='relu', name='conv1')
conv_2 = Conv1D(128, 3, activation='relu', name='conv2')
conv_3 = Conv1D(256, 3, activation='relu', name='conv3')
conv_4 = Conv1D(256, 3, activation='relu', name='conv4')
conv_5 = Conv1D(256, 3, activation='relu', name='conv5')
conv_6 = Conv1D(1024, 3, activation='relu', name='conv6')
conv_7 = Conv1D(1024, 3, activation='relu', name='conv7')
conv_8 = Conv1D(1024, 3, activation='relu', name='conv8')

pool_1 = AveragePooling1D(pool_length=3, name='pool1')
pool_2 = AveragePooling1D(pool_length=3,  name='pool2')
pool_3 = MaxPooling1D(pool_length=3, name='pool3')
pool_4 = MaxPooling1D(pool_length=3, name='pool4')

lstm_1 = LSTM(256, name='lstm1', return_sequences=True)
lstm_2 = LSTM(128, name='lstm2', return_sequences=True)
lstm_3 = LSTM(64, name='lstm3')

gru_1 = GRU(256, name='gru1', return_sequences=True)
gru_2 = GRU(256, name='gru2', return_sequences=True)
gru_3 = GRU(256, name='gru3')

bi_lstm_1 = Bidirectional(lstm_1, name='bilstm1')
bi_lstm_2 = Bidirectional(lstm_2, name='bilstm2')
bi_lstm_3 = Bidirectional(lstm_3, name='bilstm3')

dense_1 = Dense(256, activation='relu', name='dense1')
dense_2 = Dense(1, activation='sigmoid', name='dense2')

drop_1 = Dropout(0.5, name='drop1')
drop_2 = Dropout(0.5, name='drop2')


def get_rnn_model(): 
    model = Sequential()   
    model.add(embed_1)
    
    model.add(conv_1)
    model.add(conv_2)
#     model.add(pool_1)
    
#     model.add(conv_3)
#     model.add(conv_4)
#     model.add(pool_2)
    
#     model.add(conv_5)  
  
#    model.add(lstm_1)
#    model.add(lstm_2)
#    model.add(lstm_3)
    model.add(bi_lstm_1)
    model.add(bi_lstm_2)
    model.add(bi_lstm_3) 
    
    model.add(dense_1)
    model.add(drop_1)
    model.add(dense_2)

    model.compile(loss='mean_squared_error', optimizer="adam")
    
    return model 
    
nn_model = KerasRegressor(build_fn = get_rnn_model, nb_epoch=12, batch_size=32, verbose=1)

#score_train = np.asarray(score_train)

#ml_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
nn_model.fit(x_shuffled, y_shuffled)


y_pred = nn_model.predict(x_test)
#y_pred = np.reshape(y_pred, len(y_pred))

print(pearsonr(y_pred, y_gold))

 
#==============================================================================
# 结果写入
#==============================================================================
y_pred =  y_pred.tolist()
with open(predictions_file_path, 'w', encoding='utf-8') as predictions_file:
    predictions_file.write('ID' + '\t' + \
            'Tweet' + '\t' + \
            'Affect' + '\t' + \
            'Dimension' + '\t' + \
            'Intensity Score'+'\n')    
    for i in range(len(y_pred)):   
        predictions_file.write(
        str(raw_test_tweets[i].id) + "\t" + \
        raw_test_tweets[i].text + "\t" + \
        'valence' + "\t" + \
        str(y_pred[i]) + "\n") 

    
#==============================================================================
# Aattention LSTM + CNN
#==============================================================================
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper, Recurrent

from keras.utils.test_utils import keras_test
from keras.layers import wrappers, recurrent, InputLayer
from keras.layers import core, convolutional, recurrent
from keras.models import Sequential, Model, model_from_json


class Attention(Wrapper):
    """
    This wrapper will provide an attention layer to a recurrent layer. 
    
    # Arguments:
        layer: `Recurrent` instance with consume_less='gpu' or 'mem'
    
    # Examples:
    
    ```python
    model = Sequential()
    model.add(LSTM(10, return_sequences=True), batch_input_shape=(4, 5, 10))
    model.add(TFAttentionRNNWrapper(LSTM(10, return_sequences=True, consume_less='gpu')))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop') 
    ```
    
    # References
    - [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)
    
    
    """
    def __init__(self, layer, **kwargs):
        assert isinstance(layer, Recurrent)
        if layer.get_config()['consume_less']=='cpu':
            raise Exception("AttentionLSTMWrapper doesn't support RNN's with consume_less='cpu'")
        self.supports_masking = True
        super(Attention, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_samples, nb_time, input_dim = input_shape

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(Attention, self).build()
        
        self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
        self.W2 = self.layer.init((self.layer.output_dim, input_dim), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((input_dim,), name='{}_b2'.format(self.name))
        self.W3 = self.layer.init((input_dim*2, input_dim), name='{}_W3'.format(self.name))
        self.b3 = K.zeros((input_dim,), name='{}_b3'.format(self.name))
        self.V = self.layer.init((input_dim,), name='{}_V'.format(self.name))

        self.trainable_weights = [self.W1, self.W2, self.W3, self.V, self.b2, self.b3]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        # This is based on [tensorflows implementation](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506).
        # First, we calculate new attention masks:
        #   attn = softmax(V^T * tanh(W2 * X +b2 + W1 * h))
        # and we make the input as a concatenation of the input and weighted inputs which is then
        # transformed back to the shape x of using W3
        #   x = W3*(x+X*attn)+b3
        # Then, we run the cell on a combination of the input and previous attention masks:
        #   h, state = cell(x, h).
        
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        h = states[0]
        X = states[-1]
        xW1 = states[-2]
        
        Xr = K.reshape(X,(-1,nb_time,1,input_dim))
        hW2 = K.dot(h,self.W2)+self.b2
        hW2 = K.reshape(hW2,(-1,1,1,input_dim)) 
        u = K.tanh(xW1+hW2)
        a = K.sum(self.V*u,[2,3])
        a = K.softmax(a)
        a = K.reshape(a,(-1, nb_time, 1, 1))
        
        # Weight attention vector by attention
        Xa = K.sum(a*Xr,[1,2])
        Xa = K.reshape(Xa,(-1,input_dim))
        
        # Merge input and attention weighted inputs into one vector of the right size.
        x = K.dot(K.concatenate([x,Xa],1),self.W3)+self.b3    
        
        h, new_states = self.layer.step(x, states)
        return h, new_states

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        
        # Calculate K.dot(x, W2) only once per sequence by making it a constant
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        Xr = K.reshape(x,(-1,nb_time,input_dim,1))
        Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
        xW1t = K.conv2d(Xrt,self.W1,border_mode='same')     
        xW1 = K.permute_dimensions(xW1t, (0, 2, 3, 1))
        constants.append(xW1)
        
        # we need to supply the full sequence of inputs to step (as the attention_vector)
        constants.append(x)
        
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)
        

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output


def get_rnn_model():  
    model = Sequential()  
    model.add(embed_1)
    
    model.add(conv_1)
    model.add(conv_2)     
    model.add(pool_1)
    
#     model.add(conv_3)
#     model.add(conv_4)
#     model.add(pool_2)
    
#     model.add(conv_5)  
    model.add(Attention(recurrent.LSTM(256,input_dim=EMBEDDING_DIM,consume_less='mem',return_sequences=True)))
    model.add(Attention(recurrent.LSTM(128,input_dim=EMBEDDING_DIM,consume_less='mem',return_sequences=True)))
    model.add(Attention(recurrent.LSTM(64,input_dim=EMBEDDING_DIM,consume_less='mem',return_sequences=False)))
     
#    model.add(bi_lstm_1)
#    model.add(bi_lstm_2)
#    model.add(bi_lstm_3)
   
    model.add(Dense(256))
    model.add(drop_1)   
    model.add(dense_2)

    model.compile(loss='mean_squared_error', optimizer="Adam")
    
    return model 


nn_model = KerasRegressor(build_fn = get_rnn_model, nb_epoch=12, batch_size=32, verbose=1)
#score_train = np.asarray(score_train)

#ml_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
nn_model.fit(x_shuffled, y_shuffled)

y_pred1 = nn_model.predict(x_test)
#y_pred = np.reshape(y_pred, len(y_pred))

print(pearsonr(y_pred1, y_gold))


#==============================================================================
#         单 CNN 模型
#==============================================================================
num_filters = 256
filter_sizes = (2, 3, 4, 5, 6)

graph_in = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)
        
if len(filter_sizes)>1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]
    
    graph = Model(input=graph_in, output=out)

def get_cnn_model():    
    model = Sequential()
    model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word_indices) + 1,input_length=MAX_SEQUENCE_LENGTH,
                           weights=[word_embedding_matrix], trainable=True))
    model.add(Dropout(0.25, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    model.add(graph)
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer="Adam") 
    return model

def get_CNN_model():  
    model = Sequential()  
    model.add(embed_1)
    
    model.add(conv_1)
    model.add(conv_2)     
    model.add(pool_3)
    
    model.add(conv_3)
    model.add(conv_4)
    model.add(pool_4)

    model.add(Flatten())

    model.add(Dense(256))
    model.add(drop_1)   
    model.add(dense_2)

    model.compile(loss='mean_squared_error', optimizer="Adam")
    
    return model     
    
nn_model = KerasRegressor(build_fn=get_CNN_model, nb_epoch=10, batch_size=32, verbose=1)
#score_train = np.asarray(score_train)

nn_model.fit(x_shuffled, y_shuffled)

y_pred2 = nn_model.predict(x_test).tolist()
print(pearsonr(y_pred2, y_gold))



#==============================================================================
#        CNN ensemble 之后模型
#==============================================================================
tmp_pred = [0,1,2,3,4,5,6,7,8,9]
for i in range(10):    
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = score_train[shuffle_indices]

    def get_rnn_model():    
        model = Sequential()  
        model.add(embed_1)
        
        model.add(conv_1)
        model.add(conv_2)     
        model.add(pool_3)
        
        model.add(conv_3)
        model.add(conv_4)
        model.add(pool_4)
    
        model.add(Flatten())
    
        model.add(Dense(256))
        model.add(drop_1)   
        model.add(dense_2)
        model.compile(loss='mean_absolute_percentage_error', optimizer="Adam")

        return model

    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=100, batch_size=32, verbose=1)
    #nn_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
    nn_model.fit(x_shuffled, y_shuffled)  
    tmp_pred[i] = nn_model.predict(x_test).tolist()


y_pred = list()

for j in range(len(tmp_pred[0])):
    temp_emotion = (tmp_pred[0][j] + tmp_pred[1][j] + tmp_pred[2][j] + \
                   tmp_pred[3][j] + tmp_pred[4][j] + tmp_pred[5][j] + \
                   tmp_pred[6][j] + tmp_pred[7][j] +tmp_pred[8][j] + tmp_pred[9][j])/10  
    y_pred.append(temp_emotion)    
        
print(pearsonr(y_pred, y_gold)) 



y_pred = list()
for i in range(len(y_pred0)):
    y_pred.append((y_pred0[i] + y_pred1[i] + y_pred2[i])/3)
print(pearsonr(y_pred, y_gold))



with open(predictions_file_path, 'w', encoding='utf-8') as predictions_file:
    predictions_file.write('ID' + '\t' + \
            'Tweet' + '\t' + \
            'Affect' + '\t' + \
            'Dimension' + '\t' + \
            'Intensity Score'+'\n')    
    for i in range(len(y_pred)):   
        predictions_file.write(
        str(raw_test_tweets[i].id) + "\t" + \
        raw_test_tweets[i].text + "\t" + \
        'valence' + "\t" + \
        str(y_pred[i]) + "\n") 

        
  
  
        
        
#==============================================================================
#         单层LSTM(Bi-LSTM) ensemble之后的效果
#==============================================================================
print('\nTraing ONE Bi-LSTM (LSTM)...')
tmp_pred = [0,1,2,3,4,5,6,7,8,9]
for i in range(10):    
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = score_train[shuffle_indices]

    def get_rnn_model():    
        model = Sequential()    
        model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word_indices) + 1,
                           weights=[word_embedding_matrix], mask_zero=True,trainable=True))
        model.add(Bidirectional(LSTM(256, return_sequences=False)))
        model.add(Dropout(0.25))    
         
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))   
        model.compile(loss='mape', optimizer="Adam") 
        return model

    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=80, batch_size=32, verbose=1)
    #nn_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
    nn_model.fit(x_shuffled, y_shuffled)  
    tmp_pred[i] = nn_model.predict(x_test).tolist()
    #print(pearsonr(tmp_pred[9], y_gold)) 

   
y_pred = list()

for j in range(len(tmp_pred[0])):
    temp_emotion = (tmp_pred[0][j] + tmp_pred[2][j] + tmp_pred[3][j]+ tmp_pred[7][j] + tmp_pred[8][j] + tmp_pred[9][j])/6
    y_pred.append(temp_emotion)    
        
print(pearsonr(y_pred, y_gold))    
 


#==============================================================================
#            两层 BiLSTM  ensemble
#==============================================================================
print('\nTraing TWO Bi-LSTM...')
tmp_pred = [0,1,2,3,4,5,6,7,8,9]
for i in range(10):
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = score_train[shuffle_indices]
    
    def get_rnn_model():    
        model = Sequential()    
        model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word_indices) + 1,
                           weights=[word_embedding_matrix], mask_zero=True,trainable=True))
        model.add(Bidirectional(LSTM(512, return_sequences=True)))
        model.add(Bidirectional(LSTM(256, return_sequences=False)))
        model.add(Dropout(0.25))    
         
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))   
        model.compile(loss='msle', optimizer="Adam") 
        return model
    
    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=8, batch_size=64, verbose=1)
    nn_model.fit(x_shuffled, y_shuffled)
    tmp_pred[i] = nn_model.predict(x_test).tolist()
    #print(pearsonr(tmp_pred[8], y_gold)) 
    
y_pred = list()
for j in range(len(tmp_pred[0])):
    temp_emotion = (tmp_pred[0][j] + tmp_pred[1][j] + tmp_pred[2][j] + \
                    tmp_pred[3][j] + tmp_pred[4][j] + tmp_pred[5][j] + \
                    tmp_pred[6][j] + tmp_pred[7][j] + tmp_pred[8][j] + tmp_pred[9][j])/10
    y_pred.append(temp_emotion)    
       
print(pearsonr(y_pred, y_gold))    
#print(spearmanr(y_pred, y_gold))

y_pred = list()

with open(predictions_file_path, 'w', encoding='utf-8') as predictions_file:
    predictions_file.write('ID' + '\t' + \
            'Tweet' + '\t' + \
            'Affect' + '\t' + \
            'Dimension' + '\t' + \
            'Intensity Score'+'\n')    
    for i in range(len(y_pred)):   
        predictions_file.write(
        str(raw_test_tweets[i].id) + "\t" + \
        raw_test_tweets[i].text + "\t" + \
        'valence' + "\t" + \
        str(y_pred[i]) + "\n") 


#np.save('./tmp_pred.npy',tmp_pred)\



#==============================================================================
#     tensorflow
#==============================================================================
#import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 


with tf.Session() as sess:
    def get_rnn_model():    
            model = Sequential()    
            model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word_indices) + 1,
                               weights=[word_embedding_matrix], mask_zero=True,trainable=True))
            model.add(Bidirectional(LSTM(256, return_sequences=False)))
            model.add(Dropout(0.25))    
             
            model.add(Dense(256,activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1,activation='sigmoid'))   
            model.compile(loss='mse', optimizer="Adam") 
            return model
    
    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=10, batch_size=32, verbose=1)
    #nn_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
    nn_model.fit(x_shuffled, y_shuffled)  
    tf_pred = nn_model.predict(x_test).tolist()





























