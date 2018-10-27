# -*- coding: utf-8 -*-
"""Created on Thu Nov  2 10:56:09 2017
@author: ynuwm
"""
import re
import html
import json
'''
import time
import pickle
import sys
import gc
import numpy
import gensim
import scipy
from pandas import DataFrame
'''
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.ensemble import AdaBoostRegressor
# from xgboost import XGBRegressor
import numpy as np

from scipy.stats import spearmanr, pearsonr
from sklearn import ensemble

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input
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

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf-8')
    model = {}
    num = 1
    for j,line in enumerate(f):
        if j == 0:
            continue
        else:
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

def loadSWMModel(swmFile):
    print("Loading SWM Model")
    f = open(swmFile,'r',encoding='utf-8')
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
    
wv_model_0 = loadGloveModel("F:\WASSA\glove.42B.300d.txt")
wv_model_1 = loadSWMModel("F:\WASSA\model_swm_300-6-10-low.w2v")


glove_dimensions = len(wv_model_0['word'])
print('glove_dimensions: %s' % glove_dimensions)
swm_dimensions = len(wv_model_1['the'])
print('swm_dimensions: %s' % swm_dimensions)

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
    
'''    
def clean_str(string):  
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    string = string.replace("_NEG", "")
    string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_(),!?\'\`]+", "@username", string) # removing any twitter handle mentions
    string = re.sub(r"\d+", " ", string) # removing any words with numbers
    string = re.sub(r"_", " ", string)
    string = re.sub(r":", " ", string)
    string = re.sub(r"/", " ", string)
    string = re.sub(r"#", " ", string)
    string = re.sub(r"\.", " . ", string)
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
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"-", " ", string)
    string = re.sub(r"<", " ", string)
    string = re.sub(r">", " ", string)
    string = re.sub(r";", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub('([a-zA-Z])\\1+', '\\1\\1', string) # limit length of repeated letters to 2
    return string.strip().lower() 
'''       
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
                                        array[2], array[3]))
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
                train_list.append(Tweet(array[0], array[1], array[2], array[3]))
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
#emotion = "anger" 
#emotion = "fear"  
#emotion = "joy"          
emotion = "sadness"  
 
training_data_file_path = '../dataset/1_EI_REG/EI-reg-En-train/EI-reg-En-'+ emotion + '-train.txt'
predictions_file_path = "../dataset/1_EI_REG/predictions/2018-EI-reg-En-"+ emotion + '-pred.txt'
dev_set_path = "../dataset/1_EI_REG/2018-EI-reg-En-dev/2018-EI-reg-En-"+ emotion + "-dev.txt"
test_data_file_path = "../dataset/1_EI_REG/2018-EI-reg-En-test/2018-EI-reg-En-"+ emotion + "-test.txt"
#debug_file_path = "../debug/" + emotion + ".tsv"
#word_embeddings_path = "../data/word_embedding/" + emotion + "-word-embeddings.pkl"

###################### Feature Extraction Snippets  ########################

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
  

#kk = get_emoji_intensity('🤐')

#Emotion Intensity Lexicon
affect_intensity_file_path = "../lexicons/NRC-AffectIntensity-Lexicon.txt"
    
def get_word_affect_intensity_dict(emotion):
    word_intensities = dict()

    with open(affect_intensity_file_path,encoding='utf-8') as affect_intensity_file:
        for line in affect_intensity_file:
            word_int_array = line.replace("\n", "").split("\t")

            if (word_int_array[2] == emotion):
                word_intensities[word_int_array[0]] = float(word_int_array[1])

    return word_intensities

word_intensities = get_word_affect_intensity_dict(emotion)

def get_emo_int_vector(word):
    score = 0.0
    if word in word_intensities.keys():
        score = float(word_intensities[word])
        
    vec_rep = np.array([score])
    
    return non_linear_factor.fit_transform([vec_rep])[0]

                                    
                                           
#SentiWordNet
def get_sentiwordnetscore(word):   
    vec_rep = np.zeros(2)    
    synsetlist = list(swn.senti_synsets(word))
    if synsetlist:
        vec_rep[0] = synsetlist[0].pos_score()
        vec_rep[1] = synsetlist[0].neg_score()

    return non_linear_factor.fit_transform([vec_rep])[0]
    
#Sentiment Emotion Presence Lexicon
sentiment_emotion_lex_file_path = "../lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

def get_affect_presence_list(emotion):
    word_list = list()
    
    with open(sentiment_emotion_lex_file_path,encoding='utf-8') as sentiment_emotion_lex_file:
        for line in sentiment_emotion_lex_file:
            word_array = line.replace("\n", "").split("\t")

            if (word_array[1] == emotion and word_array[2] == '1'):
                word_list.append(word_array[0])
                
    return word_list
    
sentiment_emotion_lex_word_list = get_affect_presence_list('anger')

def get_sentiment_emotion_feature(word):
    
    score = 0.0
    if word in sentiment_emotion_lex_word_list:
        score = 1.0
    vec_rep = np.array([score])
    
    return non_linear_factor.fit_transform([vec_rep])[0]

#Hashtag Emotion Intensity
hashtag_emotion_lex_file_path = "../lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2.txt"
    
def get_hashtag_emotion_intensity(emotion):
    hastag_intensities = dict()
    
    with open(hashtag_emotion_lex_file_path,encoding='utf-8') as hashtag_emotion_lex_file:
        for line in hashtag_emotion_lex_file:
            word_array = line.replace("\n", "").split("\t")
            if (word_array[0] == emotion):
                hastag_intensities[word_array[1]] = float(word_array[2])
                
    return hastag_intensities
    
hashtag_emotion_intensities = get_hashtag_emotion_intensity(emotion)
 
def get_hashtag_emotion_vector(word):
    score = 0.0    
    if word in hashtag_emotion_intensities.keys():
        score = float(hashtag_emotion_intensities[word])
        
    vec_rep = np.array([score])
            
    return non_linear_factor.fit_transform([vec_rep])[0]

#Emoticon Sentiment Lexicon
emoticon_lexicon_unigrams_file_path = "../lexicons/Emoticon-unigrams.txt"
    
emoticon_lexicon_unigrams = dict()

def get_emoticon_lexicon_unigram_dict():
    with open(emoticon_lexicon_unigrams_file_path,encoding='utf-8') as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_lexicon_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return emoticon_lexicon_unigrams

emoticon_lexicon_unigram_dict = get_emoticon_lexicon_unigram_dict()

def get_unigram_sentiment_emoticon_lexicon_vector(word):
    
    vec_rep = np.zeros(3)
    if word in emoticon_lexicon_unigram_dict.keys():
        vec_rep = emoticon_lexicon_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

#Emoticon Sentiment Aff-Neg Lexicon
emoticon_afflex_unigrams_file_path ="../lexicons/Emoticon-AFFLEX-NEGLEX-unigrams.txt"
    
emoticon_afflex_unigrams = dict()

def get_emoticon_afflex_unigram_dict():
    with open(emoticon_afflex_unigrams_file_path,encoding='utf-8') as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_afflex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return emoticon_afflex_unigrams

emoticon_afflex_unigram_dict = get_emoticon_afflex_unigram_dict()

def get_unigram_sentiment_emoticon_afflex_vector(word):
    word = 'love'
    vec_rep = np.zeros(3)
    if word in emoticon_afflex_unigram_dict.keys():
        vec_rep = emoticon_afflex_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]
                                           
'''
bb = non_linear_factor.fit_transform([vec_rep])[0]
cc = scale(np.array(bb))
'''                                     

#Hashtag Sentiment Aff-Neg Lexicon
hashtag_affneglex_unigrams_file_path = "../lexicons/HS-AFFLEX-NEGLEX-unigrams.txt"
    
hashtag_affneglex_unigrams = dict()

def get_hashtag_affneglex_unigram_dict():
    with open(hashtag_affneglex_unigrams_file_path,encoding='utf-8') as hashtag_sent_lex_file:
        for line in hashtag_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            hashtag_affneglex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return hashtag_affneglex_unigrams

hashtag_affneglex_unigram_dict = get_hashtag_affneglex_unigram_dict()

def get_unigram_sentiment_hashtag_affneglex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in hashtag_affneglex_unigram_dict.keys():
        vec_rep = hashtag_affneglex_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

#Hashtag Sentiment Lexicon
hash_sent_lex_unigrams_file_path = "../lexicons/HS-unigrams.txt"

def get_hash_sent_lex_unigram_dict():
    
    hash_sent_lex_unigrams = dict()
    with open(hash_sent_lex_unigrams_file_path,encoding='utf-8') as hash_sent_lex_file:
        for line in hash_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            if clean_str(word_array[0]):
                hash_sent_lex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])
    
    return hash_sent_lex_unigrams

hash_sent_lex_unigram_dict = get_hash_sent_lex_unigram_dict()

def get_unigram_sentiment_hash_sent_lex_vector(word):
    
    vec_rep = np.zeros(3)
    if word in hash_sent_lex_unigram_dict.keys():
        vec_rep = hash_sent_lex_unigram_dict[word]
        
    return non_linear_factor.fit_transform([vec_rep])[0]

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

#################   Reading & Vectorizing Data   #######################
def is_active_vector_method(string):
    return int(string)

#==============================================================================
# 4    
#==============================================================================
def tweetToSWNVector(word):
    vec = np.zeros(3)
    pos_score, neg_score, obj_score = 0, 0, 0
    l = list(swn.senti_synsets('21'))
    try:
        pos_score += l[0].pos_score()
        neg_score -= l[0].neg_score()
        obj_score += l[0].obj_score()              
    except IndexError:
        pos_score += 0.0
        neg_score -= 0.0
        obj_score += 0.0                       
    
    vec[0], vec[1], vec[2] = pos_score, neg_score, obj_score
    return vec
    

#==============================================================================
#   5
#==============================================================================
bingliu_pos = open('../lexicons/BingLiu_positive-words.txt',encoding='utf-8').readlines()
bingliu_neg = open("../lexicons/BingLiu_negative-words.txt",encoding='utf-8').readlines()
pos_words = []
neg_words = []

for line in bingliu_pos:
    pos_words.append(line.replace("\n", ""))
for line in bingliu_neg:
    neg_words.append(line.replace("\n", ""))       
            
def tweetToBingLiuVector(word):
    vec = np.zeros(1)
    if word in neg_words:
        vec = np.array(-1)
    if word in pos_words:
        vec = np.array(-1)
    return vec

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
 

def learn_unigram_word_embedding(word):    
    word_feature_embedding_dict = dict()
    
    '''Pre-trained Word embeddings'''

    index = 0
    word_feature_embedding_dict[index] = get_word2vec_embedding(word, wv_model_0, glove_dimensions)
    
    index = 1
    word_feature_embedding_dict[index] = get_word2vec_embedding(word, wv_model_1, swm_dimensions)
      
    '''NRC Emotion Intensity Lexicon'''
    index = 2
    word_feature_embedding_dict[index] = get_emo_int_vector(word)

    '''WordNet'''
    index = 3
    word_feature_embedding_dict[index] = get_sentiwordnetscore(word)

    '''NRC Sentiment Lexica'''
    index = 4
    word_feature_embedding_dict[index] = get_sentiment_emotion_feature(word)
      
    index = 5
    word_feature_embedding_dict[index] = get_unigram_sentiment_emoticon_lexicon_vector(word)

    index = 6
    word_feature_embedding_dict[index] = get_unigram_sentiment_emoticon_afflex_vector(word)

    '''NRC Hashtag Lexica'''
    index = 7
    word_feature_embedding_dict[index] = get_hashtag_emotion_vector(word)

    index = 8
    word_feature_embedding_dict[index] = get_unigram_sentiment_hash_sent_lex_vector(word)

    index = 9
    word_feature_embedding_dict[index] = get_unigram_sentiment_hashtag_affneglex_vector(word)

    '''Emoji Polarities'''
    index = 10
    word_feature_embedding_dict[index] = get_emoji_intensity(word)
    
    '''Depeche Mood'''
    index = 11
    word_feature_embedding_dict[index] = get_depeche_mood_vector(word)

    index = 12
    word_feature_embedding_dict[index] = tweetToSWNVector(word)

    index = 13
    word_feature_embedding_dict[index] = tweetToBingLiuVector(word)
    
    '''valence dict'''
    index = 14
    word_feature_embedding_dict[index] = tweetToValenceVector(word)
    
    return word_feature_embedding_dict
     
    
def get_unigram_embedding(word, word_embedding_dict, bin_string):      
    word_feature_embedding_dict = word_embedding_dict[word]
    final_embedding = np.array([])
    
    for i in range(15):
        if is_active_vector_method(bin_string[i]):
            final_embedding = np.append(final_embedding, word_feature_embedding_dict[i])
    
    return final_embedding

'''
import os
os.getcwd()
os.chdir("F:\\2018_SemEval\\SemEval2018\\src\\EI_REG")  
'''

unigram_feature_string = "101111111111010" 

training_tweets = read_training_data(training_data_file_path)
dev_tweets = read_training_data(dev_set_path)


score_train = list()
tweet_train = list()
for tweet in training_tweets:
    tweet_train.append(tweet.text)
    score_train.append(float(tweet.intensity))

for tweet in dev_tweets:
    tweet_train.append(tweet.text)
    score_train.append(float(tweet.intensity))

print('len(tweet_train): %s' % len(tweet_train))
score_train = np.asarray(score_train)



raw_test_tweets = read_training_data_verbatim(test_data_file_path)
test_tweets = read_training_data(test_data_file_path)

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
# with open(word_embeddings_path, 'wb') as word_embeddings_file:
#    pickle.dump(embedding_info, word_embeddings_file)
              
embeddings_index = embedding_info[0]
MAX_SEQUENCE_LENGTH = embedding_info[1]

EMBEDDING_DIM = len(get_unigram_embedding("glad", embedding_info[0], unigram_feature_string))
print('MAX_SEQUENCE_LENGTH: %s' % MAX_SEQUENCE_LENGTH) 
print('EMBEDDING_DIM: %s' % EMBEDDING_DIM)                

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


word_embedding_matrix = list()
word_embedding_matrix.append(np.zeros(EMBEDDING_DIM))

for word in sorted(word_indices, key=word_indices.get):
    embedding_features = get_unigram_embedding(word, embedding_info[0], unigram_feature_string)    
    word_embedding_matrix.append(embedding_features)

word_embedding_matrix = np.asarray(word_embedding_matrix, dtype='f')
#print('\nword_embedding_matrix.shape: %s' % word_embedding_matrix.shape)

word_embedding_matrix = scale(word_embedding_matrix)

#==============================================================================
#  CNN Implementation in Keras   
#==============================================================================
x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding="pre")
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding="pre")

len(x_train), len(x_test), len(x_train[0])

#shuffle_indices = np.random.permutation(np.arange(len(x_train)))
#x_shuffled = x_train[shuffle_indices]
#y_shuffled = score_train[shuffle_indices]

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
#    model.add(pool_1)    
#    model.add(conv_3)
#    model.add(conv_4)
#    model.add(pool_2)  
#    model.add(conv_5)    
    model.add(lstm_1)
    model.add(lstm_2)
    model.add(lstm_3)
#    model.add(bi_lstm_1)
#    model.add(bi_lstm_2)
#    model.add(bi_lstm_3) 
    
    model.add(dense_1)
    model.add(drop_1)
    model.add(dense_2)

    model.compile(loss='mean_squared_error', optimizer="adam")
    
    return model 
    
nn_model = KerasRegressor(build_fn = get_rnn_model, nb_epoch=12, batch_size=32, verbose=1)
#ml_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
nn_model.fit(x_shuffled, y_shuffled)

y_pred0 = nn_model.predict(x_test)
#y_pred = np.reshape(y_pred, len(y_pred))
print(pearsonr(y_pred0, y_gold))
print(spearmanr(y_pred0, y_gold))




#==============================================================================
#        单层LSTM(Bi-LSTM) ensemble之后的效果
#==============================================================================
tmp_pred = [0,1,2,3,4,5,6,7,8,9]
for i in range(10):    
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = score_train[shuffle_indices]

    def get_rnn_model():    
        model = Sequential()    
        model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word_indices) + 1,
                           weights=[word_embedding_matrix], mask_zero=True,trainable=True))
        model.add(LSTM(256, return_sequences=False))
        model.add(Dropout(0.25))    
         
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))   
        model.compile(loss='mean_absolute_error', optimizer="Adam") 
        return model

    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=30, batch_size=32, verbose=1)
    #nn_model = AdaBoostRegressor(base_estimator=nn_model, n_estimators=10)
    nn_model.fit(x_shuffled, y_shuffled)  
    tmp_pred[i] = nn_model.predict(x_test).tolist()
    #print(pearsonr(tmp_pred[9], y_gold)) 
    
y_pred = list()
for j in range(len(tmp_pred[0])):
    temp_emotion = (tmp_pred[0][j] + tmp_pred[1][j] + tmp_pred[2][j] + \
                    tmp_pred[3][j] + tmp_pred[4][j] + tmp_pred[5][j] + \
                    tmp_pred[6][j] + tmp_pred[7][j] + tmp_pred[8][j] + \
                    tmp_pred[7][j] + tmp_pred[8][j] + tmp_pred[9][j])/10
    y_pred.append(temp_emotion)    
        
print(pearsonr(y_pred, y_gold))    
#print(spearmanr(y_pred, y_gold))    
    
    
    
    
    
  
'''
model = Sequential()    
model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word_indices) + 1,
                           weights=[word_embedding_matrix], mask_zero=True,trainable=True))
model.add(Bidirectional(LSTM(256, return_sequences=False)))
model.add(Dropout(0.25))    
         
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))   
model.compile(loss='mean_squared_error', optimizer="Adam") 
    
shuffle_indices = np.random.permutation(np.arange(len(x_train)))    
x_shuffled = x_train[shuffle_indices]
y_shuffled = score_train[shuffle_indices]

nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=10, batch_size=32, verbose=1)
nn_model.fit(x_shuffled, y_shuffled)

y_pred1 = nn_model.predict(x_test)
print(pearsonr(y_pred1, y_gold))
print(spearmanr(y_pred1, y_gold))
'''


#==============================================================================
#             其他模型: 单CNN
#==============================================================================
print('\nTraing单模型CNN...')
tmp_pred = [0,1,2,3,4,5,6,7,8,9]
for i in range(10):
    def get_rnn_model(): 
        model = Sequential()   
        model.add(embed_1)
        
        model.add(conv_1)
        model.add(conv_2)
        model.add(pool_1)
        
        model.add(conv_3)
        model.add(conv_4)
        model.add(pool_2)
    
        model.add(Dense(128))
        model.add(drop_1)
        model.add(Flatten())
        model.add(dense_2)
    
        model.compile(loss='mean_squared_error', optimizer="adam")
        
        return model 
    
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))  
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = score_train[shuffle_indices]
    
    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=10, batch_size=32, verbose=1)
    nn_model.fit(x_shuffled, y_shuffled)   
    tmp_pred[i] = nn_model.predict(x_test).tolist()

    
y_pred = list()
for j in range(len(tmp_pred[0])):
    temp_emotion = (tmp_pred[0][j] + tmp_pred[1][j] + tmp_pred[2][j] + \
                    tmp_pred[3][j] + tmp_pred[4][j] + tmp_pred[5][j] + \
                    tmp_pred[6][j] + tmp_pred[7][j] + tmp_pred[8][j] + \
                    tmp_pred[7][j] + tmp_pred[8][j] + tmp_pred[9][j])/10
    y_pred.append(temp_emotion)    
        
print(pearsonr(y_pred, y_gold))    
print(spearmanr(y_pred, y_gold))


#==============================================================================
#             两层 BiLSTM  ensemble
#==============================================================================
print('\nTraing两层 BiLSTM...')
tmp_pred = [0,1,2,3,4,5,6,7,8,9]
for i in range(10):
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
        model.compile(loss='mae', optimizer="Adam") 
        return model
        
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_shuffled = x_train[shuffle_indices]
    y_shuffled = score_train[shuffle_indices]
    
    nn_model = KerasRegressor(build_fn=get_rnn_model, nb_epoch=10, batch_size=32, verbose=1)
    nn_model.fit(x_shuffled, y_shuffled)
    tmp_pred[i] = nn_model.predict(x_test).tolist()
         
    #print(pearsonr(tmp_pred[9], y_gold))    
    
y_pred = list()
for j in range(len(tmp_pred[0])):
    temp_emotion = (tmp_pred[0][j] + tmp_pred[1][j] + tmp_pred[2][j] + \
                    tmp_pred[3][j] + tmp_pred[4][j] + tmp_pred[5][j] + \
                    tmp_pred[6][j] + tmp_pred[7][j] + tmp_pred[8][j] + tmp_pred[9][j])/10
    y_pred.append(temp_emotion)    
       
print(pearsonr(y_pred, y_gold))    
#print(spearmanr(y_pred, y_gold))


#==============================================================================
# 结果写入
#==============================================================================
#y_pred =  y_pred.tolist()
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
        emotion + "\t" + \
        str(y_pred[i]) + "\n") 










