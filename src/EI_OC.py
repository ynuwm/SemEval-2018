import json
import pickle
import re
import html
import gensim

from sklearn import ensemble
import voting_classifier

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, scale

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import sequence
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords

from keras.optimizers import SGD, Adagrad
from keras.layers.convolutional import Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from keras.layers.normalization import BatchNormalization

#########################   GloVe   ######################
def get_word2vec_embedding(word, model, dimensions):
    vec_rep = np.zeros(dimensions)
    if word in model:
        vec_rep = model[word]    
    return vec_rep

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

# Twitter pretrained vectors
wv_model_4 = loadGloveModel("H:\WASSA\glove.42B.300d.txt")

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
    string = re.sub(r"@[A-Za-z0-9_(),!?\'\`]+", "@username", string) # removing any twitter handle mentions
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
    #string = re.sub(r"'", " ", string
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
#####################  Metadata and Class Definitions  ###################
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
                                        array[2], int(array[3][0])))
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
                train_list.append(Tweet(array[0], array[1], array[2], int(array[3][0])))
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

non_linear_factor = PolynomialFeatures(2)
#emotion = "anger" 
#emotion = "fear"  
#emotion = "joy"          
emotion = "sadness"  
 
training_data_file_path = '../dataset/2_EI_OC/EI-oc-En-train/EI-oc-En-'+ emotion + '-train.txt'
predictions_file_path = "../dataset/2_EI_OC/predictions/"
#dev_set_path = "../data/dev/label/" + emotion + ".txt"
test_data_file_path = "../dataset/2_EI_OC/2018-EI-oc-En-dev/2018-EI-oc-En-"+ emotion + "-dev.txt"
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
emoticon_lexicon_bigrams_file_path ="../lexicons/Emoticon-bigrams.txt"
    
emoticon_lexicon_unigrams = dict()
emoticon_lexicon_bigrams = dict()

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

get_depeche_mood_vector("thanks")

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
    


    
def learn_unigram_word_embedding(word):
    
    word_feature_embedding_dict = dict()
    
    '''Pre-trained Word embeddings'''

    index = 0
    word_feature_embedding_dict[index] = get_word2vec_embedding(word, wv_model_4, 300)

    '''NRC Emotion Intensity Lexicon'''
    index = 1
    word_feature_embedding_dict[index] = get_emo_int_vector(word)

    '''WordNet'''
    index = 2
    word_feature_embedding_dict[index] = get_sentiwordnetscore(word)

    '''NRC Sentiment Lexica'''
    index = 3
    word_feature_embedding_dict[index] = get_sentiment_emotion_feature(word)

    index = 4
    word_feature_embedding_dict[index] = tweetToSWNVector(word)

    index = 5
    word_feature_embedding_dict[index] = tweetToBingLiuVector(word)

    '''NRC Hashtag Lexica'''
    index = 6
    word_feature_embedding_dict[index] = get_hashtag_emotion_vector(word)

    index = 7
    word_feature_embedding_dict[index] = get_unigram_sentiment_hash_sent_lex_vector(word)

    index = 8
    word_feature_embedding_dict[index] = get_unigram_sentiment_hashtag_affneglex_vector(word)

    '''Emoji Polarities'''
    index = 9
    word_feature_embedding_dict[index] = get_emoji_intensity(word)
    
    '''Depeche Mood'''
    index = 10
    word_feature_embedding_dict[index] = get_depeche_mood_vector(word)

    return word_feature_embedding_dict
    
    
def get_unigram_embedding(word, word_embedding_dict, bin_string):      
    word_feature_embedding_dict = word_embedding_dict[word]
    final_embedding = np.array([])
    
    for i in range(11):
        if is_active_vector_method(bin_string[i]):
            final_embedding = np.append(final_embedding, word_feature_embedding_dict[i])
    
    return final_embedding


unigram_feature_string = "11111110011" 

training_tweets = read_training_data(training_data_file_path)
#dev_tweets = read_training_data(dev_set_path)


score_train = list()
tweet_train = list()
for tweet in training_tweets:
    tweet_train.append(tweet.text)
    score_train.append(float(tweet.intensity))

y2 = score_train    
print(len(score_train))
y = np.asarray(score_train)

from keras.utils  import np_utils
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
encoder.fit(y)
encoded_Y=encoder.transform(y)
y_train =np_utils.to_categorical(encoded_Y)



raw_test_tweets = read_training_data_verbatim(test_data_file_path)
test_tweets = read_training_data(test_data_file_path)

tweet_test_raw = list()
tweet_test = list()
y_gold = list()

for tweet in raw_test_tweets:
    tweet_test_raw.append(tweet.text)

for tweet in test_tweets:
    tweet_test.append(tweet.text)
    y_gold.append(tweet.intensity)
    

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


encoder=LabelEncoder()
tmp = np.asarray(y_gold)
encoder.fit(tmp)
encoded_tmp = encoder.transform(tmp)
dev_label = np_utils.to_categorical(encoded_tmp)

len(word_indices)

word_embedding_matrix = list()
word_embedding_matrix.append(np.zeros(EMBEDDING_DIM))

for word in sorted(word_indices, key=word_indices.get):
    embedding_features = get_unigram_embedding(word, embedding_info[0], unigram_feature_string)    
    word_embedding_matrix.append(embedding_features)

word_embedding_matrix = np.asarray(word_embedding_matrix, dtype='f')

print(word_embedding_matrix.shape)

#word_embedding_matrix = scale(word_embedding_matrix)
#==============================================================================
#  CNN Implementation in Keras   
#==============================================================================
x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, padding="pre")
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, padding="pre")

len(x_train), len(x_test), len(x_train[0])

from sklearn.model_selection import StratifiedKFold 
from keras.layers import Merge,Convolution1D
seed = 7
np.random.seed(seed)
filter_sizes = (2, 3, 4, 5)
num_filters = 128
embedding_dim = EMBEDDING_DIM

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x_train[shuffle_indices]
y_shuffled = y_train[shuffle_indices] 


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

def graph_model():
    model = Sequential()

    model.add(Embedding(len(word_indices) + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], 
                        input_length=MAX_SEQUENCE_LENGTH, trainable=True))   
    model.add(Dropout(0.25, input_shape=(MAX_SEQUENCE_LENGTH, embedding_dim)))
    model.add(graph)  
    
    model.add(Dense(150))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(4, activation="softmax"))
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    
    
clf1 = KerasClassifier(build_fn=graph_model, verbose=1, batch_size=32, epochs=10)
clf2 = KerasClassifier(build_fn=graph_model, verbose=1, batch_size=32, epochs=10)
clf3 = KerasClassifier(build_fn=graph_model, verbose=1, batch_size=32, epochs=10)

eclf1 = voting_classifier.VotingClassifier(estimators=[('lstm1', clf1), ('lstm2', clf2), ('lstm3', clf3)], voting='hard')
eclf1 = eclf1.fit(x_shuffled, y_shuffled)


y_pred = eclf1.predict(x_test)    
y_pred = y_pred.tolist()
  
count = 0
for j in range(len(y_pred)):
    if y_pred[j]==y_gold[j]:
        count = count+1
print("dev_accuracy:%.4f%%" % (count/len(y_pred)*100))

from scipy.stats import pearsonr
print('------------------')
print(emotion)
print('Personr',pearsonr(y_pred, y_gold))















 
"""
index = []
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
for train,test in kfold.split(x_train,y2):
    index.append(len(test))
    '''
    print('Train: %s | test: %s' % (train, test))
    print(" ")
    print(len(train),len(test))
    '''
cvscores = []
for train, test in kfold.split(x_train,y2):
    model = Sequential()
    '''
    model.add(Embedding(output_dim=363, input_dim=len(word_indices) + 1,
                   weights=[word_embedding_matrix], mask_zero=True))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    '''
    model.add(Embedding(len(word_indices) + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], 
                        input_length=MAX_SEQUENCE_LENGTH, trainable=True))   
    model.add(Dropout(0.25, input_shape=(MAX_SEQUENCE_LENGTH, embedding_dim)))
    model.add(graph)
   
    
    model.add(Dense(150))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(4, activation="softmax"))
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train[train], y_train[train], nb_epoch=10, batch_size=32, verbose=2)
     
    scores = model.evaluate(x_train[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("mean-cvsore: "+"%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


y_pred = model.predict(x_test)  

count = 0
dev_predict = []
for j in range(len(y_pred)):
    dev_predict.append(list(y_pred[j]).index(max(y_pred[j])))
    if dev_predict[j]==y_gold[j]:
        count = count+1


print("dev_accuracy:%.4f%%" % (count/len(y_pred)*100))

from scipy.stats import pearsonr
print('------------------')
print(emotion)
print('Personr',pearsonr(dev_predict, y_gold))











def end_string(k):
    if k == 0:
        temp=": no anger can be inferred"
    elif k==1:
        temp=": low amount of anger can be inferred"
    elif k==2:
        temp=": moderate amount of anger can be inferred"
    else:
        temp=": high amount of anger can be inferred"
    return temp

with open(predictions_file_path, 'w', encoding='utf-8') as predictions_file:
    for i in range(len(y_pred)):        
        predictions_file.write(
            str(raw_test_tweets[i].id) + "\t" + \
            raw_test_tweets[i].text + "\t" + \
            raw_test_tweets[i].emotion + "\t" + \
            str(dev_predict[i]) + end_string(dev_predict[i]) + "\n") 

"""
















