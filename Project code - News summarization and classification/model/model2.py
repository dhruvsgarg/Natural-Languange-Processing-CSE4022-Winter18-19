import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.externals import joblib
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import networkx as nx
from nltk.cluster.util import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

app = Flask(__name__, template_folder='./template')


word_embeddings = {}
glove_file = '/home/dhruvsgarg/NLPProject/BBC-Dataset-News/model/glove.6B.100d.txt'
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)

glove_model = KeyedVectors.load_word2vec_format(tmp_file)

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    #string = re.encode('utf-8').strip()
    return string.strip().lower()

data = pd.read_csv('/home/dhruvsgarg/NLPProject/BBC-Dataset-News/dataset/dataset.csv')
print(type(data['news']))
x = data['news'].tolist()
print(x)
y = data['type'].tolist()

for index,value in enumerate(x):
    #print("processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

print (x)
vect = TfidfVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(x)
print(X)
Y = np.array(y)

print("no of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print("train size:", X_train.shape)
print("test size:", X_test.shape)

model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print("Confusion Matrix:\n", c_mat)
print("\nKappa: ",kappa)
print("\nAccuracy: ",acc)

#Stoing the model as pickle file
model_file = 'modelv1.sav'
joblib.dump(model, open(model_file, 'wb'))
print("Modle saved to disk..")


#Summary
def sentence_similarity_we(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
  #  print(sent1)
  #  print(sent2)
    
    all_words = list(set(sent1 + sent2))
 
   # vector1 = [0] * len(all_words)
   # vector2 = [0] * len(all_words)
    
    sent1 = " ".join([i for i in sent1 if i not in stopwords])
    sent2 = " ".join([i for i in sent2 if i not in stopwords])
    
    # build the vector for the first sentence
    
    
  
 
    # build the vector for the second sentence
    
    #print(sent1)
    #print ("vector 1 : ",vector1)
    #print(sent2)
    #print("vector2 : ",vector2)
    #print ("END")
    #print(1-cosine_distance(vector1,vector2))
    sent_vector1 = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent1.split()])/(len(sent1.split())+0.001)
    sent_vector2 = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent2.split()])/(len(sent2.split())+0.001)
     
    return cosine_similarity(sent_vector1.reshape(1,100), sent_vector2.reshape(1,100))[0,0]
 
def build_similarity_matrix_we(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity_we(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def sentence_similarity_tfidf(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    #print(sent1)
    #print ("vector 1 : ",vector1)
    #print(sent2)
    #print("vector2 : ",vector2)
    #print ("END")
    #print(1-cosine_distance(vector1,vector2))
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix_tfidf(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity_tfidf(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def sentence_similarity_wmd(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
  #  print(sent1)
  #  print(sent2)
    
    #all_words = list(set(sent1 + sent2))
 
   # vector1 = [0] * len(all_words)
   # vector2 = [0] * len(all_words)
    
    sent1 = " ".join([i for i in sent1 if i not in stopwords])
    sent2 = " ".join([i for i in sent2 if i not in stopwords])
    
    #sent_vector1 = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent1.split()])/(len(sent1.split())+0.001)
    #sent_vector2 = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent2.split()])/(len(sent2.split())+0.001)
     
    return glove_model.wmdistance(sent1, sent2)
 
def build_similarity_matrix_wmd(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity_wmd(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix



@app.route('/')
def student():
   return render_template('input_text.html')

@app.route('/input',methods = ['POST'])
def result():
    if request.method == 'POST':
        result = request.form
        print(type(result))
        print_dict = {}
        
        text = request.form['input article']
        print_dict['input article'] = text
        text = pd.Series(text)
        print(type(text))
        x = text.tolist()
        for index,value in enumerate(x):
            #print("processing data:",index)
            x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
        print(x)
        X = vect.transform(x)
        print(text)
        print(X)
        loaded_model = joblib.load(model_file)
        print("Model loaded")
        y_pred = loaded_model.predict(X)
        print(y_pred)
        print_dict['Predicted class'] = y_pred
    
    #SUMMARY: Using word embedding features    
    f = open('glove.6B.100d.txt','r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    stop_words = stopwords.words('english')
    article = ""
    article = request.form['input article']
    article=str(article)
    print(type(article))
    summarize_text = []
    scores=nx.Graph()
    article = article.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentence_similarity_martix = build_similarity_matrix_we(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)
    for i in range(5):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        #print("Summarize Text: \n", ". ".join(summarize_text))
    print(". ".join(summarize_text))
    print_dict['Article summary [Glove]'] = (". ".join(summarize_text))
    
    #SUMMARY: Using TF-IDF vectorizer features
    stop_words = stopwords.words('english')
    article = request.form['input article']
    summarize_text = []
    scores=nx.Graph()
    article = article.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentence_similarity_martix = build_similarity_matrix_tfidf(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)
    for i in range(4):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        #print("Summarize Text: \n", ". ".join(summarize_text))	 		
    print(". ".join(summarize_text))
        
    print_dict['Article summary [BOW]'] = (". ".join(summarize_text))
    
    #WMD Implementation
    stop_words = stopwords.words('english')
    article = ""
    article = request.form['input article']
    article=str(article)
    print(type(article))
    summarize_text = []
    scores=nx.Graph()
    article = article.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentence_similarity_martix = build_similarity_matrix_wmd(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)
    for i in range(10):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        #print("Summarize Text: \n", ". ".join(summarize_text))
    print(". ".join(summarize_text))
    print_dict['Article summary [WMD]'] = (". ".join(summarize_text))
    
    return render_template("classification_summary.html",result = print_dict)

if __name__ == '__main__':
   app.run(debug = True)
