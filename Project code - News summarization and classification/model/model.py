import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.externals import joblib
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import networkx as nx
from nltk.cluster.util import cosine_distance

app = Flask(__name__, template_folder='./template')

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
def sentence_similarity(sent1, sent2, stopwords=None):
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
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

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
        
        stop_words = stopwords.words('english')
        article = request.form['input article']
        summarize_text = []
        scores=nx.Graph()
        article = article.split(". ")
        sentences = []
        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank_numpy(sentence_similarity_graph)
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)
        for i in range(5):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
            #print("Summarize Text: \n", ". ".join(summarize_text))	 		
        print(". ".join(summarize_text))
        
        print_dict['Article summary [BOW]'] = (". ".join(summarize_text))
        
        return render_template("classification_summary.html",result = print_dict)


if __name__ == '__main__':
   app.run(debug = True)
