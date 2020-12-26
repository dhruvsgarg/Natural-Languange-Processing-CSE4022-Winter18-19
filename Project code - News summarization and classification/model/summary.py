
#!/usr/bin/env python
# coding: utf-8
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
#def read_article(file_name):
    #file = open(file_name, "r")
    #filedata = file.readlines()
    #article = filedata[0].split(". ")
    #sentences = []

#    for sentence in article:
 #       print(sentence)
  #      sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
   # sentences.pop() 
    
    #return sentences
word_embeddings = {}
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    #vector1 = [0] * len(all_words)
    #vector2 = [0] * len(all_words)
    sent1 = " ".join([i for i in sent1 if i not in stopwords])
    sent2 = " ".join([i for i in sent2 if i not in stopwords])
    #sent_vector1=[]
    #sent_vector2=[]
    # build the vector for the first sentence
    #for w in sent1:
    #    if w in stopwords:
    #        continue
    #    vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    #for w in sent2:
    #    if w in stopwords:
    #        continue
    #    vector2[all_words.index(w)] += 1
    #print(sent1)
    #print ("vector 1 : ",vector1)
    #print(sent2)
    #print("vector2 : ",vector2)
    #print ("END")
    #print(1-cosine_distance(vector1,vector2))
    #return 1 - cosine_distance(vector1, vector2)

    sent_vector1 = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent1.split()])/(len(sent1.split())+0.001)
    sent_vector2 = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent2.split()])/(len(sent2.split())+0.001)
     
    return cosine_similarity(sent_vector1.reshape(1,100), sent_vector2.reshape(1,100))[0,0] 

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


'''def generate_summary( top_n=2):
    	x=[]
	#df=pd.read_csv('/home/shantanu/Desktop/BBC-Dataset-News-Classification-master/dataset.csv')
	stop_words = stopwords.words('english')
	
	for i in range(5):
		df=pd.read_csv('/home/shantanu/Desktop/BBC-Dataset-News-Classification-master/dataset.csv')
		j=i*445    	
		df=df.iloc[j:j+445,:]
		print(df)
		#article=df['news'][0]
		for article in df['news']:
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
			for i in range(top_n):
			      summarize_text.append(" ".join(ranked_sentence[i][1]))
			#print("Summarize Text: \n", ". ".join(summarize_text))
			x.append(". ".join(summarize_text))
	
	data = {'summary': x}       
	print(data)
		#df2 = pd.DataFrame(data)
	
	print 'writing csv flie ...'
		#df2.to_csv('/home/shantanu/Desktop/BBC-Dataset-News-Classification-master/dataset.csv', index=False)
	df=pd.read_csv('/home/shantanu/Desktop/BBC-Dataset-News-Classification-master/dataset.csv')
        df['summary'] = x
        df.to_csv('/home/shantanu/Desktop/BBC-Dataset-News-Classification-master/dataset.csv')'''

def single_summary(article):
    f = open('glove.6B.100d.txt','r')	
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	word_embeddings[word] = coefs
    f.close()
    stop_words = stopwords.words('english')
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
    for i in range(2):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        #print("Summarize Text: \n", ". ".join(summarize_text))	 		
    
    print(". ".join(summarize_text)) 
    # Step 1 - Read text anc split it
    # sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    #sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    #sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    #scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    #ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    #for i in range(top_n):
     # summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin




#generate_summary(  2)

art = """News Corp, the media company controlled by Australian billionaire Rupert Murdoch, is eyeing a move into the video games market. According to the Financial Times, chief operating officer Peter Chernin said that News Corp is ""kicking the tires of pretty much all video games companies"". Santa Monica-based Activison is said to be one firm on its takeover list. Video games are ""big business"", the paper quoted Mr Chernin as saying. We ""would like to get into it"". The success of products such as Sony's Playstation, Microsoft's X-Box and Nintendo's Game Cube have boosted demand for video games. The days of arcade classics such as Space Invaders, Pac-Man and Donkey Kong are long gone. Today, games often have budgets big enough for feature films and look to give gamers as real an experience as possible. And with their price tags reflecting the heavy investment by development companies, video games are proving almost as profitable as they are fun. Mr Chernin, however, told the FT that News Corp was finding it difficult to identify a suitable target. ""We are struggling with the gap between companies like Electronic Arts, which comes with a high price tag, and the next tier of companies,"" he explained during a conference in Phoenix, Arizona. ""These may be too focused on one or two product lines."""

art2 = """A day after Uttar Pradesh Chief Minister Yogi Adityanath's controversial virus remark, Indian Union Muslim League (IUML) approached Election Commission seeking action against BJP-NDA leaders including Adityanath. IUML delegation led by national secretary Khorrum A Omer and Advocate Haris Beeran filed a complaint to EC demanding FIR for Model Code of Conduct (MCC) violation. The complaint says Adityanath's 'virus' remark is loathsome allegation having no historical backing and an FIR be registered under IPC 153A. IUML in its complaint also requested EC to initiate punitive measures including disabling of social media accounts of Yogi Adityanath, MS Sirsa, Shefali Vaidya, Koena Mitra, Giriraj Singh, BJP IT cell head Amit Malviya for making 'baseless, malicious' allegations. Tweets and comments from about two dozen handles, which intend to create communal hatred, against IUML are annexed in the complaint. The complaint says "IUML is committed to the Constitutional values of secularism and democracy. Yogi Adityanath has called IUML a virus responsible for the partition of India. The IUML was formed in Madras on March 10, 1948 after the tragic partition of India. The party is committed to the Constitution of India and the Constitutional principles of secularism, fraternity and the unity and integrity of the nation. The IUML has no role in the partition. This has been done intentionally for creating hatred among communities which is punishable under section 153A of the IPC."
"""
print(type(art))
single_summary(art)
