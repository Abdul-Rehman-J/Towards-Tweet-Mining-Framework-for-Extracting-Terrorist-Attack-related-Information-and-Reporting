from matplotlib import pyplot as plt
import fastcluster 
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram


from time import time
import os
import json
import sys
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from nltk.corpus import stopwords
from nltk import download
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from nltk.tokenize import TweetTokenizer
import re
import logging
from scipy.cluster.hierarchy import fcluster
import os
from scipy.cluster.hierarchy import cophenet
from dynamicTreeCut import cutreeHybrid
from scipy.cluster.hierarchy import to_tree
import gensim.models.keyedvectors as word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim import corpora, models, similarities
try:
    import gensim
except NameError:
    print('gensim is not installed')

from scipy.cluster.hierarchy import cophenet


  


sys.setrecursionlimit(0x100000)
#configuration file with all details about data, operations etc. 
with open('configuration/config.json', 'r') as f:
    config = json.load(f)

features = config['DEFAULT']['FEATURES'] 
distance_calculator = config['DEFAULT']['DISTANCE'] 
path = config['DEFAULT']['DATA_PATH']
result_path = config['DEFAULT']['RESULT_PATH']
#path = './data/testdata/'
ARIcsv=""
puritycsv=""
sillcsv = ""
mutualcsv=""
folkcsv=""
stop_words = stopwords.words('english')
stop_words.append('rt')
labels_true = []
np.set_printoptions(threshold=np.inf)
# Initialize logging.
logging.basicConfig(filename=features+distance_calculator+'.log',level=logging.ERROR)
np.set_printoptions(precision=5, suppress=True)

#threshhold for clustering
totalWordsFound = 0
euclidean_th = 0.99
cosine_th = 0.96
lsi_cosine_th = 0.34
wmd_word2vec_th = 0.80
wmd_glove_th = 0.80
wmd_fasttext_th = 0.75

tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def initialize_words():
    content = None
    with open('./data/wordslist.txt') as f: # A file containing common english words
        content = f.readlines()
    return [word.rstrip('\n') for word in content]
    
    
wordlist = initialize_words() # to remove stopwords

#function to average all words vectors in a given paragraph
# it is used to convert word2vec into doc2vec
def avg_sentence_vector(words, model, num_features, index2word_set):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    global totalWordsFound    
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
    totalWordsFound += nwords
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec	





# customized distance function	
def mydist(p1, p2):
    diff = p1 - p2
    return pow(np.average(diff),2)
	

    



# this function checks for the hashtag in the sentence and use parse_tag to split 
def parse_sentence(sentence, wordlist):
    new_sentence = "" # output   
    terms = sentence.split(' ')    
    for term in terms:
        if len(term) > 0:
            if term[0] == '#': # this is a hashtag, parse it
                new_sentence += parse_tag(term, wordlist)
            else: # Just append the word
                new_sentence += term
            new_sentence += " "
    return new_sentence 




def parse_tag(term, wordlist):
    words = []
    # Remove hashtag, split by dash
    tags = term[1:].split('-')
    for tag in tags:
        word = find_word(tag, wordlist)    
        while word != None and len(tag) > 0:
            words.append(word)            
            if len(tag) == len(word): # Special case for when eating rest of word
                break
            tag = tag[len(word):]
            word = find_word(tag, wordlist)
    return " ".join(words)




def find_word(token, wordlist):
    i = len(token) + 1
    while i > 1:
        i -= 1
        if token[:i] in wordlist:
            return token[:i]
    return None 


    
def euclidean(sentence_1_avg_vector, sentence_2_avg_vector):
    return distance.euclidean(sentence_1_avg_vector, sentence_2_avg_vector)
    



def readFile(filename):
    with open(filename) as f:
        content = f.readlines()
    return content



def cleantext(line):
    #return ' '.join(re.sub("(@[A-Za-z0-9#]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", line).split())
    line=re.sub(r"http\S+", "", line).rstrip()
    line = re.sub('RT @[\w_]+: ', '', line)
    line = re.sub(r'@[A-Za-z0-9]+','',line)
    return re.sub("[^a-zA-Z#]", " ", line)



#filters out of vocabulary words for word2vec model
def filterVowelsword2vec(alphabet):
    if(alphabet in embed_map.vocab):
        return True
    else:
        return False



#filters out of vocabulary words for glove model       
def filterVowelsglove(alphabet):
    if(alphabet in model.vocab):
        return True
    else:
        return False
        


        
#filters out of vocabulary words for fasttext model               
def filterVowelsfasttext(alphabet):
    if(alphabet in model_fasttext.vocab):
        return True
    else:
        return False


 
#word mover distance using word2vec model 
def wmd_word2vec(sentence_1, sentence_2):
    sentence_1 = sentence_1[0].lower().split()
    sentence_2 = sentence_2[0].lower().split()
    distance = embed_map.wmdistance(sentence_1, sentence_2)
    return distance



#word mover distance using glove model 
def wmd_glove(sentence_1, sentence_2):
    sentence_1 = sentence_1[0].lower().split()
    sentence_2 = sentence_2[0].lower().split()
    distance = model.wmdistance(sentence_1, sentence_2)
    return distance    
    



#word mover distance using fasttext model 
def wmd_fasttext(sentence_1, sentence_2):
    sentence_1 = sentence_1[0].lower().split()
    sentence_2 = sentence_2[0].lower().split()
    distance = model_fasttext.wmdistance(sentence_1, sentence_2)
    return distance 


# this will plot results and also cut dandeogram to get flat clusters    
def plotResults(X, Z, method, labels_true, number_of_cluster, x):
    global ARIcsv
    global puritycsv
    global sillcsv
    global mutualcsv
    global folkcsv
    max_height = np.max(get_heights(Z))
    print("reference heigh is " +str(max_height))
    # percentage of height to cut the dandeogram
    x=0.999
    max_d = max_height *x
    print(max_d)
    
    fig= plt.figure(figsize=(12, 10))
    plt.scatter(X[0], X[1])
    plt.title('Hierarchical Clustering Dendrogram '+method+'_'+ number_of_cluster)
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    #plt.show()
    #plt.figure(figsize=(25, 10))
    #plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('sample index')
    #plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    #fig.savefig(method+'_'+number_of_cluster+'_'+str(x)+'_.png')
    #plt.show()
    #print(Z[-20:,2])
    fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d= max_d,  # plot a horizontal cut-off line
    )
    #plt.show()
    #clusters = fcluster(Z, , criterion='maxclust')
    #clusters = fcluster(Z, max_d, criterion='distance')
    #print(clusters)
    #logging.error('threshhold' + str(x))
    #labels_pred = clusters
    #ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    #print("adjested rand score", str(ARI))
    #logging.error('adjusted rand score' + str(ARI))
    #ARIcsv = ARIcsv+","+str(ARI)
    #PS = purity_score(labels_true, labels_pred)
    #print("purity score" + str(PS))
    #logging.error('purity score'+str(PS))
    #puritycsv = puritycsv+","+str(PS)
    #SC = metrics.silhouette_score(squareform(X) , clusters, metric='precomputed')
    #print("silhouette score", str(SC))
    #logging.error('silhouette score'+str(SC))
    #sillcsv = sillcsv+","+ str(SC)
    #AMIS = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    #print(" adjusted mutual info score", str(AMIS))
    #logging.error(' adjusted mutual info score'+str(AMIS))
    #mutualcsv = mutualcsv+","+ str(AMIS)
    #HS = metrics.fowlkes_mallows_score(labels_true, labels_pred) 
    #print("Fowlkes-Mallows scores score", str(HS))
    #logging.error('Fowlkes-Mallows scores score'+ str(HS))
    #folkcsv = folkcsv+","+ str(HS)
    
    #if not os.path.exists(path+"results/"):
    #    os.makedirs(path+"results/")
    #for cluster, text in zip(clusters, filtered_documents):
     #   f=open(path+"results/"+str(cluster)+".txt", "a+")
      #  f.write(text)
      #  f.close()
        
    #for cluster, text in zip(clusters, processed_text):
    #    f=open(path+"results/"+str(cluster)+".txt", "a+")
    #    f.write(text)
    #    f.close()   
        
        

    
    clusters = cutreeHybrid(Z, X, deepSplit = True)
    print(clusters["labels"])
    labels_pred = clusters["labels"]
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    print("dynamic", ARI)
    logging.error('dynamic adjusted rand score' + str(ARI))
    ARIcsv = ARIcsv+","+str(ARI)
    PS = purity_score(labels_true, labels_pred)
    print("dynamic purity score"+ str(PS))
    logging.error('dynamic purity score'+str(PS))
    puritycsv = puritycsv+","+str(PS)
    SC = metrics.silhouette_score(squareform(X) , labels_pred, metric='precomputed')
    print("dynamic silhouette score", str(SC))
    logging.error('dynamic silhouette score'+str(SC))
    sillcsv = sillcsv+","+ str(SC)

    AMIS = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print("dynamic adjusted mutual info score", str(AMIS))
    logging.error('dynamic adjusted mutual info score'+str(AMIS))
    mutualcsv = mutualcsv+","+ str(AMIS)
    HS = metrics.fowlkes_mallows_score(labels_true, labels_pred) 
    print("dynamic Fowlkes-Mallows scores score", str(HS))
    logging.error('dynamic Fowlkes-Mallows scores score' + str(HS))
    folkcsv = folkcsv+","+ str(HS)
    if not os.path.exists(path+"results_dynamic/"):
        os.makedirs(path+"results_dynamic/")
    for cluster, text in zip(labels_pred, filtered_documents):
        f=open(path+"results_dynamic/"+str(cluster)+".txt", "a+")
        f.write(text)
        f.close()
    #sentence = ""   
    #j=1
    #for cluster, text in zip(labels_pred, processed_text):
    #    f=open(path+"results_dynamic/"+str(cluster)+".csv", "a+")
    #    sentence  = "Sentence: "+str(j)
    #    for token in text:
    #       sentence = sentence+","+token+",O,O\n"
    #    f.write(sentence)
    #    f.close()
    #    j=j+1
    
    
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
   

def get_heights(Z):
    #python verison of R's dendro$height
    #height = np.zeros(len(dendro["dcoord"]))
    
    #for i, d in enumerate(dendro["dcoord"]):
        #height[i] = d[1]
    clusternode = to_tree(Z, True)
    height = np.array([c.dist for c in clusternode[1] if c.is_leaf() != True])
    return(height)



def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    ddata = dendrogram(*args, **kwargs)
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

   


 

#if features=='word2vec':
    #from gensim.models import Word2Vec
if not os.path.exists(config['DEFAULT']['WORD2VEC_PATH']):
    raise ValueError("SKIP: You need to download the google news model")    
embed_map = word2vec.KeyedVectors.load_word2vec_format('./data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=1000000)

#embed_map = word2vec.KeyedVectors.load_word2vec_format('./data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
print("word2vec loaded")

#if features=='glove' or features =='tfidf':
glove_input_file = config['DEFAULT']['GLOVE_PATH']
word2vec_output_file = glove_input_file+'.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
# load the Stanford GloVe model
filename = glove_input_file+'.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
print("glove loaded")    

#if features=='fasttext':
filename = config['DEFAULT']['FASTTEXT_PATH']
model_fasttext = KeyedVectors.load_word2vec_format(filename, binary=False, limit=1000000)
model_fasttext.save_word2vec_format(filename+".bin", binary=True)
model_fasttext = KeyedVectors.load_word2vec_format(filename+".bin", binary=True)
print("fastetx loaded")

        
files = os.listdir(path)
cluster_counter=1
documents = []
filtered_documents = []
processed_text = []


div= ''
if(features=='word2vec'):
    for name in files:
        if os.path.isfile(path+name):
            counter= 0 
            content = readFile(path+name)
            #print(len(content))
            #print(len(content))
            #content = np.append(content, readFile(filename))
            #print(len(content))
            for line in content:
                tokenized_line = cleantext(line)
                tokenized_line = parse_sentence(tokenized_line, wordlist)
                tokenized_line = tknzr.tokenize(tokenized_line)
                s=tokenized_line
                    
                tokenized_line = [w for w in tokenized_line if w not in stop_words and len(w) >2]
                if(len(tokenized_line)>3):
                    filteredVowels = list(filter(filterVowelsglove, tokenized_line))
                    filteredVowelsword2vec = list(filter(filterVowelsword2vec, tokenized_line))
                    filteredVowelsfastetx = list(filter(filterVowelsfasttext, tokenized_line))
                    #s=tokenized_line
                    tokenized_line = " ".join(tokenized_line)
                    if len(filteredVowels)>1 and len(filteredVowelsword2vec)>1 and len(filteredVowelsfastetx)>1 and [tokenized_line] not in documents :
                    
                        documents.append([tokenized_line])
                        filtered_documents.append(str(cluster_counter)+"\t"+line)
                        processed_text.append(s)
                        counter= counter+1
            div=div+'('+str(len(labels_true))+'_'+str(counter-1+len(labels_true))+')'
            a = [cluster_counter] * counter
            labels_true = labels_true + a
            cluster_counter= cluster_counter+1
        #max_d = wmd_word2vec_th # max_d as in max_distance    
    print(np.shape(documents))
    #documents = np.unique(documents)
    #documents = documents.reshape((documents.shape[0], 1))
    #print(np.shape(documents))
    X=pdist(documents, metric=wmd_word2vec)
    #for method in ['complete', 'average', 'single']:
    for method in ['average']:

        print(method)
        Z = fastcluster.linkage(X, method=method)
        c, coph_dists = cophenet(Z, X)
        logging.error(method+'w2v'+ str(len(files))+div)
        #logging.error(coph_dists)
        #logging.error(Z[:])
        plotResults(X,Z, method+'w2v', labels_true, str(len(files))+div, wmd_word2vec_th)
    f=open(result_path, "a+")
    f.write("word2vec+wmd+ARI"+ARIcsv+"\n")
    f.write("word2vec+wmd+purity"+puritycsv+"\n")
    f.write("word2vec+wmd+sill"+sillcsv+"\n")
    f.write("word2vec+wmd+MIS"+mutualcsv+"\n")
    f.write("word2vec+wmd+folk"+folkcsv+"\n")
    f.close()
elif features=='glove':
    for name in files:
        if os.path.isfile(path+name):
            counter= 0 
            content = readFile(path+name)
            #print(path)
            #print(len(content))
            #content = np.append(content, readFile(filename))
            #print(len(content))
            for line in content:
                tokenized_line = cleantext(line)
                tokenized_line = parse_sentence(tokenized_line, wordlist)
                tokenized_line = tknzr.tokenize(tokenized_line)
                s=tokenized_line
                    
                tokenized_line = [w for w in tokenized_line if w not in stop_words and len(w) >2]

                #line= [w for w in line if w not in stop_words and len(w) >1]
                #only use those sentences whose non stop words are greater than 0
                if(len(tokenized_line)>3) :
                    filteredVowels = list(filter(filterVowelsglove, tokenized_line))
                    filteredVowelsword2vec = list(filter(filterVowelsword2vec, tokenized_line))
                    filteredVowelsfastetx = list(filter(filterVowelsfasttext, tokenized_line))
                    #s=tokenized_line
                    tokenized_line = " ".join(tokenized_line)
                    if len(filteredVowels)>1 and len(filteredVowelsword2vec)>1 and len(filteredVowelsfastetx)>1 and [tokenized_line] not in documents :
                    
                    #X = np.append(X, [avg_vec], axis=0)
                        documents.append([tokenized_line])
                        filtered_documents.append(str(cluster_counter)+"\t"+line)
                        processed_text.append(s)
                        counter= counter+1
            div=div+'('+str(len(labels_true))+'_'+str(counter-1+len(labels_true))+')'
            a = [cluster_counter] * counter
            labels_true = labels_true + a
            cluster_counter= cluster_counter+1
    max_d = wmd_glove_th # max_d as in max_distance    
    print(np.shape(documents))
    X=pdist(documents, metric=wmd_glove)
    #for method in ['complete', 'average', 'single']:
    for method in [ 'average']:
    #for method in ['single']:
        print(method)
        start2 = time()
        Z = fastcluster.linkage(X, method=method)
        end2 = time()
        print(end2 - start2)
        logging.error(method+'glove'+ str(len(files))+div)
        plotResults(X,Z, method+'glove', labels_true, str(len(files))+div, wmd_glove_th)
    f=open(result_path, "a+")
    f.write("glove+wmd+ARI"+ARIcsv+"\n")
    f.write("glove+wmd+purity"+puritycsv+"\n")
    f.write("glove+wmd+sill"+sillcsv+"\n")
    f.write("glove+wmd+MIS"+mutualcsv+"\n")
    f.write("glove+wmd+folk"+folkcsv+"\n")
    f.close()

elif features=='fasttext':
    print(len(files))
    for name in files:
        print(path+name)
        if os.path.isfile(path+name):
            counter= 0 
            content = readFile(path+name)
            #print(len(content))
            #content = np.append(content, readFile(filename))
            #print(len(content))
            for line in content:
                tokenized_line = cleantext(line)
                tokenized_line = parse_sentence(tokenized_line, wordlist)
                tokenized_line = tknzr.tokenize(tokenized_line)
                s=tokenized_line
       
                tokenized_line = [w for w in tokenized_line if w not in stop_words and len(w) >2]

               # line= [w for w in line if w not in stop_words and len(w) >1]
            #file = open(path+name, "r") 
            #for line in file: 
            #line = line.lower().split() 
                #line=  re.sub(r"http\S+", "", line).rstrip()
                #print(line)
                #line=parse_sentence(line, wordlist)
                #line = tknzr.tokenize(line)
                #line= [w for w in line if w not in stop_words and len(w) >1]
                #only use those sentences whose non stop words are greater than 0
                if(len(tokenized_line)>3) :
                    filteredVowels = list(filter(filterVowelsglove, tokenized_line))
                    filteredVowelsword2vec = list(filter(filterVowelsword2vec, tokenized_line))
                    filteredVowelsfastetx = list(filter(filterVowelsfasttext, tokenized_line))
                    tokenized_line = " ".join(tokenized_line)
                    if len(filteredVowels)>1 and len(filteredVowelsword2vec)>1 and len(filteredVowelsfastetx)>1 and [tokenized_line] not in documents :
                    #print(line)
                
                    #avg_vec=avg_sentence_vector(line, model=embed_map, num_features=300, index2word_set=embed_map.wv)
                    #X = np.append(X, [avg_vec], axis=0)
                        documents.append([tokenized_line])
                        filtered_documents.append(str(cluster_counter)+"\t"+line)
                        processed_text.append(s)
                        counter= counter+1
            div=div+'('+str(len(labels_true))+'_'+str(counter-1+len(labels_true))+')'
            a = [cluster_counter] * counter
            labels_true = labels_true + a
            cluster_counter= cluster_counter+1
    #max_d = wmd_fasttext_th # max_d as in max_distance    
    print(np.shape(documents))
    X=pdist(documents, metric=wmd_fasttext)
    #for method in ['complete', 'average', 'single']:
    for method in ['average']:
        Z = fastcluster.linkage(X, method=method)
        c, coph_dists = cophenet(Z, X)
        logging.error(method+'fasttext'+ str(len(files))+div)

        #logging.error(Z[:])
        plotResults(X,Z, method+'fasttext', labels_true, str(len(files))+div, wmd_fasttext_th)
    f=open(result_path, "a+")
    f.write("fasttext+wmd+ARI"+ARIcsv+"\n")
    f.write("fasttext+wmd+purity"+puritycsv+"\n")
    f.write("fasttext+wmd+sill"+sillcsv+"\n")
    f.write("fasttext+wmd+MIS"+mutualcsv+"\n")
    f.write("fasttext+wmd+folk"+folkcsv+"\n")
    f.close()

    

else:
    for name in files:
        if os.path.isfile(path+name):
            counter= 0 
            content = readFile(path+name)
            #print(len(content))
            #content = np.append(content, readFile(filename))
            #print(len(content))
            for line in content:
                tokenized_line = cleantext(line)
                tokenized_line = parse_sentence(tokenized_line, wordlist)
                tokenized_line = tknzr.tokenize(tokenized_line)
                tokenized_line = [w for w in tokenized_line if w not in stop_words and len(w) >2]
            #file = open(path+name, "r") 
            #for line in file: 
            #line = line.lower().split() 
                #line=  re.sub(r"http\S+", "", line).rstrip()
                #print(line)
                #line = parse_sentence(line, wordlist)
                #line = tknzr.tokenize(line)
                #line = [w for w in line if w not in stop_words and len(w) >1]
                #str1 = ''.join(line)
                #logging.error(str(counter))
                #logging.error(str(line))
                #only use those sentences whose non stop words are greater than 0
                if len(tokenized_line)>3:
                    filteredVowels = list(filter(filterVowelsglove, tokenized_line))
                    filteredVowelsword2vec = list(filter(filterVowelsword2vec, tokenized_line))
                    filteredVowelsfastetx = list(filter(filterVowelsfasttext, tokenized_line))
                    if len(filteredVowels)>1 and len(filteredVowelsword2vec)>1 and len(filteredVowelsfastetx)>1 and tokenized_line not in documents :
                    #avg_vec=avg_sentence_vector(line, model=embed_map, num_features=300, index2word_set=embed_map.wv)
                    #X = np.append(X, [avg_vec], axis=0)
                        documents.append(tokenized_line)
                        #print("adding doc")
                        filtered_documents.append(line)
                        counter= counter+1
            div=div+'('+str(len(labels_true))+'_'+str(counter-1+len(labels_true))+')'

            labels_true = labels_true + ([cluster_counter] * counter)
            cluster_counter= cluster_counter+1
            print(np.shape(documents))
    if features=='tfidf':
        print('this is for tfidf')
        print('creating dictionary')
        
        dictionary = corpora.Dictionary(documents)
        print('dictionary length', len(dictionary))
        #dictionary.save('/tmp/deerwester.dict')
        X = np.empty((0, len(dictionary)))
        corpus = [dictionary.doc2bow(text) for text in documents]
        #corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
        tfidf_model = models.TfidfModel(corpus, normalize=True) # step 1 -- initialize a model
        # corpus tf-idf
        print('converting corpus to tfidf')
        vector = tfidf_model[corpus] 
        
        #print(vector)
        #vector = gensim.models.tfidfmodel.smartirs_normalize(, 'c', True)
        #print(vector)
        for doc in vector:
            d = [0] * len(dictionary)
            for index, value in doc :
                d[index]  = value
            X = np.append(X, [d], axis=0)
            #print(len(X))
        #corpus_tfidf = tfidf[corpus]
       
        #f_m= gensim.matutils.corpus2dense(corpus, 225)
        #print (f_m)
        
    if features=='lsi':
        dictionary = corpora.Dictionary(documents)
        dictionary.save('/tmp/deerwester.dict')
        X= np.empty((0, len(dictionary)))
        corpus = [dictionary.doc2bow(text) for text in documents]
        corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
        lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=200)
        vector = lsi_model[corpus]  # vectorize input copus in BoW format
        #tfidf_model = models.TfidfModel(corpus) # step 1 -- initialize a model
        # corpus tf-idf
        #vector = tfidf_model[corpus] 
        
        for doc in vector:
            ds = []
            d = [0] * len(dictionary)
            for index, value in doc :
                d[index]  = value
            #ds.append(d)
            X = np.append(X, [d], axis=0)
        #corpus_tfidf = tfidf[corpus]
       
        #f_m= gensim.matutils.corpus2dense(corpus, 225)
        #print (f_m)

        
    if features=='rp':
        dictionary = corpora.Dictionary(documents)
        dictionary.save('/tmp/deerwester.dict')
        X= np.empty((0, len(dictionary)))
        corpus = [dictionary.doc2bow(text) for text in documents]
        corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
        rp_model = models.RpModel(corpus, id2word=dictionary, num_topics=200)
        vector = rp_model[corpus]  # vectorize input copus in BoW format
        #tfidf_model = models.TfidfModel(corpus) # step 1 -- initialize a model
        # corpus tf-idf
        #vector = tfidf_model[corpus] 
        
        for doc in vector:
            ds = []
            d = [0] * len(dictionary)
            for index, value in doc :
                d[index]  = value
            #ds.append(d)
            X = np.append(X, [d], axis=0)
        #corpus_tfidf = tfidf[corpus]
       
        #f_m= gensim.matutils.corpus2dense(corpus, 225)
        #print (f_m)
    

    



# generate the linkage matrix
    if(distance_calculator == 'euclidean'):
        print('distance matrix')
        print('hi')
        X=pdist(X, metric='euclidean')
        #for method in ['complete', 'average', 'single']:
        for method in [ 'average']:
            print(method)
            print(np.shape(X))
            Z = fastcluster.linkage(X, method=method)
            #max_d = euclidean_th # max_d as in max_distance
            logging.error(method+'euclidean'+ str(len(files))+div)
            #logging.error(Z[:])
            plotResults(X,Z, method+'euclidean', labels_true, str(len(files))+div, euclidean_th)
        print("openin file "+ result_path)    
        f=open(result_path, "a+")
        f.write("tf_idf+euclidean+ARI"+ARIcsv +"\n")
        f.write("tf_idf+euclidean+purity"+puritycsv+"\n")
        f.write("tf_idf+euclidean+sill"+sillcsv+"\n")
        f.write("tf_idf+euclidean+MIS"+mutualcsv+"\n")
        f.write("tf_idf+euclidean+folk"+folkcsv+"\n")
        f.close()

    elif(distance_calculator == 'cosine' and features == 'tfidf'):
        print('distance matrix')
        print(np.shape(X))
        X=pdist(X, metric='cosine')
        #for method in [ 'complete', 'average', 'single']:
        for method in ['average']:
            print(method)
            print(np.shape(X))
            Z = fastcluster.linkage(X, method=method)
            #max_d = euclidean_th # max_d as in max_distance
            logging.error(method+'cosine'+ str(len(files))+div)
            #logging.error(Z[:])
            plotResults(X,Z, method+'cosine', labels_true, str(len(files))+div, cosine_th)
        f=open(result_path, "a+")
        f.write("tf_idf+cosine+ARI"+ARIcsv+"\n")
        f.write("tf_idf+cosine+purity"+puritycsv+"\n")
        f.write("tf_idf+cosine+sill"+sillcsv+"\n")
        f.write("tf_idf+cosine+MIS"+mutualcsv+"\n")
        f.write("tf_idf+cosine+folk"+folkcsv+"\n")
        f.close()
        
    
     

