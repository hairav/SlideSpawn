import numpy as np
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from math import log
from collections import deque

def cosine(a,b):
    a = np.array(a)
    b = np.array(b)
    c=np.sqrt(np.sum(a**2)*np.sum(b**2))
    if c==0:
        return 0
    return np.sum(a*b)/c

def comm(a,b):
    a = np.array(a)
    b = np.array(b)
    c = 0
    for i in range(len(a)):
        if a[i]!=0 and b[i]!=0:
            c+=1
    return c


def getXnY(testno,both = True):
    #this function calculates either X & Y or only X as a consequence of the boolean value of the variable "both".
    Dataset_path = '../Dataset'
    
    #opening files
    if both:
        t2 = open(Dataset_path+'/'+str(testno)+'/'+str(testno)+'.txt', 'r')
    t1 = open(Dataset_path+'/'+str(testno)+'/'+str(testno)+'p.txt', 'r')
    t3 = open(Dataset_path+'/'+str(testno)+'/title.txt', 'r')
    
    #reading files
    if both:
        ppt = t2.read()
    paper = t1.read()
    title = t3.read()
    
    #loading NLP tools
    ps = PorterStemmer() 
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer_regex = RegexpTokenizer(r'\w+')
    stop = set(stopwords.words('english'))
    grammar = nltk.data.load('grammars/large_grammars/atis.cfg')
    parser = nltk.parse.BottomUpChartParser(grammar)
    
    
    #processing paper
    a=tokenizer.tokenize(paper)
    b=[tokenizer_regex.tokenize(i) for i in a]
    sentences=[' '.join(i) for i in b]
    rm_sentences = [[j for j in sentence.lower().split() if j not in stop] for sentence in sentences]
    stem_sentences = [[ps.stem(w) for w in j] for j in rm_sentences]
    enumerated_sentences=list(enumerate(stem_sentences))
    tf=[{} for i in range(len(stem_sentences))]
    for i in enumerated_sentences:
        for j in i[1]:
            tf[i[0]][j]=tf[i[0]].get(j,0)+1
        for j in tf[i[0]]:
            tf[i[0]][j]/=len(i[1])
    
    if both:
        #processing ppt

        a1=tokenizer.tokenize(ppt)
        b1=[tokenizer_regex.tokenize(i) for i in a1]
        sentences1=[' '.join(i) for i in b1]
        rm_sentences1 = [[j for j in sentence.lower().split() if j not in stop] for sentence in sentences1]
        stem_sentences1 = [[ps.stem(w) for w in j] for j in rm_sentences1]
        enumerated_sentences1=list(enumerate(stem_sentences1))
        tf1=[{} for i in range(len(stem_sentences1))]
        for i in enumerated_sentences1:
            for j in i[1]:
                tf1[i[0]][j]=tf1[i[0]].get(j,0)+1
            for j in tf1[i[0]]:
                tf1[i[0]][j]/=len(i[1])
            
            
    #processing title
    a2=tokenizer.tokenize(title)
    b2=[tokenizer_regex.tokenize(i) for i in a2]
    sentences2=[' '.join(i) for i in b2]
    rm_sentences2 = [[j for j in sentence.lower().split() if j not in stop] for sentence in sentences2]
    stem_sentences2 = [[ps.stem(w) for w in j] for j in rm_sentences2]
    enumerated_sentences2=list(enumerate(stem_sentences2))
    tf2=[{} for i in range(len(stem_sentences2))]
    for i in enumerated_sentences2:
        for j in i[1]:
            tf2[i[0]][j]=tf2[i[0]].get(j,0)+1
        for j in tf2[i[0]]:
            tf2[i[0]][j]/=len(i[1])
            
    #making word dictionary
    d = {}
    for i in tf:
        for j in i:
            d[j] = 1
    if both:
        for i in tf1:
            for j in i:
                d[j] = 1
    for i in tf2:
        for j in i:
            d[j] = 1
    
    #retreiving word list
    words=list(d.keys())
    
    
    #creating sentece vectors for paper, ppt and title
    #paper
    pvec = [[] for i in range(len(enumerated_sentences))]
    for i in enumerated_sentences:
        for j in words:
            pvec[i[0]].append(tf[i[0]].get(j,0))
    
    if both:
        #ppt     

        ppvec = [[] for i in range(len(enumerated_sentences1))]
        for i in enumerated_sentences1:
            for j in words:
                ppvec[i[0]].append(tf1[i[0]].get(j,0))
    #title
    tit = [[] for i in range(len(enumerated_sentences2))]
    for i in enumerated_sentences2:
        for j in words:
            tit[i[0]].append(tf2[i[0]].get(j,0))
    if both:
        #calculating target values
        Y = [0 for i in pvec]
        for i in range(len(pvec)):
            for j in ppvec:
                #print(len(pvec[i]),len(j))
                Y[i] = max(Y[i],cosine(pvec[i],j))
    
    #calculating independent values
    X =[]
    for i in range(len(pvec)):
        t =[]
        t.append(cosine(tit[0],pvec[i]))   #title similarity
        t.append(comm(tit[0],pvec[i]))   #common with title
        t.append(len([0 for j in pvec[i] if j!=0]))     #len of sentence without stop words
        t.append(len(b[i]))        
        if(len(sentences[i])==0): 
            t.append(0)
        else:
            t.append((len(sentences[i]) - len(rm_sentences[i]))/len(sentences[i]))#stop words percentage
        t.append(i/len(pvec))        #sentence position
        nouns = 0
        verbs = 0
        for word,pos in nltk.pos_tag(b[i]):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns+=1
            if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
                verbs+=1
        t.append(nouns)
        t.append(verbs)
        X.append(t)
    if both:
        if len(X)<100:
            return [],[]
        return X,Y
    else:
        return X