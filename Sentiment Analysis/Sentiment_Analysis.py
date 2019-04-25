import numpy as np
import pandas as pd
import time

# function to read te text file 
def dataReader():
    with open('datasets/naive_bayes_data.txt','r+',encoding='utf-8') as file:
        dataFile=file.read()
    return dataFile

''' function to process the list of the tweets/comments given in the dataset 
    and separate into two lists one of target variable and one of the text'''
def processList(entries):
    lines=[]
    trgt=[]
    for entry in entries:        
        trgt.append(entry[:entry.find(' ')])
        value=entry[entry.find(' ')+1:]
        
        lines.append( (value[value.find("txt "):]).replace('txt ','') )
    return [lines, trgt]
        
# function that reads the datafile and splits into the separate comments
def makeList(dataFile):
    lines=[]
    entries=dataFile.split('\n')
    for entry in entries:
        entry=entry.lower()
        lines.append(entry[entry.find(' ')+1:])
    np.random.shuffle(lines)
    return lines

# function to cretae 80 20 split in the data file
def testSplit(values):
    num_of_rows = int(len(values) * 0.8)
    train_data =values[:num_of_rows] 
    test_data = values[num_of_rows:]
    return train_data,test_data

''' function to create two dictionarys storing workds as keys and their frequencies
    of occurance in positive and negative classes as their values'''
def makeVocab(lines,target):
    unwanted_chars = ".,-_ )!(?':;&/\* 1234567890 @#$%^+="
    poswordfreq = {}
    negwordfreq={}
    count=0;
    for words in lines: 
        words=words.split(' ')
        for raw_word in words:
            word = raw_word.strip(unwanted_chars)
            if target[count]=='pos':
                if word not in poswordfreq:
                    poswordfreq[word] = 0 
                poswordfreq[word] += 1
            else:
                if word not in negwordfreq:
                    negwordfreq[word] = 0 
                negwordfreq[word] += 1
        count=count+1
    del poswordfreq[""]
    del negwordfreq[""]    
    return poswordfreq,negwordfreq

'''function to find probability of occurance in positive and negative comments'''
def findProbability(wordfreq):
    length=sum(wordfreq.values())
    for key in wordfreq:
        wordfreq[key]=wordfreq.get(key)/length
    return wordfreq
    
''' function to find probability of belonging to and specific class'''
def findclassProbability(target):
    poscount=target.count('pos')/len(target)
    negcount=target.count('neg')/len(target)
    return poscount,negcount

# function that makes predictions using the naive bayes classifier algorithm
def makePrediction(test_lines,posfreq,negfreq,pprob,nprob):
    pred=[]
    for line in test_lines:
        line=line.split(' ')
        posprob=1
        negprob=1
        for word in line:
            if posfreq.get(word)!=None:
                posprob=posprob*posfreq.get(word)
            if negfreq.get(word)!=None:
                negprob=negprob*negfreq.get(word)
        posprob=posprob*pprob
        negprob=nprob*negprob
        if negprob>posprob:
            pred.append('neg')
        else:
            pred.append('pos')
    return pred

# function to calculate error 
def findError(pred,trgt):
    error=0
    for i in range(len(trgt)):
         if pred[i]!=trgt[i]:
                error=error+1
    error=error/len(trgt)*100
    return error
        
# driver to call above functions and print accuracy       
if __name__=='__main__':

    start_time = time.time()

    dataFile=dataReader()
    values=makeList(dataFile)
    train,test=testSplit(values)
    train_lines,train_target=processList(train)
    test_lines,test_target=processList(train)
    posfreq,negfreq=makeVocab(train_lines,train_target)
    posfreq=findProbability(posfreq)
    negfreq=findProbability(negfreq)
    posprob,negprob=findclassProbability(train_target)
    pred=makePrediction(test_lines,posfreq,negfreq,posprob,negprob)
    error=findError(pred,test_target)
    print(error)
    print("Time taken = ",(time.time()-start_time))

